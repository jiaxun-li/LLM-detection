"""Configuration-first entry point for one dataset × target-model experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

from llm_detection.config import load_config, resolved_run_config
from llm_detection.pipeline import (
    evaluate_run,
    prepare_run_data,
    score_run,
    select_source_manifest,
    source_manifest_path,
)
from llm_detection.runtime import build_manifest, update_manifest
from llm_detection.runtime import load_manifest
from llm_detection.io import iter_jsonl


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--dataset", choices=["xsum", "squad", "writingprompts"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--stage",
        choices=["select-sources", "prepare", "score", "evaluate", "all"],
        default="all",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--skip-binoculars", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    config = load_config(workspace / args.config, args.overrides)
    run_id = args.run_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{args.dataset}-{safe_name(args.model)}"
    )
    run_config = resolved_run_config(config, args.dataset, args.model, run_id)
    if args.skip_binoculars:
        run_config["scoring"]["detectors"] = [
            detector
            for detector in run_config["scoring"]["detectors"]
            if detector != "binoculars"
        ]
    source_path = source_manifest_path(config, args.dataset, workspace)
    source_path = select_source_manifest(config, args.dataset, source_path)
    if args.stage == "select-sources":
        print(source_path)
        return

    run_dir = workspace / config["paths"]["runs"] / run_id
    results_dir = workspace / config["paths"]["results"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    outputs = {
        "run_directory": str(run_dir),
        "results_directory": str(results_dir),
    }
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if (
            manifest.get("run_id") != run_id
            or manifest.get("target_model") != args.model
            or manifest.get("dataset", {}).get("name") != args.dataset
        ):
            raise ValueError("existing manifest does not match requested run")
        outputs.update(manifest.get("outputs", {}))
    else:
        manifest = build_manifest(
            run_config,
            workspace,
            {"source_manifest": str(source_path)},
            outputs,
            str(source_path),
        )
        if args.skip_binoculars:
            manifest["explicitly_skipped_detectors"] = ["binoculars"]
    update_manifest(manifest_path, manifest)

    stages = (
        ["prepare", "score", "evaluate"] if args.stage == "all" else [args.stage]
    )
    try:
        for stage in stages:
            if stage == "prepare":
                outputs.update(prepare_run_data(run_config, source_path, run_dir))
                base = next(iter_jsonl(outputs["base_generations"]))
                manifest["dataset"]["resolved_revision"] = base.get(
                    "dataset_resolved_revision"
                )
                manifest["target_model_resolved_revision"] = base.get(
                    "target_model_resolved_revision"
                )
                manifest["target_tokenizer_resolved_revision"] = base.get(
                    "target_tokenizer_resolved_revision"
                )
            elif stage == "score":
                data_path = run_dir / "data.jsonl"
                outputs.update(
                    score_run(
                        run_config,
                        data_path,
                        run_dir,
                        include_binoculars=not args.skip_binoculars,
                    )
                )
                first_target = next(iter_jsonl(outputs["target_scores"]))
                manifest["target_model_resolved_revision"] = first_target.get(
                    "scoring_model_revision"
                )
                manifest["target_tokenizer_resolved_revision"] = first_target.get(
                    "scoring_tokenizer_revision"
                )
                if "binoculars_scores" in outputs:
                    first_pair = next(iter_jsonl(outputs["binoculars_scores"]))
                    manifest["binoculars_resolved_revisions"] = {
                        "performer": first_pair.get(
                            "binoculars_performer_revision"
                        ),
                        "observer": first_pair.get("binoculars_observer_revision"),
                    }
            elif stage == "evaluate":
                outputs["metrics_csv"] = evaluate_run(
                    run_config, run_dir, results_dir
                )
            if stage not in manifest["completed_stages"]:
                manifest["completed_stages"].append(stage)
            update_manifest(
                manifest_path,
                manifest,
                outputs=outputs,
                completion_status="running",
            )
    except Exception as exc:
        update_manifest(
            manifest_path,
            manifest,
            outputs=outputs,
            completion_status="failed",
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    update_manifest(
        manifest_path,
        manifest,
        outputs=outputs,
        completion_status="complete"
        if args.stage == "all"
        else f"stage-{args.stage}-complete",
    )
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
