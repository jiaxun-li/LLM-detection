#!/usr/bin/env python3
"""Run prepare, score, evaluate, plot, and validate stages for Beemo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_detection.io import atomic_write_json

from Beemo.beemo_pipeline import (
    evaluate_stage,
    initial_manifest,
    load_beemo_config,
    plot_stage,
    prepare_stage,
    run_paths,
    score_stage,
    validate_artifacts,
    validate_resume_manifest,
)


def parse_args(default_stage: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--config", default="Beemo/config.json")
    parser.add_argument(
        "--stage",
        choices=["prepare", "score", "evaluate", "plot", "validate", "all"],
        default=default_stage or "all",
    )
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int)
    parser.add_argument("--skip-binoculars", action="store_true")
    parser.add_argument(
        "--include-granite",
        action="store_true",
        help="Also run the optional Granite 3.3 8B reference scorer.",
    )
    parser.add_argument("--debug-only", action="store_true")
    return parser.parse_args()


def main(default_stage: str | None = None) -> None:
    args = parse_args(default_stage)
    workspace = Path(args.workspace).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace / config_path
    config = load_beemo_config(config_path)
    run_dir, results_dir = run_paths(workspace, config, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume_manifest(
            manifest,
            config,
            run_id=args.run_id,
            limit=args.limit_records,
            skip_binoculars=args.skip_binoculars,
            include_granite=args.include_granite,
        )
        if manifest.get("bootstrap_repetitions_override") != args.bootstrap_repetitions:
            raise ValueError(
                "existing Beemo manifest uses a different bootstrap override; choose a new run id"
            )
    else:
        manifest = initial_manifest(
            args.run_id,
            workspace,
            config,
            limit=args.limit_records,
            skip_binoculars=args.skip_binoculars,
            include_granite=args.include_granite,
            debug_only=args.debug_only,
        )
        manifest["config_path"] = str(config_path)
        manifest["bootstrap_repetitions_override"] = args.bootstrap_repetitions
    was_complete = manifest.get("completion_status") == "complete"
    atomic_write_json(manifest_path, manifest)
    stages = (
        ["prepare", "score", "evaluate", "plot", "validate"]
        if args.stage == "all"
        else [args.stage]
    )
    try:
        for stage in stages:
            if stage == "prepare":
                outputs = prepare_stage(run_dir, config, limit=args.limit_records)
            elif stage == "score":
                outputs = score_stage(
                    run_dir,
                    config,
                    skip_binoculars=args.skip_binoculars,
                    include_granite=args.include_granite,
                )
            elif stage == "evaluate":
                outputs = evaluate_stage(
                    run_dir,
                    results_dir,
                    config,
                    bootstrap_repetitions=args.bootstrap_repetitions,
                    skip_binoculars=args.skip_binoculars,
                    include_granite=args.include_granite,
                )
            elif stage == "plot":
                outputs = plot_stage(results_dir)
            else:
                outputs = {
                    "validation_report": validate_artifacts(
                        run_dir,
                        results_dir,
                        config,
                        skip_binoculars=args.skip_binoculars,
                        include_granite=args.include_granite,
                    )
                }
            manifest["outputs"].update(outputs)
            if stage not in manifest["completed_stages"]:
                manifest["completed_stages"].append(stage)
            manifest["completion_status"] = "running"
            manifest.pop("failure", None)
            atomic_write_json(manifest_path, manifest)
    except Exception as exc:
        manifest["completion_status"] = "failed"
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_write_json(manifest_path, manifest)
        raise
    manifest["completion_status"] = (
        "complete"
        if args.stage == "all" or was_complete
        else f"stage-{args.stage}-complete"
    )
    atomic_write_json(manifest_path, manifest)
    print(f"beemo_run_id={args.run_id}")
    print(f"beemo_run_dir={run_dir}")
    print(f"beemo_results_dir={results_dir}")
    print(f"beemo_status={manifest['completion_status']}")


if __name__ == "__main__":
    main()
