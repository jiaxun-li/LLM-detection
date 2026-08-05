"""Run the frozen Granite-XSum paired splice-artifact audit."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

from llm_detection.config import load_config, resolved_run_config
from llm_detection.generation import TransformersBackend
from llm_detection.io import atomic_write_json
from llm_detection.pipeline import score_run
from llm_detection.runtime import build_manifest, load_manifest, update_manifest
from llm_detection.splice_audit import (
    AUDIT_CONDITIONS,
    construct_splice_audit_data,
    evaluate_splice_audit,
    generate_alternative_continuations,
    select_test_base_rows,
)
from llm_detection.splice_audit_plot import plot_splice_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-config",
        default="configs/splice_artifact_audit_granite_xsum.json",
    )
    parser.add_argument("--paper-config", default="configs/paper.json")
    parser.add_argument("--source-run-id")
    parser.add_argument("--audit-id", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument(
        "--stage", choices=["prepare", "score", "evaluate", "all"], default="all"
    )
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int)
    parser.add_argument("--skip-binoculars", action="store_true")
    parser.add_argument("--debug-only", action="store_true")
    return parser.parse_args()


def _first_metrics_row(
    path: Path, detector: str | None = None
) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "test" and (
                detector is None or row.get("detector") == detector
            ):
                return row
    suffix = "" if detector is None else f" for detector {detector}"
    raise ValueError(f"no test metrics{suffix} in {path}")


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _validate_source_manifest(manifest: dict[str, Any], source_run_id: str) -> None:
    if manifest.get("run_id") != source_run_id:
        raise ValueError("source manifest run ID disagrees with requested source")
    if manifest.get("completion_status") != "complete":
        raise ValueError("source run is not complete")
    if set(manifest.get("completed_stages", [])) != {
        "prepare",
        "score",
        "evaluate",
    }:
        raise ValueError("source run does not contain all three completed stages")
    if manifest.get("dataset", {}).get("name") != "xsum":
        raise ValueError("splice audit source must be the Granite XSum cell")
    if manifest.get("target_model") != "ibm-granite/granite-3.3-8b-base":
        raise ValueError("splice audit source must use Granite 3.3 8B Base")


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    audit_config_path = workspace / args.audit_config
    audit_config = json.loads(audit_config_path.read_text(encoding="utf-8"))
    if tuple(audit_config["conditions"]) != AUDIT_CONDITIONS:
        raise ValueError("audit condition order differs from the frozen protocol")
    if audit_config.get("decision_thresholds") != {
        "minimum_human_degradation": 0.02,
        "low_artifact_share_upper": 0.3,
        "high_artifact_share_lower": 0.7,
    }:
        raise ValueError("audit decision thresholds differ from the frozen protocol")
    source_run_id = args.source_run_id or audit_config["source_run_id"]
    sample_count = int(args.sample_count or audit_config["sample_count"])
    repetitions = int(
        args.bootstrap_repetitions or audit_config["bootstrap_repetitions"]
    )
    debug_only = bool(args.debug_only or sample_count < audit_config["sample_count"])

    paper_config = load_config(workspace / args.paper_config)
    source_run_dir = workspace / paper_config["paths"]["runs"] / source_run_id
    source_results_dir = workspace / paper_config["paths"]["results"] / source_run_id
    source_manifest_path = source_run_dir / "manifest.json"
    source_base_path = source_run_dir / "base_generations.jsonl"
    source_metrics_path = source_results_dir / "metrics.csv"
    for required in (source_manifest_path, source_base_path, source_metrics_path):
        if not required.is_file():
            raise FileNotFoundError(f"required source artifact is missing: {required}")
    source_manifest = load_manifest(source_manifest_path)
    _validate_source_manifest(source_manifest, source_run_id)
    selected = select_test_base_rows(
        source_base_path,
        sample_count,
        int(audit_config["selection_seed"]),
    )
    first_base = selected[0]
    model_revision = first_base.get("target_model_resolved_revision")
    tokenizer_revision = first_base.get("target_tokenizer_resolved_revision")
    if not model_revision or not tokenizer_revision:
        raise ValueError("source cell does not record resolved target revisions")
    if any(
        row.get("target_model_resolved_revision") != model_revision
        or row.get("target_tokenizer_resolved_revision") != tokenizer_revision
        for row in selected
    ):
        raise ValueError("selected source rows disagree on resolved target revisions")
    run_config = resolved_run_config(
        paper_config,
        "xsum",
        "ibm-granite/granite-3.3-8b-base",
        args.audit_id,
    )
    run_config["experiment_name"] = audit_config["audit_name"]
    run_config["result_label"] = audit_config["result_label"]
    run_config["debug_only"] = debug_only
    run_config["target_model_revision"] = model_revision
    run_config["target_tokenizer_revision"] = tokenizer_revision
    run_config["generation"]["device_map"] = "auto"
    run_config["scoring"]["device_map"] = "auto"
    pair = run_config["models"]["binoculars"]
    if not args.skip_binoculars:
        binoculars_metrics = _first_metrics_row(
            source_metrics_path, "binoculars"
        )
        pair["performer_revision"] = binoculars_metrics.get(
            "binoculars_performer_revision"
        )
        pair["observer_revision"] = binoculars_metrics.get(
            "binoculars_observer_revision"
        )
        if not (pair.get("performer_revision") and pair.get("observer_revision")):
            raise ValueError("source metrics do not pin both Binoculars revisions")
    if args.skip_binoculars:
        run_config["scoring"]["detectors"] = [
            detector
            for detector in run_config["scoring"]["detectors"]
            if detector != "binoculars"
        ]

    audit_run_dir = (
        workspace
        / paper_config["paths"]["runs"]
        / "splice_artifact_audits"
        / args.audit_id
    )
    audit_results_dir = (
        workspace
        / paper_config["paths"]["results"]
        / "splice_artifact_audits"
        / args.audit_id
    )
    audit_run_dir.mkdir(parents=True, exist_ok=True)
    audit_results_dir.mkdir(parents=True, exist_ok=True)
    alternative_path = audit_run_dir / "alternative_generations.jsonl"
    data_path = audit_run_dir / "data.jsonl"
    manifest_path = audit_run_dir / "manifest.json"
    outputs = {
        "run_directory": str(audit_run_dir),
        "results_directory": str(audit_results_dir),
        "alternative_generations": str(alternative_path),
        "data": str(data_path),
    }
    protocol = {
        **audit_config,
        "source_run_id": source_run_id,
        "sample_count": sample_count,
        "bootstrap_repetitions": repetitions,
        "debug_only": debug_only,
        "binoculars_included": not args.skip_binoculars,
    }
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest.get("audit_protocol") != protocol:
            raise ValueError(
                "existing audit manifest uses different settings; choose a new audit ID"
            )
        outputs.update(manifest.get("outputs", {}))
    else:
        manifest = build_manifest(
            run_config,
            workspace,
            {
                "source_run_manifest": str(source_manifest_path),
                "source_base_generations": str(source_base_path),
                "source_frozen_metrics": str(source_metrics_path),
                "audit_config": str(audit_config_path),
            },
            outputs,
            source_manifest.get("source_sample_manifest", str(source_base_path)),
        )
        manifest["audit_protocol"] = protocol
        manifest["source_run_id"] = source_run_id
        manifest["selected_sample_ids"] = [
            str(row["sample_id"]) for row in selected
        ]
    update_manifest(manifest_path, manifest)

    stages = ["prepare", "score", "evaluate"] if args.stage == "all" else [args.stage]
    try:
        for stage in stages:
            if stage == "prepare":
                backend = TransformersBackend(
                    run_config["target_model"],
                    run_config.get("target_model_revision"),
                    run_config.get("target_tokenizer_revision"),
                    run_config["generation"],
                )
                alternative_count = generate_alternative_continuations(
                    selected,
                    alternative_path,
                    backend,
                    int(audit_config["alternative_generation_seed"]),
                    int(run_config["generation"].get("checkpoint_examples", 96)),
                    int(run_config["generation"].get("min_new_tokens", 0)),
                )
                data_count = construct_splice_audit_data(
                    args.audit_id,
                    source_run_id,
                    selected,
                    alternative_path,
                    data_path,
                    backend.tokenizer,
                    [float(value) for value in audit_config["ratios"]],
                    int(audit_config["corruption_seed"]),
                )
                atomic_write_json(
                    audit_run_dir / "prepare.complete.json",
                    {
                        "selected_sources": sample_count,
                        "alternative_generation_rows": alternative_count,
                        "data_rows": data_count,
                    },
                )
                del backend
                _release_cuda()
            elif stage == "score":
                outputs.update(
                    score_run(
                        run_config,
                        data_path,
                        audit_run_dir,
                        include_binoculars=not args.skip_binoculars,
                    )
                )
                _release_cuda()
            elif stage == "evaluate":
                binoculars_path = audit_run_dir / "binoculars_scores.jsonl"
                summary = evaluate_splice_audit(
                    audit_run_dir / "target_scores.jsonl",
                    source_metrics_path,
                    audit_results_dir,
                    [float(value) for value in audit_config["ratios"]],
                    repetitions,
                    int(audit_config["bootstrap_seed"]),
                    int(audit_config["boundary_radius_tokens"]),
                    binoculars_path if binoculars_path.exists() else None,
                )
                outputs.update(
                    {
                        "metrics_csv": str(audit_results_dir / "metrics.csv"),
                        "artifact_share_csv": str(
                            audit_results_dir / "artifact_share.csv"
                        ),
                        "boundary_diagnostics_csv": str(
                            audit_results_dir / "boundary_diagnostics.csv"
                        ),
                        "summary_json": str(audit_results_dir / "summary.json"),
                    }
                )
                outputs["plots"] = plot_splice_audit(
                    audit_results_dir / "metrics.csv",
                    audit_results_dir / "artifact_share.csv",
                    audit_results_dir,
                )
                atomic_write_json(
                    audit_results_dir / "evaluate.complete.json", summary
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
        completion_status=(
            "complete" if args.stage == "all" else f"stage-{args.stage}-complete"
        ),
    )
    print(f"audit_manifest={manifest_path}")
    print(f"audit_results={audit_results_dir}")


if __name__ == "__main__":
    main()
