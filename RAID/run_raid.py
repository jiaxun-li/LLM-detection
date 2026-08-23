#!/usr/bin/env python3
"""Run the frozen RAID prepare, score, evaluate, plot, and validate stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_detection.io import atomic_write_json

from RAID.raid_pipeline import (
    STAGES,
    evaluate_stage,
    initial_manifest,
    load_raid_config,
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
    parser.add_argument("--config", default="RAID/config.json")
    parser.add_argument(
        "--stage",
        choices=[*STAGES, "all"],
        default=default_stage or "all",
    )
    parser.add_argument(
        "--data-path",
        help="Local official RAID train.csv/JSONL. Required for Delta full runs.",
    )
    parser.add_argument("--limit-sources", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int)
    parser.add_argument("--skip-binoculars", action="store_true")
    parser.add_argument("--debug-only", action="store_true")
    parser.add_argument(
        "--num-shards",
        type=int,
        help="Deterministic source shards used by independent scorer jobs.",
    )
    parser.add_argument(
        "--index-cache-dir",
        help="Persistent checksum-keyed RAID SQLite cache directory.",
    )
    parser.add_argument(
        "--reuse-index-path",
        help="Explicit completed RAID index beside its source run manifest.",
    )
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Rebuild only this run's RAID SQLite preparation index.",
    )
    return parser.parse_args()


def main(default_stage: str | None = None) -> None:
    args = parse_args(default_stage)
    if args.limit_sources is not None and args.limit_sources < 40:
        raise ValueError(
            "RAID bounded runs require at least 40 sources so all eight domains "
            "can contribute to tuning, calibration, and testing"
        )
    if args.bootstrap_repetitions is not None and args.bootstrap_repetitions < 0:
        raise ValueError("bootstrap repetitions cannot be negative")
    if args.num_shards is not None and args.num_shards < 1:
        raise ValueError("RAID num_shards must be positive")
    workspace = Path(args.workspace).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace / config_path
    config = load_raid_config(config_path)
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
            data_path=args.data_path,
            limit_sources=args.limit_sources,
            bootstrap_repetitions=args.bootstrap_repetitions,
            skip_binoculars=args.skip_binoculars,
            num_shards=args.num_shards,
            index_cache_dir=args.index_cache_dir,
            reuse_index_path=args.reuse_index_path,
        )
    else:
        if args.data_path is not None:
            print(
                f"RAID initialization: hashing input provenance for {args.data_path}",
                flush=True,
            )
        manifest = initial_manifest(
            args.run_id,
            workspace,
            config,
            data_path=args.data_path,
            limit_sources=args.limit_sources,
            bootstrap_repetitions=args.bootstrap_repetitions,
            skip_binoculars=args.skip_binoculars,
            debug_only=args.debug_only,
            num_shards=args.num_shards or 1,
            index_cache_dir=args.index_cache_dir,
            reuse_index_path=args.reuse_index_path,
        )
        print("RAID initialization: input provenance complete", flush=True)
        manifest["config_path"] = str(config_path)
    was_complete = manifest.get("completion_status") == "complete"
    atomic_write_json(manifest_path, manifest)
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    try:
        for stage in stages:
            if args.stage == "all" and stage in manifest["completed_stages"]:
                continue
            if stage == "prepare":
                outputs = prepare_stage(
                    run_dir,
                    config,
                    data_path=args.data_path,
                    limit_sources=args.limit_sources,
                    reset_index=args.reset_index,
                    data_provenance=manifest.get("data_input"),
                    num_shards=int(manifest.get("num_score_shards", 1)),
                    index_cache_dir=manifest.get("index_cache_dir"),
                    reuse_index_path=manifest.get("reuse_index_path"),
                )
            elif stage == "score":
                outputs = score_stage(
                    run_dir, config, skip_binoculars=args.skip_binoculars
                )
            elif stage == "evaluate":
                outputs = evaluate_stage(
                    run_dir,
                    results_dir,
                    config,
                    bootstrap_repetitions=args.bootstrap_repetitions,
                    skip_binoculars=args.skip_binoculars,
                )
            elif stage == "plot":
                outputs = plot_stage(results_dir)
            else:
                outputs = {
                    "validation_report": validate_artifacts(
                        run_dir,
                        results_dir,
                        config,
                        limit_sources=args.limit_sources,
                        bootstrap_repetitions=args.bootstrap_repetitions,
                        skip_binoculars=args.skip_binoculars,
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
        if args.stage == "all"
        or was_complete
        or set(STAGES).issubset(manifest["completed_stages"])
        else f"stage-{args.stage}-complete"
    )
    atomic_write_json(manifest_path, manifest)
    print(f"raid_run_id={args.run_id}")
    print(f"raid_run_dir={run_dir}")
    print(f"raid_results_dir={results_dir}")
    print(f"raid_status={manifest['completion_status']}")


if __name__ == "__main__":
    main()
