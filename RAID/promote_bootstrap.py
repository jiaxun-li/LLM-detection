#!/usr/bin/env python3
"""Re-evaluate a completed RAID score pack with the frozen bootstrap count."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_core.infrastructure.io import atomic_write_json
from RAID.raid_pipeline import (
    evaluate_stage,
    load_raid_config,
    plot_stage,
    run_paths,
    validate_artifacts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(workspace: Path) -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={workspace.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _artifact_record(path: Path, *, hash_contents: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required RAID promotion artifact is missing: {path}")
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_contents:
        record["sha256"] = _sha256(path)
    return record


def _csv_without_intervals(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            key: value
            for key, value in row.items()
            if not key.endswith(("_ci_low", "_ci_high"))
        }
        for row in rows
    ]


def _assert_point_estimates_unchanged(source: Path, promoted: Path) -> None:
    """Ensure the promotion changes intervals, never fitted choices or points."""

    source_specs = json.loads((source / "frozen_specs.json").read_text("utf-8"))
    promoted_specs = json.loads((promoted / "frozen_specs.json").read_text("utf-8"))
    if source_specs != promoted_specs:
        raise ValueError("bootstrap promotion changed the frozen RAID specifications")

    for filename in (
        "metrics.csv",
        "attack_summary.csv",
        "contamination_summary.csv",
        "rate_bound_tradeoff_summary.csv",
        "binoculars_sanity.csv",
        "calibration_summary.csv",
        "contamination_records.csv",
    ):
        if _csv_without_intervals(source / filename) != _csv_without_intervals(
            promoted / filename
        ):
            raise ValueError(
                "bootstrap promotion changed non-interval values in " + filename
            )


def promote_bootstrap_evaluation(
    workspace: str | Path,
    config_path: str | Path,
    *,
    source_run_id: str,
    promotion_id: str,
    bootstrap_repetitions: int,
    debug_only: bool = False,
) -> dict[str, Any]:
    """Create new result artifacts while reusing immutable completed scores."""

    root = Path(workspace).resolve()
    config_source = Path(config_path)
    if not config_source.is_absolute():
        config_source = root / config_source
    config = load_raid_config(config_source)

    for label, value in (("source run ID", source_run_id), ("promotion ID", promotion_id)):
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError(f"invalid RAID {label}: {value!r}")
    if source_run_id == promotion_id:
        raise ValueError("RAID bootstrap promotion requires a new result ID")

    final_repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    repetitions = int(bootstrap_repetitions)
    if repetitions < 0:
        raise ValueError("bootstrap repetitions cannot be negative")
    if not debug_only and repetitions != final_repetitions:
        raise ValueError(
            "non-debug RAID bootstrap promotion must use the frozen final count "
            f"of {final_repetitions}"
        )

    source_run_dir, source_results_dir = run_paths(root, config, source_run_id)
    _, promotion_results_dir = run_paths(root, config, promotion_id)
    source_manifest_path = source_run_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_report_path = source_results_dir / "validation_report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))

    if source_manifest.get("protocol") != config["protocol_name"]:
        raise ValueError("source RAID run uses a different protocol")
    if source_manifest.get("protocol_config") != config:
        raise ValueError("source RAID run uses a different frozen configuration")
    if source_manifest.get("completion_status") != "complete" or set(
        source_manifest.get("completed_stages", ())
    ) != {"prepare", "score", "evaluate", "plot", "validate"}:
        raise ValueError("source RAID run is not complete")
    if source_report.get("validation_status") != "pass":
        raise ValueError("source RAID run did not pass validation")
    if not debug_only:
        if int(source_report.get("bootstrap_repetitions", -1)) != 500:
            raise ValueError("final promotion expects the audited 500-bootstrap source run")
        if int(source_report.get("development_source_exclusions", -1)) != 500:
            raise ValueError("source RAID run lacks the frozen 500-source exclusion")

    source_exclusions = source_manifest.get("source_exclusions")
    source_exclusion_path = (
        None if source_exclusions is None else source_exclusions.get("path")
    )
    source_artifacts = {
        "manifest": _artifact_record(source_manifest_path, hash_contents=True),
        "validation_report": _artifact_record(source_report_path, hash_contents=True),
        "prepare_marker": _artifact_record(
            source_run_dir / "prepare.complete.json", hash_contents=True
        ),
        "score_marker": _artifact_record(
            source_run_dir / "score.complete.json", hash_contents=True
        ),
        "evaluate_marker": _artifact_record(
            source_results_dir / "evaluate.complete.json", hash_contents=True
        ),
        "plot_marker": _artifact_record(
            source_results_dir / "plot.complete.json", hash_contents=True
        ),
        "prepared_data": _artifact_record(source_run_dir / "data.jsonl"),
        "falcon_scores": _artifact_record(source_run_dir / "falcon_scores.jsonl"),
        "binoculars_scores": _artifact_record(source_run_dir / "binoculars_scores.jsonl"),
    }

    promotion_results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = promotion_results_dir / "promotion_manifest.json"
    expected = {
        "promotion_id": promotion_id,
        "source_run_id": source_run_id,
        "protocol": config["protocol_name"],
        "protocol_config": config,
        "bootstrap_repetitions": repetitions,
        "debug_only": bool(debug_only),
        "source_git_commit": source_manifest.get("git_commit"),
        "promotion_git_commit": _git_commit(root),
        "source_exclusions": source_exclusions,
        "source_artifacts": source_artifacts,
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(f"RAID bootstrap promotion resume mismatch: {key}")
        if previous.get("completion_status") == "complete":
            completion = json.loads(
                (promotion_results_dir / "promotion.complete.json").read_text(
                    encoding="utf-8"
                )
            )
            report = json.loads(
                (promotion_results_dir / "validation_report.json").read_text(
                    encoding="utf-8"
                )
            )
            if completion.get("completion_status") != "complete" or report.get(
                "validation_status"
            ) != "pass":
                raise ValueError("completed RAID promotion lacks a passing result")
            if int(report.get("bootstrap_repetitions", -1)) != repetitions:
                raise ValueError("completed RAID promotion has the wrong bootstrap count")
            return previous
        manifest = previous
    else:
        if any(promotion_results_dir.iterdir()):
            raise ValueError(
                "RAID promotion result directory is nonempty without a promotion manifest"
            )
        manifest = {
            **expected,
            "creation_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "completion_status": "running",
            "completed_stages": [],
        }
        atomic_write_json(manifest_path, manifest)

    try:
        evaluate_stage(
            source_run_dir,
            promotion_results_dir,
            config,
            bootstrap_repetitions=repetitions,
            skip_binoculars=False,
        )
        _assert_point_estimates_unchanged(
            source_results_dir, promotion_results_dir
        )
        manifest["completed_stages"] = ["evaluate"]
        atomic_write_json(manifest_path, manifest)

        plot_stage(promotion_results_dir)
        manifest["completed_stages"].append("plot")
        atomic_write_json(manifest_path, manifest)

        report = validate_artifacts(
            source_run_dir,
            promotion_results_dir,
            config,
            limit_sources=source_manifest.get("limit_sources"),
            bootstrap_repetitions=repetitions,
            skip_binoculars=False,
            source_exclusion_path=source_exclusion_path,
            debug_only=debug_only,
        )
        manifest["completed_stages"].append("validate")
        manifest["validation_status"] = report["validation_status"]
        manifest["completion_status"] = "complete"
        manifest["completion_time_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(
            promotion_results_dir / "promotion.complete.json",
            {
                "promotion_id": promotion_id,
                "source_run_id": source_run_id,
                "bootstrap_repetitions": repetitions,
                "validation_status": report["validation_status"],
                "completion_status": "complete",
            },
        )
    except Exception as exc:
        manifest["completion_status"] = "failed"
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_write_json(manifest_path, manifest)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--promotion-id", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--config", default="configs/raid.json")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--debug-only", action="store_true")
    args = parser.parse_args()

    manifest = promote_bootstrap_evaluation(
        args.workspace,
        args.config,
        source_run_id=args.source_run_id,
        promotion_id=args.promotion_id,
        bootstrap_repetitions=args.bootstrap_repetitions,
        debug_only=args.debug_only,
    )
    print(f"raid_source_run_id={args.source_run_id}")
    print(f"raid_promotion_id={args.promotion_id}")
    print(f"raid_promotion_status={manifest['completion_status']}")


if __name__ == "__main__":
    main()
