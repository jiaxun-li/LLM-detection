#!/usr/bin/env python3
"""Score one deterministic RAID source shard without mutating shared state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RAID.raid_pipeline import (
    load_raid_config,
    run_paths,
    score_shard_stage,
    validate_resume_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scorer", required=True, choices=("falcon", "binoculars"))
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--config", default="RAID/config.json")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace / config_path
    config = load_raid_config(config_path)
    run_dir, _ = run_paths(workspace, config, args.run_id)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RAID preparation manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "prepare" not in manifest.get("completed_stages", []):
        raise ValueError("RAID shard scoring requires a completed prepare stage")
    validate_resume_manifest(
        manifest,
        config,
        run_id=args.run_id,
        data_path=None,
        limit_sources=manifest.get("limit_sources"),
        bootstrap_repetitions=manifest.get("bootstrap_repetitions_override"),
        skip_binoculars=not bool(manifest.get("binoculars_included", True)),
        num_shards=args.num_shards,
    )
    result = score_shard_stage(
        run_dir,
        config,
        scorer_name=args.scorer,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(f"raid_run_id={args.run_id}")
    print(f"raid_scorer={args.scorer}")
    print(f"raid_shard={args.shard_index}/{args.num_shards}")
    print(f"raid_score_rows={result['score_rows']}")


if __name__ == "__main__":
    main()
