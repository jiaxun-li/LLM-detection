#!/usr/bin/env python3
"""Validate a completed Delta Qwen 0.5B smoke run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from llm_detection.io import atomic_write_json
from llm_detection.smoke_validation import validate_smoke_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/delta_smoke_qwen_0_5b.json"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--write-baseline")
    parser.add_argument("--compare-baseline")
    parser.add_argument("--write-completion-marker", action="store_true")
    parser.add_argument("--materialize-views", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = validate_smoke_run(
            args.config,
            args.run_dir,
            report_path=args.report,
            write_baseline=args.write_baseline,
            compare_baseline=args.compare_baseline,
            write_completion_marker=args.write_completion_marker,
            materialize_views=args.materialize_views,
        )
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "run_dir": str(Path(args.run_dir).resolve()),
        }
        atomic_write_json(args.report, failure)
        print("DELTA SMOKE VALIDATION: FAIL", file=sys.stderr)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        raise
    print("DELTA SMOKE VALIDATION: PASS")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
