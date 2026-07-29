"""Deprecated alias for the configuration-first preparation stage.

Use the same arguments as ``run_experiment.py``. This filename is retained so
old cluster notes fail forward into the deterministic, leakage-safe pipeline.
"""

from __future__ import annotations

import sys

from run_experiment import main


if __name__ == "__main__":
    if "--stage" not in sys.argv:
        sys.argv.extend(["--stage", "prepare"])
    main()
