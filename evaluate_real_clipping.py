"""Deprecated alias for independent-split clipping evaluation.

Use the same arguments as ``run_experiment.py``. This alias selects the
evaluation stage, which tunes on ``clipping_tuning``, calibrates on
``calibration``, and reports only on ``test``.
"""

from __future__ import annotations

import sys

from run_experiment import main


if __name__ == "__main__":
    if "--stage" not in sys.argv:
        sys.argv.extend(["--stage", "evaluate"])
    main()
