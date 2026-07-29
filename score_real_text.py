"""Deprecated alias for the configuration-first scoring stage.

The historical standalone scorer loaded the full input and performed one
forward per document. Use the same arguments as ``run_experiment.py``; this
alias selects its restartable, batched scoring stage.
"""

from __future__ import annotations

import sys

from run_experiment import main


if __name__ == "__main__":
    if "--stage" not in sys.argv:
        sys.argv.extend(["--stage", "score"])
    main()
