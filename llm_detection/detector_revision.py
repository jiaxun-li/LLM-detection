"""Explicit, opt-in detector amendment; historical score files remain immutable."""
from __future__ import annotations

import numpy as np

REVISION = "binocular-origin-lrr-constant-v1"
PAIR_METHODS = ("binoculars", "binocular_gap", "binocular_origin")
ORIGIN_NLL = "binocular_origin_nll"
ORIGIN_DENOMINATOR = "binocular_origin_denominator"


def origin_components(row):
    """RAID uses namespaced official features; primary uses its saved window."""
    features = row["token_features"]
    if ORIGIN_NLL in features:
        nll = np.asarray(features[ORIGIN_NLL], dtype=float)
        denominator = float(row["doc_scores"][ORIGIN_DENOMINATOR])
    else:
        nll = np.asarray(features["performer_nll"], dtype=float)
        ce = np.asarray(features["observer_to_performer_cross_entropy"], dtype=float)
        if len(nll) != len(ce):
            raise ValueError("conditional Binoculars requires matching saved windows")
        denominator = float(ce.mean())
    if nll.ndim != 1 or not len(nll) or not np.isfinite(nll).all():
        raise ValueError("invalid Binoculars numerator")
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("invalid Binoculars denominator")
    return nll, denominator


def origin_score(row, spec=None):
    nll, denominator = origin_components(row)
    if spec:
        if not np.any(nll > spec["nll_upper"]):
            # Do not manufacture a change solely by re-rounding an unaffected
            # official BF16 score in the float64 analysis code.
            return origin_score(row)
        return float(np.minimum(nll, spec["nll_upper"]).mean() / denominator)
    # Preserve upstream dtype/reduction exactly for official RAID raw scores.
    if "binocular_origin" in row.get("doc_scores", {}):
        value = float(row["doc_scores"]["binocular_origin"])
        if not np.isfinite(value):
            raise ValueError("nonfinite official Binoculars score")
        return value
    return float(nll.mean() / denominator)


def constant_clean_scores(human, machine):
    """Exact degeneracy only: no tolerance, no test/calibration observations."""
    values = np.asarray(list(human) + list(machine), dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("clean tuning scores must be nonempty and finite")
    return bool(np.all(values == values[0]))
