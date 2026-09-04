"""Explicit, opt-in detector amendment; historical score files remain immutable."""
from __future__ import annotations

import numpy as np

REVISION = "binocular-origin-lrr-constant-v1"
IMPLEMENTATION_VERSION = "native-falcon-numerics-gate-v3"
PAIR_METHODS = ("binoculars", "binocular_gap", "binocular_origin")
ORIGIN_NLL = "binocular_origin_nll"
ORIGIN_DENOMINATOR = "binocular_origin_denominator"


def round_bfloat16(values):
    """Round finite float32 values to BF16, nearest with ties to even."""
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite BF16 input")
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def official_nll_mean(values, upper=None):
    """Replay CUDA BF16 tensor division, NOT CPU/scalar division.

    Saved NLL is already BF16. Round the optional capped values back to that
    dtype before reduction. The scorer verifies this replay against the actual
    Torch numerator for EVERY new row; the GPU gate tests capped reductions too.
    CUDA converts the integer mask-count tensor to BF16 for this operation.
    Odd counts above 256 can round (257 -> 256, 259 -> 260). A Python scalar
    divisor follows a different kernel path and must not be used as reference.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or (values < 0).any():
        raise ValueError("invalid official numerator values")
    if upper is not None:
        if not np.isfinite(upper) or upper < 0:
            raise ValueError("invalid NLL cap")
        values = np.minimum(values, upper)
    values = round_bfloat16(values)
    total = round_bfloat16(values.sum(dtype=np.float64))
    divisor = round_bfloat16(len(values))
    return float(round_bfloat16(np.float32(total) / np.float32(divisor)))


def structurally_constant_candidate(rows, detector, direction, spec):
    """Exact full saturation, BEFORE length-dependent floating-point means.

    No tolerance and no new tuning parameter. Origin is excluded: even a
    constant numerator can retain real information through its denominator.
    """
    if not spec or detector == "binocular_origin":
        return False
    for row in rows:
        f = row["token_features"]
        if detector == "lrr":
            if not (np.all(-np.asarray(f["logp"]) >= spec["nll_upper"])
                    and np.all(np.asarray(f["log_rank"]) >= spec["log_rank_upper"])):
                return False
        else:
            if detector in {"binoculars", "binocular_gap"}:
                values = np.asarray(f["performer_nll"]) - np.asarray(f["observer_to_performer_cross_entropy"])
            elif detector == "entropy_gap":
                values = -np.asarray(f["logp"]) - np.asarray(f["entropy"])
            else:
                values = np.asarray(f[{"log_likelihood":"logp", "rank":"rank",
                                       "log_rank":"log_rank", "entropy":"entropy"}[detector]])
            if not np.all(direction * values <= spec["lower"]):
                return False
    return bool(rows)


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
        if ORIGIN_NLL in row["token_features"]:
            numerator = official_nll_mean(nll, spec["nll_upper"])
            return float(np.float32(numerator) / np.float32(denominator))
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
