"""Compact Beemo TPR plots: seven detectors in each FPR row."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


DISPLAY_CONDITIONS = ("original", "expert", "llama_all", "gpt_all")
CONDITION_LABELS = {
    "original": "Original\nMGT",
    "expert": "Expert\nedit",
    "llama_all": "Llama\nedits",
    "gpt_all": "GPT-4o\nedits",
}
DETECTOR_LABELS = {
    "log_likelihood": "Log likelihood",
    "rank": "Rank",
    "log_rank": "Log rank",
    "lrr": "LRR",
    "entropy": "Entropy",
    "entropy_gap": "Entropy gap",
    "binoculars": "Binoculars",
}


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_beemo(metrics_path: str | Path, output_dir: str | Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Beemo plotting requires matplotlib") from exc
    rows = _read_rows(metrics_path)
    fprs = sorted({float(row["target_fpr"]) for row in rows})
    scorer_keys = list(
        dict.fromkeys(
            row["scorer_key"]
            for row in rows
            if row["scorer_key"] != "binoculars"
        )
    )
    if not rows or not scorer_keys or not fprs:
        raise ValueError("Beemo metrics contain no plottable rows")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for scorer_key in scorer_keys:
        scorer_rows = [
            row
            for row in rows
            if row["scorer_key"] in {scorer_key, "binoculars"}
        ]
        detectors = list(
            dict.fromkeys(
                row["detector"]
                for row in scorer_rows
                if row["scorer_key"] == scorer_key
            )
        )
        if any(row["scorer_key"] == "binoculars" for row in scorer_rows):
            detectors.append("binoculars")
        figure, axes = plt.subplots(
            len(fprs),
            len(detectors),
            figsize=(3.15 * len(detectors), 3.0 * len(fprs)),
            sharey=True,
            squeeze=False,
        )
        x = list(range(len(DISPLAY_CONDITIONS)))
        styles = {
            "raw": {"color": "#2878B5", "marker": "o", "linestyle": "--"},
            "clipped": {"color": "#D43F3A", "marker": "s", "linestyle": "-"},
        }
        for row_index, target_fpr in enumerate(fprs):
            for column, detector in enumerate(detectors):
                axis = axes[row_index][column]
                required_scorer = "binoculars" if detector == "binoculars" else scorer_key
                for aggregation in ("raw", "clipped"):
                    selected = {
                        row["condition"]: row
                        for row in scorer_rows
                        if row["scorer_key"] == required_scorer
                        and row["detector"] == detector
                        and row["aggregation"] == aggregation
                        and abs(float(row["target_fpr"]) - target_fpr) < 1e-12
                        and row["condition"] in DISPLAY_CONDITIONS
                    }
                    if set(selected) != set(DISPLAY_CONDITIONS):
                        raise ValueError(
                            f"missing Beemo plot rows for {scorer_key} {detector} "
                            f"{aggregation} {target_fpr}"
                        )
                    y = [float(selected[name]["tpr"]) for name in DISPLAY_CONDITIONS]
                    low = [float(selected[name]["tpr_ci_low"]) for name in DISPLAY_CONDITIONS]
                    high = [float(selected[name]["tpr_ci_high"]) for name in DISPLAY_CONDITIONS]
                    axis.errorbar(
                        x,
                        y,
                        yerr=[
                            [max(value - lower, 0.0) for value, lower in zip(y, low)],
                            [max(upper - value, 0.0) for value, upper in zip(y, high)],
                        ],
                        linewidth=1.6,
                        markersize=4,
                        capsize=2,
                        label=aggregation.capitalize(),
                        **styles[aggregation],
                    )
                axis.set_ylim(-0.03, 1.03)
                axis.grid(alpha=0.25, linewidth=0.6)
                axis.set_xticks(
                    x,
                    [CONDITION_LABELS[name] for name in DISPLAY_CONDITIONS],
                    fontsize=8,
                )
                axis.set_title(
                    f"{DETECTOR_LABELS.get(detector, detector)}\n"
                    f"TPR at {target_fpr:.0%} FPR",
                    fontsize=10,
                )
                if column == 0:
                    axis.set_ylabel("True positive rate")
                if row_index == 0 and column == len(detectors) - 1:
                    axis.legend(loc="best", fontsize=8)
        scoring_model = next(
            row["scoring_model"]
            for row in scorer_rows
            if row["scorer_key"] == scorer_key
        )
        figure.suptitle(
            f"Beemo output-only scoring: {scoring_model}",
            fontsize=14,
            y=1.01,
        )
        figure.tight_layout()
        overview = output / f"beemo_tpr_{scorer_key}_2x7.png"
        figure.savefig(overview, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs.append(str(overview))
    return outputs
