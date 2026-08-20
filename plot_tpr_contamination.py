#!/usr/bin/env python3
"""Plot TPR at calibrated 1% and 5% FPR versus contamination ratio.

Example:

    python plot_tpr_contamination.py \
      --metrics /path/to/results/<run-id>/metrics.csv \
      --output /path/to/results/<run-id>/tpr_contamination.png

The plot uses the primary frozen-mixture analysis by default. Colors distinguish
random and tail contamination; line styles distinguish raw and clipped scores.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DETECTOR_ORDER = [
    "log_likelihood",
    "rank",
    "log_rank",
    "lrr",
    "entropy",
    "entropy_gap",
    "binoculars",
]

DETECTOR_LABELS = {
    "log_likelihood": "Log likelihood",
    "rank": "Rank",
    "log_rank": "Log rank",
    "lrr": "LRR",
    "entropy": "Entropy",
    "entropy_gap": "Entropy gap",
    "binoculars": "Binoculars",
}

MODE_COLORS = {
    "random": "#2474b5",
    "tail": "#d1495b",
}

AGGREGATION_STYLES = {
    "raw": ("--", "o"),
    "clipped": ("-", "s"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--detectors",
        help="Comma-separated detector names; default: every detector in the CSV",
    )
    parser.add_argument(
        "--analysis",
        default="primary_frozen_mixture",
        help="Metrics analysis label to plot",
    )
    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Do not draw bootstrap 95%% confidence bands",
    )
    parser.add_argument(
        "--layout",
        choices=["detector-rows", "detector-columns"],
        default="detector-rows",
        help="Arrange detectors vertically (default) or horizontally",
    )
    parser.add_argument(
        "--target-fprs",
        default="0.01,0.05",
        help="Comma-separated target FPRs to draw; available: 0.01,0.05",
    )
    return parser.parse_args()


def read_metrics(path: Path, analysis: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "split",
            "analysis",
            "detector",
            "aggregation",
            "contamination_mode",
            "requested_contamination_ratio",
            "target_fpr",
            "tpr",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"metrics CSV is missing columns: {sorted(missing)}")

        rows: list[dict[str, Any]] = []
        for row in reader:
            if row["split"] != "test" or row["analysis"] != analysis:
                continue
            if row["contamination_mode"] not in MODE_COLORS:
                continue
            if row["aggregation"] not in AGGREGATION_STYLES:
                continue
            target_fpr = float(row["target_fpr"])
            if not any(abs(target_fpr - value) < 1e-12 for value in (0.01, 0.05)):
                continue
            parsed = dict(row)
            for name in (
                "requested_contamination_ratio",
                "target_fpr",
                "tpr",
                "tpr_ci_low",
                "tpr_ci_high",
            ):
                value = row.get(name, "")
                parsed[name] = float(value) if value not in {"", None} else None
            rows.append(parsed)
    if not rows:
        raise ValueError(f"no matching test metrics found in {path}")
    return rows


def selected_detectors(
    rows: list[dict[str, Any]], requested: str | None
) -> list[str]:
    available = {str(row["detector"]) for row in rows}
    if requested:
        selected = [value.strip() for value in requested.split(",") if value.strip()]
        missing = set(selected) - available
        if missing:
            raise ValueError(f"requested detectors are absent: {sorted(missing)}")
        return selected
    ordered = [name for name in DEFAULT_DETECTOR_ORDER if name in available]
    return ordered + sorted(available - set(ordered))


def plot(
    rows: list[dict[str, Any]],
    detectors: list[str],
    output: Path,
    show_ci: bool,
    *,
    layout: str = "detector-rows",
    target_fprs: tuple[float, ...] = (0.01, 0.05),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["detector"] in detectors:
            key = (
                str(row["detector"]),
                float(row["target_fpr"]),
                str(row["contamination_mode"]),
                str(row["aggregation"]),
            )
            grouped[key].append(row)

    if layout == "detector-columns":
        row_count, column_count = len(target_fprs), len(detectors)
        figure_size = (max(4.0 * len(detectors), 6.0), 3.8 * len(target_fprs))
    else:
        row_count, column_count = len(detectors), len(target_fprs)
        figure_size = (5.75 * len(target_fprs), max(3.4 * len(detectors), 4.2))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=figure_size,
        dpi=120 if layout == "detector-columns" else 150,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    legend_handles: dict[str, Any] = {}

    for detector_index, detector in enumerate(detectors):
        for fpr_index, target_fpr in enumerate(target_fprs):
            if layout == "detector-columns":
                axis = axes[fpr_index][detector_index]
            else:
                axis = axes[detector_index][fpr_index]
            for mode in ("random", "tail"):
                for aggregation in ("raw", "clipped"):
                    points = sorted(
                        grouped.get((detector, target_fpr, mode, aggregation), []),
                        key=lambda item: float(item["requested_contamination_ratio"]),
                    )
                    if not points:
                        continue
                    x_values = [
                        float(item["requested_contamination_ratio"]) for item in points
                    ]
                    y_values = [float(item["tpr"]) for item in points]
                    linestyle, marker = AGGREGATION_STYLES[aggregation]
                    label = f"{mode.title()} / {aggregation}"
                    (line,) = axis.plot(
                        x_values,
                        y_values,
                        color=MODE_COLORS[mode],
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=2.0 if aggregation == "clipped" else 1.5,
                        markersize=4,
                        label=label,
                    )
                    legend_handles[label] = line
                    if show_ci and all(
                        item.get("tpr_ci_low") is not None
                        and item.get("tpr_ci_high") is not None
                        for item in points
                    ):
                        axis.fill_between(
                            x_values,
                            [float(item["tpr_ci_low"]) for item in points],
                            [float(item["tpr_ci_high"]) for item in points],
                            color=MODE_COLORS[mode],
                            alpha=0.08,
                            linewidth=0,
                        )

            detector_label = DETECTOR_LABELS.get(detector, detector)
            if layout == "detector-columns" and len(target_fprs) == 1:
                axis.set_title(detector_label)
            else:
                axis.set_title(f"{detector_label} — TPR at {target_fpr:.0%} FPR")
            axis.set_xlim(-0.01, 0.51)
            axis.set_ylim(-0.02, 1.02)
            axis.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            axis.set_xticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
            axis.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)
            is_first_column = (
                detector_index == 0
                if layout == "detector-columns"
                else fpr_index == 0
            )
            is_bottom_row = (
                fpr_index == len(target_fprs) - 1
                if layout == "detector-columns"
                else detector_index == len(detectors) - 1
            )
            if is_first_column:
                axis.set_ylabel("True positive rate")
            if is_bottom_row:
                axis.set_xlabel("Requested human-token contamination")

    first = rows[0]
    fpr_label = ", ".join(f"{value:.0%}" for value in target_fprs)
    figure.suptitle(
        f"{first.get('dataset', 'dataset')} — {first.get('model', 'model')}\n"
        f"TPR at {fpr_label} FPR; random/tail contamination; raw/clipped",
        fontsize=13,
    )
    labels = ["Random / raw", "Random / clipped", "Tail / raw", "Tail / clipped"]
    available_labels = [label for label in labels if label in legend_handles]
    if available_labels:
        figure.legend(
            [legend_handles[label] for label in available_labels],
            available_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=len(available_labels),
            frameon=False,
        )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(f"plotted_rows={sum(len(value) for value in grouped.values())}")
    print(f"saved={output.resolve()}")


def main() -> None:
    args = parse_args()
    rows = read_metrics(args.metrics, args.analysis)
    detectors = selected_detectors(rows, args.detectors)
    target_fprs = tuple(
        float(value.strip())
        for value in args.target_fprs.split(",")
        if value.strip()
    )
    if not target_fprs or any(
        not any(abs(value - available) < 1e-12 for available in (0.01, 0.05))
        for value in target_fprs
    ):
        raise ValueError("--target-fprs must contain 0.01 and/or 0.05")
    plot(
        rows,
        detectors,
        args.output,
        not args.no_ci,
        layout=args.layout,
        target_fprs=target_fprs,
    )


if __name__ == "__main__":
    main()
