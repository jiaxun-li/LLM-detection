"""Compact figures for the paired splice-artifact audit."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


DETECTOR_ORDER = (
    "log_likelihood",
    "rank",
    "log_rank",
    "lrr",
    "entropy",
    "entropy_gap",
    "binoculars",
)
LABELS = {
    "log_likelihood": "Log likelihood",
    "rank": "Rank",
    "log_rank": "Log rank",
    "lrr": "LRR",
    "entropy": "Entropy",
    "entropy_gap": "Entropy gap",
    "binoculars": "Binoculars",
}
CONDITION_STYLE = {
    "token_human": ("#d62728", "-", "Human / token windows"),
    "token_llm": ("#1f77b4", "-", "LLM / token windows"),
    "sentence_human": ("#d62728", "--", "Human / sentence aligned"),
    "sentence_llm": ("#1f77b4", "--", "LLM / sentence aligned"),
}


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = dict(row)
            for key in (
                "target_fpr",
                "requested_contamination_ratio",
                "tpr",
                "tpr_ci_low",
                "tpr_ci_high",
                "boundary_artifact_share",
            ):
                if key in parsed and parsed[key] not in {"", None}:
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def _plot_curves(
    metrics: list[dict[str, Any]],
    aggregation: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detectors = [
        detector
        for detector in DETECTOR_ORDER
        if any(row["detector"] == detector for row in metrics)
    ]
    figure, axes = plt.subplots(
        2,
        len(detectors),
        figsize=(4.0 * len(detectors), 7.6),
        dpi=120,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    handles: dict[str, Any] = {}
    for column, detector in enumerate(detectors):
        for row_index, target_fpr in enumerate((0.01, 0.05)):
            axis = axes[row_index][column]
            for condition, (color, linestyle, label) in CONDITION_STYLE.items():
                points = sorted(
                    [
                        row
                        for row in metrics
                        if row["detector"] == detector
                        and row["aggregation"] == aggregation
                        and abs(row["target_fpr"] - target_fpr) < 1e-12
                        and row["condition"] == condition
                    ],
                    key=lambda row: row["requested_contamination_ratio"],
                )
                if not points:
                    continue
                x = [row["requested_contamination_ratio"] for row in points]
                y = [row["tpr"] for row in points]
                (line,) = axis.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    marker="o" if condition.endswith("human") else "s",
                    linewidth=2,
                    markersize=4,
                    label=label,
                )
                handles[label] = line
                axis.fill_between(
                    x,
                    [row["tpr_ci_low"] for row in points],
                    [row["tpr_ci_high"] for row in points],
                    color=color,
                    alpha=0.07,
                    linewidth=0,
                )
            axis.set_title(
                f"{LABELS.get(detector, detector)} — {target_fpr:.0%} FPR"
            )
            axis.set_xlim(-0.01, 0.51)
            axis.set_ylim(-0.02, 1.02)
            axis.set_xticks([0.0, 0.1, 0.3, 0.5])
            axis.set_xticklabels(["0%", "10%", "30%", "50%"])
            axis.grid(True, color="#dddddd", alpha=0.8)
            if column == 0:
                axis.set_ylabel("True positive rate")
            if row_index == 1:
                axis.set_xlabel("Requested contamination")
    figure.suptitle(
        f"Granite-8B × XSum paired splice audit — {aggregation}", fontsize=14
    )
    ordered_labels = [value[2] for value in CONDITION_STYLE.values()]
    figure.legend(
        [handles[label] for label in ordered_labels if label in handles],
        [label for label in ordered_labels if label in handles],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_artifact_share(
    rows: list[dict[str, Any]], output_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    detectors = [
        detector
        for detector in DETECTOR_ORDER
        if any(row["detector"] == detector for row in rows)
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=140, squeeze=False)
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["target_fpr"], row["placement"])].append(row)
    for row_index, target_fpr in enumerate((0.01, 0.05)):
        for column, placement in enumerate(("token", "sentence")):
            axis = axes[row_index][column]
            x = np.arange(len(detectors))
            for offset, aggregation in ((-0.18, "raw"), (0.18, "clipped")):
                indexed = {
                    row["detector"]: row
                    for row in grouped[(target_fpr, placement)]
                    if row["aggregation"] == aggregation
                }
                values = [
                    indexed.get(detector, {}).get(
                        "boundary_artifact_share", float("nan")
                    )
                    for detector in detectors
                ]
                axis.bar(
                    x + offset,
                    values,
                    width=0.34,
                    label=aggregation.title(),
                    color="#777777" if aggregation == "raw" else "#e377c2",
                )
            axis.axhline(0.3, color="#2ca02c", linestyle="--", linewidth=1)
            axis.axhline(0.7, color="#d62728", linestyle="--", linewidth=1)
            axis.set_title(f"{placement.title()} placement — {target_fpr:.0%} FPR")
            axis.set_xticks(x)
            axis.set_xticklabels(
                [LABELS.get(value, value) for value in detectors],
                rotation=30,
                ha="right",
            )
            axis.set_ylabel("Boundary-artifact share")
            axis.grid(True, axis="y", color="#dddddd", alpha=0.7)
            axis.legend(frameon=False)
    figure.suptitle(
        "Boundary degradation as a fraction of human-donor degradation",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_splice_audit(
    metrics_path: str | Path,
    artifact_share_path: str | Path,
    output_dir: str | Path,
) -> list[str]:
    metrics = _read_csv(metrics_path)
    artifacts = _read_csv(artifact_share_path)
    root = Path(output_dir)
    outputs = []
    for aggregation in ("raw", "clipped"):
        path = root / f"splice_audit_{aggregation}_2x7.png"
        _plot_curves(metrics, aggregation, path)
        outputs.append(str(path))
    artifact_path = root / "boundary_artifact_share.png"
    _plot_artifact_share(artifacts, artifact_path)
    outputs.append(str(artifact_path))
    return outputs
