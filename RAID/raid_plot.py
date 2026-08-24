"""Publication-ready diagnostic plots for the RAID benchmark."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


DETECTOR_LABELS = {
    "log_likelihood": "Log likelihood",
    "rank": "Rank",
    "log_rank": "Log rank",
    "lrr": "LRR",
    "entropy": "Entropy",
    "entropy_gap": "Entropy gap",
    "binoculars": "Binoculars",
}
ATTACK_ORDER = (
    "none",
    "paraphrase",
    "synonym",
    "perplexity_misspelling",
    "homoglyph",
    "whitespace",
    "article_deletion",
    "alternative_spelling",
    "insert_paragraphs",
    "number",
    "upper_lower",
    "zero_width_space",
)
ATTACK_LABELS = {
    "none": "None",
    "paraphrase": "Paraphrase",
    "synonym": "Synonym",
    "perplexity_misspelling": "Misspelling",
    "homoglyph": "Homoglyph",
    "whitespace": "Whitespace",
    "article_deletion": "Delete articles",
    "alternative_spelling": "Alt. spelling",
    "insert_paragraphs": "Insert paragraphs",
    "number": "Numbers",
    "upper_lower": "Case",
    "zero_width_space": "Zero-width",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float("nan") if value in {"", None} else float(value)


def _axes_grid(plt: Any) -> tuple[Any, list[Any]]:
    figure, grid = plt.subplots(4, 2, figsize=(15, 13), sharey=True)
    axes = list(grid.flat)
    axes[-1].axis("off")
    return figure, axes


def _plot_series(
    axis: Any,
    x: list[int],
    rows: list[dict[str, str]],
    prefix: str,
    *,
    label: str,
    color: str,
    marker: str,
) -> None:
    values = [_number(row, f"{prefix}_tpr") for row in rows]
    low = [_number(row, f"{prefix}_tpr_ci_low") for row in rows]
    high = [_number(row, f"{prefix}_tpr_ci_high") for row in rows]
    axis.plot(x, values, color=color, marker=marker, linewidth=1.8, label=label)
    if all(value == value for value in [*low, *high]):
        axis.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)


def _attack_plot(results_dir: Path, plt: Any) -> Path:
    rows = _read_csv(results_dir / "attack_summary.csv")
    sanity = {
        row["condition"]: _number(row, "published_tpr")
        for row in _read_csv(results_dir / "binoculars_sanity.csv")
    }
    figure, axes = _axes_grid(plt)
    for axis, (detector, title) in zip(axes, DETECTOR_LABELS.items()):
        lookup = {
            row["condition"]: row for row in rows if row["detector"] == detector
        }
        selected = [lookup[attack] for attack in ATTACK_ORDER]
        x = list(range(len(selected)))
        _plot_series(
            axis,
            x,
            selected,
            "raw",
            label="Raw",
            color="#2b6cb0",
            marker="o",
        )
        _plot_series(
            axis,
            x,
            selected,
            "clipped",
            label="Universal clipped",
            color="#d64541",
            marker="s",
        )
        if detector == "binoculars":
            published_x = [i for i, name in enumerate(ATTACK_ORDER) if name in sanity]
            axis.scatter(
                published_x,
                [sanity[ATTACK_ORDER[i]] for i in published_x],
                marker="x",
                s=52,
                linewidths=2,
                color="#202020",
                label="RAID published",
                zorder=4,
            )
        axis.set_title(title)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [ATTACK_LABELS[name] for name in ATTACK_ORDER],
            rotation=55,
            ha="right",
            fontsize=8,
        )
        axis.set_ylabel("TPR at calibrated 5% FPR")
        axis.legend(fontsize=8, loc="best")
    figure.suptitle("RAID attack robustness: raw versus universal clipping")
    figure.tight_layout()
    output = results_dir / "plots" / "raid_attack_tpr.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _contamination_plot(results_dir: Path, plt: Any) -> Path:
    rows = _read_csv(results_dir / "contamination_summary.csv")
    figure, axes = _axes_grid(plt)
    for axis, (detector, title) in zip(axes, DETECTOR_LABELS.items()):
        selected = sorted(
            (row for row in rows if row["detector"] == detector),
            key=lambda row: int(row["contamination_bin_index"]),
        )
        x = list(range(len(selected)))
        _plot_series(
            axis,
            x,
            selected,
            "raw",
            label="Raw",
            color="#2b6cb0",
            marker="o",
        )
        _plot_series(
            axis,
            x,
            selected,
            "clipped",
            label="Universal clipped",
            color="#d64541",
            marker="s",
        )
        _plot_series(
            axis,
            x,
            selected,
            "rate_adaptive_clipped",
            label="Rate-oracle clipped",
            color="#2f855a",
            marker="^",
        )
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_title(title)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [row["contamination_bin"] for row in selected],
            rotation=35,
            ha="right",
            fontsize=8,
        )
        axis.set_ylabel("TPR at calibrated 5% FPR")
        axis.set_xlabel("Realized Falcon-token edit rate interval")
        axis.legend(fontsize=8, loc="best")
    figure.suptitle(
        "RAID robustness in four fixed realized-contamination intervals\n"
        "(rate zero and rates above 0.50 excluded from this analysis)"
    )
    figure.tight_layout()
    output = results_dir / "plots" / "raid_contamination_tpr.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_raid(results_dir: str | Path) -> list[Path]:
    """Create the two frozen RAID result figures."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("RAID plotting requires matplotlib") from exc
    root = Path(results_dir)
    for name in ("attack_summary.csv", "contamination_summary.csv", "binoculars_sanity.csv"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"RAID plotting is missing {root / name}")
    return [_attack_plot(root, plt), _contamination_plot(root, plt)]
