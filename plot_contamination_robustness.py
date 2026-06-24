"""Plot robustness curves as contamination increases.

This is the main toy figure for the clipping idea:

    x-axis: human contamination ratio in mostly-LLM documents
    y-axis: TPR at a fixed FPR

Run:

    python plot_contamination_robustness.py

Outputs:

    plots/random_contamination_tpr_at_fpr.png
    plots/tail_contamination_tpr_at_fpr.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_robust_sum_methods import COLORS, METHOD_LABELS, make_dataset, oriented_method_scores
from robust_sum_methods_toy import (
    contaminate_llm_with_human,
    score_methods,
    tpr_at_fpr,
    tune_clip_specs,
)


N_TRIALS = 100


def plot_mode(
    contamination_mode: str,
    fpr: float,
    output_dir: Path,
) -> None:
    n_val = 1_000
    n_test = 2_000
    n_tokens = 80
    min_effective_tokens = 25
    max_effective_tokens = n_tokens
    contamination_grid = np.array([0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])

    raw_tprs = {method: [] for method in METHOD_LABELS}
    clipped_tprs = {method: [] for method in METHOD_LABELS}

    base_seed = 101 if contamination_mode == "random" else 131
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(base_seed + trial)

        val_human, val_llm, val_human_for_contamination = make_dataset(
            rng,
            n_val,
            n_tokens,
            min_effective_tokens,
            max_effective_tokens,
        )
        clip_specs = tune_clip_specs(
            val_human,
            val_llm,
            val_human_for_contamination,
            rng,
            contamination=0.25,
            contamination_mode=contamination_mode,
        )

        test_human, clean_test_llm, human_for_contamination = make_dataset(
            rng,
            n_test,
            n_tokens,
            min_effective_tokens,
            max_effective_tokens,
        )

        raw_clean_human = score_methods(test_human)
        raw_clean_llm = score_methods(clean_test_llm)
        clipped_clean_human = score_methods(test_human, clip_specs)
        clipped_clean_llm = score_methods(clean_test_llm, clip_specs)

        trial_raw_tprs = {method: [] for method in METHOD_LABELS}
        trial_clipped_tprs = {method: [] for method in METHOD_LABELS}

        for contamination in contamination_grid:
            mixed_llm = contaminate_llm_with_human(
                rng,
                clean_test_llm,
                human_for_contamination,
                contamination=float(contamination),
                mode=contamination_mode,
            )

            raw_oriented = oriented_method_scores(
                raw_clean_human,
                raw_clean_llm,
                score_methods(test_human),
                score_methods(mixed_llm),
            )
            clipped_oriented = oriented_method_scores(
                clipped_clean_human,
                clipped_clean_llm,
                score_methods(test_human, clip_specs),
                score_methods(mixed_llm, clip_specs),
            )

            for method in METHOD_LABELS:
                human_raw, llm_raw = raw_oriented[method]
                human_clip, llm_clip = clipped_oriented[method]
                trial_raw_tprs[method].append(tpr_at_fpr(human_raw, llm_raw, fpr=fpr))
                trial_clipped_tprs[method].append(tpr_at_fpr(human_clip, llm_clip, fpr=fpr))

        for method in METHOD_LABELS:
            raw_tprs[method].append(trial_raw_tprs[method])
            clipped_tprs[method].append(trial_clipped_tprs[method])

    for method in METHOD_LABELS:
        raw_tprs[method] = np.mean(raw_tprs[method], axis=0)
        clipped_tprs[method] = np.mean(clipped_tprs[method], axis=0)

    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=160)

    for method, label in METHOD_LABELS.items():
        ax.plot(
            contamination_grid,
            raw_tprs[method],
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=COLORS[method],
            alpha=0.45,
            label=f"{label} raw",
        )
        ax.plot(
            contamination_grid,
            clipped_tprs[method],
            linestyle="-",
            marker="o",
            markersize=4,
            linewidth=2.0,
            color=COLORS[method],
            label=f"{label} clipped",
        )

    ax.set_xlim(0.0, 0.50)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(contamination_grid)
    ax.set_xticklabels([f"{int(x * 100)}%" for x in contamination_grid], rotation=35)
    ax.set_xlabel("Human contamination ratio in LLM documents")
    ax.set_ylabel(f"TPR at {fpr:.0%} FPR")
    ax.set_title(f"{contamination_mode.title()} contamination robustness (mean over {N_TRIALS} trials)")
    ax.grid(True, color="#d9d9d9", linewidth=1.0)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.92, ncol=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{contamination_mode}_contamination_tpr_at_fpr.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    output_dir = Path("plots")
    fpr = 0.05
    plot_mode("random", fpr, output_dir)
    plot_mode("tail", fpr, output_dir)


if __name__ == "__main__":
    main()
