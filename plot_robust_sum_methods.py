"""Plot paper-style FPR curves for the toy clipped-score experiment.

Run:

    python plot_robust_sum_methods.py

Outputs:

    plots/random_contamination_fpr_curve.png
    plots/tail_contamination_fpr_curve.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robust_sum_methods_toy import (
    contaminate_llm_with_human,
    orient_scores,
    sample_tokens,
    score_methods,
    tune_clip_specs,
    with_random_lengths,
)


N_TRIALS = 100


METHOD_LABELS = {
    "log_likelihood": "LogLik",
    "log_rank": "LogRank",
    "lrr_both": "LRR",
    "likelihood_ratio": "LikRatio",
    "binoculars": "Binoculars",
    "entropy_gap": "Entropy-gap",
}


COLORS = {
    "log_likelihood": "#d00000",
    "log_rank": "#00a6c8",
    "lrr_both": "#7a1fff",
    "likelihood_ratio": "#f8961e",
    "binoculars": "#8a5a00",
    "entropy_gap": "#00851b",
}


def tpr_curve(human_scores: np.ndarray, llm_scores: np.ndarray, fprs: np.ndarray) -> np.ndarray:
    """TPR values at each FPR, where larger scores mean more LLM-like."""
    return np.array(
        [
            np.mean(llm_scores >= np.quantile(human_scores, 1.0 - fpr))
            for fpr in fprs
        ]
    )


def oriented_method_scores(
    clean_human_scores: dict[str, np.ndarray],
    clean_llm_scores: dict[str, np.ndarray],
    test_human_scores: dict[str, np.ndarray],
    test_llm_scores: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Orient every method so larger score means more LLM-like."""
    out = {}
    for method in test_human_scores:
        _, _, direction = orient_scores(clean_human_scores[method], clean_llm_scores[method])
        out[method] = (direction * test_human_scores[method], direction * test_llm_scores[method])
    return out


def make_dataset(
    rng: np.random.Generator,
    n_docs: int,
    n_tokens: int,
    min_effective_tokens: int,
    max_effective_tokens: int,
):
    human = with_random_lengths(
        rng,
        sample_tokens(rng, "human", n_docs, n_tokens),
        min_effective_tokens,
        max_effective_tokens,
    )
    llm = with_random_lengths(
        rng,
        sample_tokens(rng, "llm", n_docs, n_tokens),
        min_effective_tokens,
        max_effective_tokens,
    )
    human_for_contamination = with_random_lengths(
        rng,
        sample_tokens(rng, "human", n_docs, n_tokens),
        min_effective_tokens,
        max_effective_tokens,
    )
    return human, llm, human_for_contamination


def plot_mode(contamination_mode: str, contamination: float, output_dir: Path) -> None:
    n_val = 1_000
    n_test = 2_000
    n_tokens = 80
    min_effective_tokens = 25
    max_effective_tokens = n_tokens
    fprs = np.array([1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001])

    raw_curves = {method: [] for method in METHOD_LABELS}
    clipped_curves = {method: [] for method in METHOD_LABELS}

    base_seed = 11 if contamination_mode == "random" else 17
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
        mixed_llm = contaminate_llm_with_human(
            rng,
            clean_test_llm,
            human_for_contamination,
            contamination=contamination,
            mode=contamination_mode,
        )

        raw_clean_human = score_methods(test_human)
        raw_clean_llm = score_methods(clean_test_llm)
        clipped_clean_human = score_methods(test_human, clip_specs)
        clipped_clean_llm = score_methods(clean_test_llm, clip_specs)

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
            raw_curves[method].append(tpr_curve(human_raw, llm_raw, fprs))
            clipped_curves[method].append(tpr_curve(human_clip, llm_clip, fprs))

    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=160)

    for method, label in METHOD_LABELS.items():
        raw_tpr = np.mean(raw_curves[method], axis=0)
        clip_tpr = np.mean(clipped_curves[method], axis=0)

        ax.plot(
            fprs,
            raw_tpr,
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=COLORS[method],
            alpha=0.45,
            label=f"{label} raw",
        )
        ax.plot(
            fprs,
            clip_tpr,
            linestyle="-",
            marker="o",
            markersize=4,
            linewidth=2.0,
            color=COLORS[method],
            label=f"{label} clipped",
        )

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlim(1.0, 0.001)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([1.0, 0.1, 0.01, 0.001])
    ax.set_xticklabels(["100%", "10%", "1%", "0.1%"])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Detection Accuracy / TPR")
    ax.set_title(
        f"{contamination_mode.title()} contamination, {contamination:.0%} human tokens "
        f"(mean over {N_TRIALS} trials)"
    )
    ax.grid(True, which="major", color="#d9d9d9", linewidth=1.0)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.92, ncol=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{contamination_mode}_contamination_fpr_curve.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    output_dir = Path("plots")
    contamination = 0.25
    plot_mode("random", contamination, output_dir)
    plot_mode("tail", contamination, output_dir)


if __name__ == "__main__":
    main()
