"""Toy experiments for clipped sum-based LLM detection scores.

This script uses synthetic token-level distributions to compare unclipped and
clipped versions of:

  1. log-likelihood
  2. log-rank
  3. DetectLLM-style LRR, both-clipped
  4. likelihood ratio
  5. Binoculars-style perplexity / cross-perplexity
  6. entropy-gap, an entropy-calibrated perplexity score

The synthetic setup is intentionally simple:

  - Token scores follow power-law-like heavy-tailed distributions.
  - Human token features are heavier-tailed, so a small amount of tail-heavy
    human contamination can dominate an unbounded sum.
  - Clipping is one-sided, in the direction that hurts mostly-LLM documents.
  - Bounds are tuned on a validation objective, not fixed quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class TokenBatch:
    """Token-level synthetic features for many documents."""

    logp_a: np.ndarray
    logp_b: np.ndarray
    logrank_a: np.ndarray
    xent_ab: np.ndarray
    entropy_a: np.ndarray
    active: np.ndarray


def power_law_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    alpha: float,
    scale: float,
) -> np.ndarray:
    """Draw positive heavy-tailed noise.

    A Pareto tail is a simple scale-law toy model:

        P(X > t) proportional to t^{-alpha}

    Lower alpha means a heavier tail.
    """
    return scale * rng.pareto(alpha, size=shape)


def sample_tokens(rng: np.random.Generator, source: str, n_docs: int, n_tokens: int) -> TokenBatch:
    """Sample synthetic token features.

    `A` is the suspected/generator-like evaluator model.
    `B` is a human/reference-like model.
    """
    shape = (n_docs, n_tokens)

    if source == "llm":
        nll_a = 1.55 + power_law_noise(rng, shape, alpha=3.8, scale=0.55)
        nll_b = 1.95 + power_law_noise(rng, shape, alpha=3.4, scale=0.55)
        logrank_a = 0.85 + power_law_noise(rng, shape, alpha=3.7, scale=0.55)
        xent_ab = 2.05 + power_law_noise(rng, shape, alpha=3.8, scale=0.35)
    elif source == "human":
        nll_a = 2.10 + power_law_noise(rng, shape, alpha=2.2, scale=0.70)
        nll_b = 1.75 + power_law_noise(rng, shape, alpha=2.6, scale=0.60)
        logrank_a = 1.35 + power_law_noise(rng, shape, alpha=2.2, scale=0.75)
        xent_ab = 2.12 + power_law_noise(rng, shape, alpha=2.7, scale=0.40)
    else:
        raise ValueError(f"unknown source: {source}")

    # Domain/style effects make whole documents easier or harder to score.
    # This variance does not disappear when token scores are averaged.
    style = rng.normal(0.0, 0.35, size=(n_docs, 1))
    rank_style = rng.normal(0.0, 0.35, size=(n_docs, 1))
    xent_style = rng.normal(0.0, 0.25, size=(n_docs, 1))
    logp_a = -(nll_a + style)
    logp_b = -(nll_b + 0.65 * style + rng.normal(0.0, 0.20, size=(n_docs, 1)))
    logrank_a = logrank_a + rank_style
    xent_ab = xent_ab + xent_style
    # Entropy is a property of model A's predictive distribution for a context,
    # not the realized next token. In this toy, it is a noisy context-level
    # uncertainty term: informative enough for an entropy-gap test, but weaker
    # than a true two-model likelihood ratio.
    entropy_a = (
        1.70
        + 0.5 * power_law_noise(rng, shape, alpha=3.4, scale=0.55)
        + 0.1 * style
        + rng.normal(0.0, 0.50, size=shape)
    )

    return TokenBatch(
        logp_a=np.minimum(logp_a, -EPS),
        logp_b=np.minimum(logp_b, -EPS),
        logrank_a=np.maximum(logrank_a, EPS),
        xent_ab=np.maximum(xent_ab, EPS),
        entropy_a=np.maximum(entropy_a, EPS),
        active=np.ones(shape, dtype=bool),
    )


def contaminate_llm_with_human(
    rng: np.random.Generator,
    llm: TokenBatch,
    human: TokenBatch,
    contamination: float,
    mode: str = "tail",
) -> TokenBatch:
    """Replace an epsilon fraction of LLM token features with human features.

    `random` contamination is ordinary mixture noise.
    `tail` contamination inserts the most harmful human tokens per document.
    """
    if mode == "random":
        mask = (rng.random(llm.logp_a.shape) < contamination) & llm.active
    elif mode == "tail":
        harmfulness = (-human.logp_a) + human.logrank_a + np.maximum(-(human.logp_a - human.logp_b), 0.0)
        harmfulness = np.where(llm.active, harmfulness, -np.inf)
        mask = np.zeros(llm.logp_a.shape, dtype=bool)
        lengths = llm.active.sum(axis=1)
        for row, length in enumerate(lengths):
            k = int(round(float(length) * contamination))
            if k <= 0:
                continue
            chosen = np.argpartition(harmfulness[row], -k)[-k:]
            mask[row, chosen] = True
    else:
        raise ValueError(f"unknown contamination mode: {mode}")

    def mix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(mask, b, a)

    return TokenBatch(
        logp_a=mix(llm.logp_a, human.logp_a),
        logp_b=mix(llm.logp_b, human.logp_b),
        logrank_a=mix(llm.logrank_a, human.logrank_a),
        xent_ab=mix(llm.xent_ab, human.xent_ab),
        entropy_a=mix(llm.entropy_a, human.entropy_a),
        active=llm.active,
    )


def with_random_lengths(
    rng: np.random.Generator,
    batch: TokenBatch,
    min_tokens: int,
    max_tokens: int,
) -> TokenBatch:
    """Mask each document down to a random effective length."""
    n_docs, n_tokens = batch.logp_a.shape
    lengths = rng.integers(min_tokens, max_tokens + 1, size=n_docs)
    active = np.arange(n_tokens)[None, :] < lengths[:, None]
    return TokenBatch(
        logp_a=batch.logp_a,
        logp_b=batch.logp_b,
        logrank_a=batch.logrank_a,
        xent_ab=batch.xent_ab,
        entropy_a=batch.entropy_a,
        active=active,
    )


def active_values(values: np.ndarray, batch: TokenBatch) -> np.ndarray:
    return values[batch.active]


def clip_one_sided(values: np.ndarray, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    if lower is not None:
        values = np.maximum(values, lower)
    if upper is not None:
        values = np.minimum(values, upper)
    return values


def score_methods(batch: TokenBatch, clip_specs: dict[str, dict[str, float]] | None = None) -> dict[str, np.ndarray]:
    """Return one document-level score per method."""
    logp = batch.logp_a
    nll = -batch.logp_a
    logrank = batch.logrank_a
    lrt = batch.logp_a - batch.logp_b
    xent = batch.xent_ab
    entropy = batch.entropy_a
    clip_specs = clip_specs or {}

    active = batch.active.astype(float)
    lengths = active.sum(axis=1)

    def masked_mean(values: np.ndarray) -> np.ndarray:
        return (values * active).sum(axis=1) / (lengths + EPS)

    ll_logp = clip_one_sided(logp, lower=clip_specs.get("log_likelihood", {}).get("logp_lower"))
    rank_vals = clip_one_sided(logrank, upper=clip_specs.get("log_rank", {}).get("logrank_upper"))

    lrr_both_spec = clip_specs.get("lrr_both", {})
    lrr_both_logp = clip_one_sided(logp, lower=lrr_both_spec.get("logp_lower"))
    lrr_both_rank = clip_one_sided(logrank, upper=lrr_both_spec.get("logrank_upper"))

    lrt_vals = clip_one_sided(lrt, lower=clip_specs.get("likelihood_ratio", {}).get("lrt_lower"))

    bino_spec = clip_specs.get("binoculars", {})
    bino_nll = clip_one_sided(nll, upper=bino_spec.get("nll_upper"))
    # Human insertions in this toy mostly hurt Binoculars through high NLL.
    # We leave the denominator unclipped to keep the intervention one-sided.
    bino_xent = xent

    entropy_spec = clip_specs.get("entropy_gap", {})
    entropy_nll = clip_one_sided(nll, upper=entropy_spec.get("nll_upper"))

    scores = {
        "log_likelihood": masked_mean(ll_logp),
        "log_rank": masked_mean(rank_vals),
        "lrr_both": -((lrr_both_logp * active).sum(axis=1))
        / ((lrr_both_rank * active).sum(axis=1) + EPS),
        "likelihood_ratio": masked_mean(lrt_vals),
        "binoculars": masked_mean(bino_nll) / (masked_mean(bino_xent) + EPS),
        "entropy_gap": masked_mean(entropy_nll - entropy),
    }
    return scores


def orient_scores(human_scores: np.ndarray, llm_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Make larger scores mean 'more LLM-like'."""
    direction = 1 if llm_scores.mean() >= human_scores.mean() else -1
    return direction * human_scores, direction * llm_scores, direction


def auroc(human_scores: np.ndarray, llm_scores: np.ndarray) -> float:
    """AUROC where larger score means more LLM-like."""
    scores = np.concatenate([human_scores, llm_scores])
    labels = np.concatenate([np.zeros_like(human_scores), np.ones_like(llm_scores)])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def tpr_at_fpr(human_scores: np.ndarray, llm_scores: np.ndarray, fpr: float = 0.05) -> float:
    """TPR at fixed FPR where larger score means more LLM-like."""
    threshold = np.quantile(human_scores, 1.0 - fpr)
    return float(np.mean(llm_scores >= threshold))


def objective(clean_h: np.ndarray, clean_m: np.ndarray, contam_h: np.ndarray, contam_m: np.ndarray) -> float:
    """Validation objective for bound tuning.

    The objective emphasizes contaminated LLM recall at low false-positive rate,
    while keeping some pressure on clean separability.
    """
    _, _, direction = orient_scores(clean_h, clean_m)
    clean_h = direction * clean_h
    clean_m = direction * clean_m
    contam_h = direction * contam_h
    contam_m = direction * contam_m
    clean_auc = auroc(clean_h, clean_m)
    contam_auc = auroc(contam_h, contam_m)
    contam_tpr = tpr_at_fpr(contam_h, contam_m, fpr=0.05)
    return 0.60 * contam_tpr + 0.25 * contam_auc + 0.15 * clean_auc


def tune_clip_specs(
    val_human: TokenBatch,
    val_llm: TokenBatch,
    val_human_for_contamination: TokenBatch,
    rng: np.random.Generator,
    contamination: float = 0.25,
    contamination_mode: str = "tail",
) -> dict[str, dict[str, float]]:
    """Tune one-sided clipping bounds on contaminated validation data."""
    val_mixed = contaminate_llm_with_human(
        rng,
        val_llm,
        val_human_for_contamination,
        contamination=contamination,
        mode=contamination_mode,
    )

    raw_clean_h = score_methods(val_human)
    raw_clean_m = score_methods(val_llm)
    raw_contam_h = score_methods(val_human)
    raw_contam_m = score_methods(val_mixed)

    pooled_logp = np.concatenate(
        [active_values(val_human.logp_a, val_human), active_values(val_llm.logp_a, val_llm)]
    )
    pooled_logrank = np.concatenate(
        [active_values(val_human.logrank_a, val_human), active_values(val_llm.logrank_a, val_llm)]
    )
    pooled_lrt = np.concatenate(
        [
            active_values(val_human.logp_a - val_human.logp_b, val_human),
            active_values(val_llm.logp_a - val_llm.logp_b, val_llm),
        ]
    )
    pooled_nll = -pooled_logp

    lower_quantiles = [None, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15]
    upper_quantiles = [None, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999]

    def best_for_method(method: str, candidates: list[dict[str, dict[str, float]]]) -> dict[str, float]:
        best_spec = {}
        best_score = objective(
            raw_clean_h[method],
            raw_clean_m[method],
            raw_contam_h[method],
            raw_contam_m[method],
        )
        for spec in candidates:
            clean_h = score_methods(val_human, spec)[method]
            clean_m = score_methods(val_llm, spec)[method]
            contam_h = score_methods(val_human, spec)[method]
            contam_m = score_methods(val_mixed, spec)[method]
            score = objective(clean_h, clean_m, contam_h, contam_m)
            if score > best_score:
                best_score = score
                best_spec = spec[method]
        return best_spec

    logp_candidates = [
        {"log_likelihood": {"logp_lower": float(np.quantile(pooled_logp, q))}}
        for q in lower_quantiles
        if q is not None
    ]
    logp_candidates.insert(0, {"log_likelihood": {}})

    logrank_candidates = [
        {"log_rank": {"logrank_upper": float(np.quantile(pooled_logrank, q))}}
        for q in upper_quantiles
        if q is not None
    ]
    logrank_candidates.insert(0, {"log_rank": {}})

    lrt_candidates = [
        {"likelihood_ratio": {"lrt_lower": float(np.quantile(pooled_lrt, q))}}
        for q in lower_quantiles
        if q is not None
    ]
    lrt_candidates.insert(0, {"likelihood_ratio": {}})

    bino_candidates = [
        {"binoculars": {"nll_upper": float(np.quantile(pooled_nll, q))}}
        for q in upper_quantiles
        if q is not None
    ]
    bino_candidates.insert(0, {"binoculars": {}})

    lrr_both_candidates = [{"lrr_both": {}}]
    for low_q in lower_quantiles:
        for high_q in upper_quantiles:
            spec = {}
            if low_q is not None:
                spec["logp_lower"] = float(np.quantile(pooled_logp, low_q))
            if high_q is not None:
                spec["logrank_upper"] = float(np.quantile(pooled_logrank, high_q))
            if spec:
                lrr_both_candidates.append({"lrr_both": spec})

    entropy_gap_candidates = [
        {"entropy_gap": {"nll_upper": float(np.quantile(pooled_nll, q))}}
        for q in upper_quantiles
        if q is not None
    ]
    entropy_gap_candidates.insert(0, {"entropy_gap": {}})

    return {
        "log_likelihood": best_for_method("log_likelihood", logp_candidates),
        "log_rank": best_for_method("log_rank", logrank_candidates),
        "lrr_both": best_for_method("lrr_both", lrr_both_candidates),
        "likelihood_ratio": best_for_method("likelihood_ratio", lrt_candidates),
        "binoculars": best_for_method("binoculars", bino_candidates),
        "entropy_gap": best_for_method("entropy_gap", entropy_gap_candidates),
    }


def evaluate(
    clean_human_scores: dict[str, np.ndarray],
    clean_llm_scores: dict[str, np.ndarray],
    test_human_scores: dict[str, np.ndarray],
    test_llm_scores: dict[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    results = {}
    for name in test_human_scores:
        _, _, direction = orient_scores(clean_human_scores[name], clean_llm_scores[name])
        h = direction * test_human_scores[name]
        m = direction * test_llm_scores[name]
        results[name] = (auroc(h, m), tpr_at_fpr(h, m, fpr=0.05))
    return results


def print_table(contamination: float, raw: dict[str, tuple[float, float]], clip: dict[str, tuple[float, float]]) -> None:
    print(f"\nHuman contamination in LLM documents: {contamination:>4.0%}")
    print("-" * 78)
    print(f"{'method':<20} {'AUROC raw':>10} {'AUROC clip':>11} {'TPR@5 raw':>11} {'TPR@5 clip':>12}")
    print("-" * 78)
    for name in raw:
        raw_auc, raw_tpr = raw[name]
        clip_auc, clip_tpr = clip[name]
        print(f"{name:<20} {raw_auc:10.3f} {clip_auc:11.3f} {raw_tpr:11.3f} {clip_tpr:12.3f}")


def run_experiment(
    rng: np.random.Generator,
    contamination_mode: str,
    n_val: int,
    n_test: int,
    n_tokens: int,
    min_effective_tokens: int,
    max_effective_tokens: int,
    contamination_grid: list[float],
) -> None:
    val_human = with_random_lengths(
        rng,
        sample_tokens(rng, "human", n_val, n_tokens),
        min_effective_tokens,
        max_effective_tokens,
    )
    val_llm = with_random_lengths(
        rng,
        sample_tokens(rng, "llm", n_val, n_tokens),
        min_effective_tokens,
        max_effective_tokens,
    )
    val_human_for_contamination = with_random_lengths(
        rng,
        sample_tokens(rng, "human", n_val, n_tokens),
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

    test_human = with_random_lengths(
        rng,
        sample_tokens(rng, "human", n_test, n_tokens),
        min_effective_tokens,
        max_effective_tokens,
    )
    clean_test_llm = with_random_lengths(
        rng,
        sample_tokens(rng, "llm", n_test, n_tokens),
        min_effective_tokens,
        max_effective_tokens,
    )
    human_for_contamination = with_random_lengths(
        rng,
        sample_tokens(rng, "human", n_test, n_tokens),
        min_effective_tokens,
        max_effective_tokens,
    )

    clean_raw_human = score_methods(test_human)
    clean_raw_llm = score_methods(clean_test_llm)
    clean_clip_human = score_methods(test_human, clip_specs)
    clean_clip_llm = score_methods(clean_test_llm, clip_specs)

    print("\n" + "=" * 78)
    print(f"Contamination mode: {contamination_mode}")
    print("=" * 78)
    print("learned clipping specs:")
    for method, spec in clip_specs.items():
        print(f"  {method:<18} {spec if spec else 'no clipping'}")

    for contamination in contamination_grid:
        mixed_llm = contaminate_llm_with_human(
            rng,
            clean_test_llm,
            human_for_contamination,
            contamination=contamination,
            mode=contamination_mode,
        )
        raw = evaluate(
            clean_raw_human,
            clean_raw_llm,
            score_methods(test_human),
            score_methods(mixed_llm),
        )
        clip = evaluate(
            clean_clip_human,
            clean_clip_llm,
            score_methods(test_human, clip_specs),
            score_methods(mixed_llm, clip_specs),
        )
        print_table(contamination, raw, clip)


def main() -> None:
    rng = np.random.default_rng(7)

    n_val = 1_000
    n_test = 2_000
    n_tokens = 80
    min_effective_tokens = 25
    max_effective_tokens = n_tokens
    contamination_grid = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    print("Synthetic robust LLM detection experiment")
    print(f"documents per class: {n_test:,}, max tokens per document: {n_tokens}")
    print(f"random effective length: {min_effective_tokens}-{max_effective_tokens} tokens")
    print("synthetic scores: power-law / Pareto heavy tails")
    print("clipping: one-sided bounds tuned on validation contamination = 25%")

    for mode in ["random", "tail"]:
        run_experiment(
            rng,
            mode,
            n_val,
            n_test,
            n_tokens,
            min_effective_tokens,
            max_effective_tokens,
            contamination_grid,
        )


if __name__ == "__main__":
    main()
