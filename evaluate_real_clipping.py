"""Evaluate raw vs clipped detectors on real scored text.

Inputs are JSONL score files from `score_real_text.py`.

The script:

  1. Splits examples 50/50 by sample_id into tune/test.
  2. Tunes one-sided clipping bounds on the tune split.
  3. Sets FPR thresholds from clean human test scores.
  4. Evaluates TPR at 1% and 5% FPR on contaminated LLM test scores.
  5. Plots contamination-ratio curves for random and tail contamination.

Example:

  python evaluate_real_clipping.py \
    --clean real_data/xsum/Qwen__Qwen2.5-0.5B/clean_scores.jsonl \
    --contaminated real_data/xsum/Qwen__Qwen2.5-0.5B/contaminated_scores.jsonl \
    --output-dir real_results/xsum_qwen
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "log_likelihood": "LogLik",
    "rank": "Rank",
    "log_rank": "LogRank",
    "entropy": "Entropy",
    "entropy_gap": "Entropy-gap",
    "binoculars": "Binoculars",
}

COLORS = {
    "log_likelihood": "#d00000",
    "rank": "#8c8c8c",
    "log_rank": "#00a6c8",
    "entropy": "#55a630",
    "entropy_gap": "#00851b",
    "binoculars": "#8a5a00",
}

GENERIC_METHOD_FEATURES = {
    "log_likelihood": "logp_a",
    "rank": "rank_a",
    "log_rank": "logrank_a",
    "entropy": "entropy_a",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_ids(rows: list[dict[str, Any]], seed: int) -> tuple[set[int], set[int]]:
    ids = sorted({int(row["sample_id"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = max(1, len(ids) // 2)
    tune = set(ids[:cut])
    test = set(ids[cut:])
    if not test:
        test = tune
    return tune, test


def filter_ids(rows: list[dict[str, Any]], ids: set[int]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["sample_id"]) in ids]


def clean_by_label(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("label") == label and row.get("contamination_mode") == "none"]


def contaminated_subset(
    rows: list[dict[str, Any]],
    mode: str,
    ratio: float,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("contamination_mode") == mode
        and abs(float(row.get("contamination_ratio", 0.0)) - ratio) < 1e-12
    ]


def available_methods(rows: list[dict[str, Any]]) -> list[str]:
    methods = ["log_likelihood", "rank", "log_rank", "entropy", "entropy_gap"]
    if rows and "cross_entropy_ba" in rows[0].get("token_features", {}):
        methods.append("binoculars")
    return methods


def token_array(row: dict[str, Any], feature: str) -> np.ndarray:
    return np.asarray(row["token_features"][feature], dtype=float)


def raw_doc_score(row: dict[str, Any], method: str) -> float:
    features = row["token_features"]
    if method in GENERIC_METHOD_FEATURES:
        return float(token_array(row, GENERIC_METHOD_FEATURES[method]).mean())
    if method == "entropy_gap":
        nll = -token_array(row, "logp_a")
        entropy = token_array(row, "entropy_a")
        return float((nll - entropy).mean())
    if method == "binoculars":
        nll = -token_array(row, "logp_a")
        xent = token_array(row, "cross_entropy_ba")
        return float(nll.mean() / (xent.mean() + 1e-12))
    raise KeyError(method)


def raw_doc_scores(rows: list[dict[str, Any]], method: str) -> np.ndarray:
    return np.asarray([raw_doc_score(row, method) for row in rows], dtype=float)


def orientation(human_rows: list[dict[str, Any]], llm_rows: list[dict[str, Any]], method: str) -> int:
    h = raw_doc_scores(human_rows, method)
    m = raw_doc_scores(llm_rows, method)
    return 1 if m.mean() >= h.mean() else -1


def oriented_raw_scores(rows: list[dict[str, Any]], method: str, direction: int) -> np.ndarray:
    return direction * raw_doc_scores(rows, method)


def clipped_doc_score(row: dict[str, Any], method: str, direction: int, spec: dict[str, float]) -> float:
    if not spec:
        return direction * raw_doc_score(row, method)

    if method in GENERIC_METHOD_FEATURES:
        local = direction * token_array(row, GENERIC_METHOD_FEATURES[method])
        local = np.maximum(local, spec["lower"])
        return float(local.mean())

    if method == "entropy_gap":
        nll = -token_array(row, "logp_a")
        entropy = token_array(row, "entropy_a")
        nll = np.minimum(nll, spec["nll_upper"])
        return float(direction * (nll - entropy).mean())

    if method == "binoculars":
        nll = -token_array(row, "logp_a")
        xent = token_array(row, "cross_entropy_ba")
        nll = np.minimum(nll, spec["nll_upper"])
        return float(direction * (nll.mean() / (xent.mean() + 1e-12)))

    raise KeyError(method)


def clipped_scores(rows: list[dict[str, Any]], method: str, direction: int, spec: dict[str, float]) -> np.ndarray:
    return np.asarray([clipped_doc_score(row, method, direction, spec) for row in rows], dtype=float)


def tpr_at_fpr(human_scores: np.ndarray, llm_scores: np.ndarray, fpr: float) -> float:
    if len(human_scores) == 0 or len(llm_scores) == 0:
        return float("nan")
    threshold = np.quantile(human_scores, 1.0 - fpr)
    return float(np.mean(llm_scores >= threshold))


def tune_spec(
    method: str,
    direction: int,
    tune_human: list[dict[str, Any]],
    tune_llm_clean: list[dict[str, Any]],
    tune_contam: list[dict[str, Any]],
) -> dict[str, float]:
    """Tune a one-sided clipping bound on tune split."""
    if not tune_contam:
        return {}

    raw_h = oriented_raw_scores(tune_human, method, direction)
    raw_clean_m = oriented_raw_scores(tune_llm_clean, method, direction)
    raw_contam_m = oriented_raw_scores(tune_contam, method, direction)
    best_spec: dict[str, float] = {}
    best_score = (
        0.50 * tpr_at_fpr(raw_h, raw_contam_m, 0.05)
        + 0.35 * tpr_at_fpr(raw_h, raw_contam_m, 0.01)
        + 0.15 * tpr_at_fpr(raw_h, raw_clean_m, 0.05)
    )

    candidates: list[dict[str, float]] = [{}]
    if method in GENERIC_METHOD_FEATURES:
        vals = np.concatenate(
            [
                direction * token_array(row, GENERIC_METHOD_FEATURES[method])
                for row in tune_human + tune_llm_clean
            ]
        )
        for q in [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15]:
            candidates.append({"lower": float(np.quantile(vals, q))})
    elif method in {"entropy_gap", "binoculars"}:
        vals = np.concatenate([-token_array(row, "logp_a") for row in tune_human + tune_llm_clean])
        for q in [0.85, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999]:
            candidates.append({"nll_upper": float(np.quantile(vals, q))})

    for spec in candidates:
        h = clipped_scores(tune_human, method, direction, spec)
        clean_m = clipped_scores(tune_llm_clean, method, direction, spec)
        contam_m = clipped_scores(tune_contam, method, direction, spec)
        score = (
            0.50 * tpr_at_fpr(h, contam_m, 0.05)
            + 0.35 * tpr_at_fpr(h, contam_m, 0.01)
            + 0.15 * tpr_at_fpr(h, clean_m, 0.05)
        )
        if score > best_score:
            best_score = score
            best_spec = spec
    return best_spec


def evaluate_mode(
    methods: list[str],
    mode: str,
    ratios: list[float],
    tune_clean: list[dict[str, Any]],
    tune_contam_all: list[dict[str, Any]],
    test_clean: list[dict[str, Any]],
    test_contam_all: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float]]]]:
    tune_human = clean_by_label(tune_clean, "human")
    tune_llm = clean_by_label(tune_clean, "llm")
    test_human = clean_by_label(test_clean, "human")
    test_llm = clean_by_label(test_clean, "llm")

    rows = []
    curves = {
        method: {"raw_1": [], "clip_1": [], "raw_5": [], "clip_5": []}
        for method in methods
    }

    for method in methods:
        direction = orientation(tune_human, tune_llm, method)
        tune_contam_25 = contaminated_subset(tune_contam_all, mode, 0.25)
        if not tune_contam_25:
            # Fall back to the middle available ratio if 25% was not generated.
            available = sorted(
                {
                    float(row["contamination_ratio"])
                    for row in tune_contam_all
                    if row.get("contamination_mode") == mode
                }
            )
            fallback = available[len(available) // 2] if available else 0.0
            tune_contam_25 = contaminated_subset(tune_contam_all, mode, fallback)

        spec = tune_spec(method, direction, tune_human, tune_llm, tune_contam_25)
        test_h_raw = oriented_raw_scores(test_human, method, direction)
        test_h_clip = clipped_scores(test_human, method, direction, spec)

        for ratio in ratios:
            test_contam = contaminated_subset(test_contam_all, mode, ratio)
            raw_m = oriented_raw_scores(test_contam, method, direction)
            clip_m = clipped_scores(test_contam, method, direction, spec)
            raw_1 = tpr_at_fpr(test_h_raw, raw_m, 0.01)
            raw_5 = tpr_at_fpr(test_h_raw, raw_m, 0.05)
            clip_1 = tpr_at_fpr(test_h_clip, clip_m, 0.01)
            clip_5 = tpr_at_fpr(test_h_clip, clip_m, 0.05)

            curves[method]["raw_1"].append(raw_1)
            curves[method]["clip_1"].append(clip_1)
            curves[method]["raw_5"].append(raw_5)
            curves[method]["clip_5"].append(clip_5)
            rows.append(
                {
                    "mode": mode,
                    "method": method,
                    "ratio": ratio,
                    "direction": direction,
                    "clip_spec": json.dumps(spec),
                    "raw_tpr_at_1_fpr": raw_1,
                    "clipped_tpr_at_1_fpr": clip_1,
                    "raw_tpr_at_5_fpr": raw_5,
                    "clipped_tpr_at_5_fpr": clip_5,
                    "n_test_human": len(test_human),
                    "n_test_llm": len(test_contam),
                }
            )

    return rows, curves


def plot_curves(
    methods: list[str],
    mode: str,
    ratios: list[float],
    curves: dict[str, dict[str, list[float]]],
    output_dir: Path,
    fpr_label: str,
) -> None:
    key_raw = "raw_1" if fpr_label == "1" else "raw_5"
    key_clip = "clip_1" if fpr_label == "1" else "clip_5"

    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=160)
    for method in methods:
        label = METHOD_LABELS[method]
        color = COLORS[method]
        ax.plot(
            ratios,
            curves[method][key_raw],
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=color,
            alpha=0.45,
            label=f"{label} raw",
        )
        ax.plot(
            ratios,
            curves[method][key_clip],
            linestyle="-",
            marker="o",
            markersize=4,
            linewidth=2.0,
            color=color,
            label=f"{label} clipped",
        )

    ax.set_xlim(min(ratios), max(ratios))
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{int(round(r * 100))}%" for r in ratios], rotation=35)
    ax.set_xlabel("Human contamination ratio in LLM documents")
    ax.set_ylabel(f"TPR at {fpr_label}% FPR")
    ax.set_title(f"Real XSum robustness: {mode} contamination")
    ax.grid(True, color="#d9d9d9", linewidth=1.0)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.92, ncol=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{mode}_contamination_tpr_at_{fpr_label}_fpr.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True)
    parser.add_argument("--contaminated", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=21)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_rows = read_jsonl(Path(args.clean))
    contaminated_rows = read_jsonl(Path(args.contaminated))

    tune_ids, test_ids = split_ids(clean_rows, args.seed)
    tune_clean = filter_ids(clean_rows, tune_ids)
    test_clean = filter_ids(clean_rows, test_ids)
    tune_contam = filter_ids(contaminated_rows, tune_ids)
    test_contam = filter_ids(contaminated_rows, test_ids)

    methods = available_methods(clean_rows)
    ratios = sorted({float(row["contamination_ratio"]) for row in contaminated_rows})
    modes = sorted({row["contamination_mode"] for row in contaminated_rows if row["contamination_mode"] != "none"})
    output_dir = Path(args.output_dir)

    print(f"methods: {', '.join(methods)}")
    print(f"tune sample_ids: {len(tune_ids)}, test sample_ids: {len(test_ids)}")
    print(f"ratios: {ratios}")

    all_rows = []
    for mode in modes:
        metric_rows, curves = evaluate_mode(
            methods,
            mode,
            ratios,
            tune_clean,
            tune_contam,
            test_clean,
            test_contam,
        )
        all_rows.extend(metric_rows)
        plot_curves(methods, mode, ratios, curves, output_dir, "1")
        plot_curves(methods, mode, ratios, curves, output_dir, "5")

    write_csv(output_dir / "metrics.csv", all_rows)
    print(f"wrote {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
