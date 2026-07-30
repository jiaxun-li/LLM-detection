"""Experiment configuration loading and validation.

JSON is intentionally used instead of environment-variable-only configuration so
every paper run has a complete, machine-readable specification.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


PAPER_MODELS = [
    "ibm-granite/granite-3.3-8b-base",
    "mistralai/Mistral-Small-24B-Base-2501",
    "Qwen/Qwen2.5-32B",
]
REPLICATION_MODELS = ["KoboldAI/GPT-NeoX-20B-Erebus"]
QWEN_SCALING_MODELS = [
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
]
DETECTORS = [
    "log_likelihood",
    "rank",
    "log_rank",
    "lrr",
    "entropy",
    "entropy_gap",
    "binoculars",
]


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load JSON and apply dotted ``key=value`` command-line overrides."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path.resolve())
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got {override!r}")
        dotted, raw = override.split("=", 1)
        value = _parse_override(raw)
        target = config
        keys = dotted.split(".")
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
    validate_config(config)
    return config


def _parse_override(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def resolved_run_config(
    config: dict[str, Any],
    dataset: str,
    model_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Return a deep-copied single dataset/model run configuration."""
    result = copy.deepcopy(config)
    result["run_id"] = run_id
    result["dataset"] = dataset
    result["target_model"] = model_id
    result["target_model_revision"] = result["models"].get("revision")
    result["target_tokenizer_revision"] = result["models"].get("tokenizer_revision")
    return result


def validate_config(config: dict[str, Any]) -> None:
    required = [
        "datasets",
        "models",
        "splits",
        "generation",
        "contamination",
        "scoring",
        "evaluation",
        "paths",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"configuration is missing sections: {missing}")

    counts = config["splits"]
    for key in ("clipping_tuning", "calibration", "test"):
        if int(counts.get(key, 0)) <= 0:
            raise ValueError(f"splits.{key} must be positive")
    seeds = config["generation"].get("seeds", [])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("generation.seeds must be a non-empty unique list")
    ratios = [float(x) for x in config["contamination"].get("ratios", [])]
    if not ratios or ratios[0] != 0.0 or any(x < 0 or x > 0.5 for x in ratios):
        raise ValueError("contamination.ratios must begin at 0 and lie in [0, 0.5]")
    if ratios != sorted(set(ratios)):
        raise ValueError("contamination.ratios must be sorted and unique")
    if int(config["contamination"].get("random_draws", 0)) <= 0:
        raise ValueError("contamination.random_draws must be positive")
    if config["generation"].get("backend") not in {"transformers", "vllm"}:
        raise ValueError("generation.backend must be transformers or vllm")
    if config["scoring"].get("dtype") not in {"bf16", "fp16", "fp32"}:
        raise ValueError("scoring.dtype must be bf16, fp16, or fp32")
    saved_top_k = int(config["scoring"].get("saved_top_k", 10))
    if saved_top_k < 2:
        raise ValueError("scoring.saved_top_k must be at least 2")
    save_pooled_hidden = config["scoring"].get(
        "save_mean_pooled_final_hidden_state", False
    )
    if not isinstance(save_pooled_hidden, bool):
        raise ValueError(
            "scoring.save_mean_pooled_final_hidden_state must be true or false"
        )
    fprs = [float(x) for x in config["evaluation"].get("target_fprs", [])]
    if fprs != [0.01, 0.05]:
        raise ValueError("evaluation.target_fprs must be [0.01, 0.05] for the primary study")


def total_examples(config: dict[str, Any]) -> int:
    return sum(
        int(config["splits"][key])
        for key in ("clipping_tuning", "calibration", "test")
    )
