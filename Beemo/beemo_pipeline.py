"""Stage functions for the Beemo benchmark run."""

from __future__ import annotations

import gc
import datetime as dt
import csv
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from llm_detection.data import data_row_key
from llm_detection.io import atomic_write_json, iter_jsonl
from llm_detection.runtime import accelerator_info, software_versions
from llm_detection.scoring import BinocularsScorer, TargetModelScorer, score_jsonl

from Beemo.beemo_data import load_official_beemo, prepare_from_records
from Beemo.beemo_evaluation import evaluate_beemo
from Beemo.beemo_plot import plot_beemo


PRIMARY_SCORER_KEYS = ("gpt2_xl", "opt_1_3b", "falcon_7b", "qwen2_7b")


def selected_scoring_models(
    config: dict[str, Any], *, include_granite: bool
) -> list[dict[str, Any]]:
    selected = []
    for model in config["scoring_models"]:
        if bool(model.get("enabled", False)) or (
            include_granite and model.get("optional_name") == "granite"
        ):
            selected.append(model)
    return selected


def load_beemo_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"protocol_name", "dataset", "splits", "scoring_models", "binoculars", "scoring", "evaluation", "paths"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Beemo config is missing sections: {missing}")
    counts = sum(int(config["splits"][name]) for name in ("clipping_tuning", "calibration", "test"))
    if counts != int(config["dataset"]["expected_records"]):
        raise ValueError("Beemo split counts must sum to dataset.expected_records")
    if [float(value) for value in config["evaluation"]["target_fprs"]] != [0.01, 0.05]:
        raise ValueError("Beemo target FPRs must be 1% and 5%")
    models = config["scoring_models"]
    if not isinstance(models, list) or not models:
        raise ValueError("Beemo scoring_models must be a non-empty list")
    keys = [str(model.get("key", "")) for model in models]
    ids = [str(model.get("id", "")) for model in models]
    if any(not key or not model_id for key, model_id in zip(keys, ids)):
        raise ValueError("every Beemo scoring model requires key and id")
    if len(keys) != len(set(keys)) or len(ids) != len(set(ids)):
        raise ValueError("Beemo scoring model keys and ids must be unique")
    enabled_keys = {model["key"] for model in models if model.get("enabled", False)}
    if enabled_keys != set(PRIMARY_SCORER_KEYS):
        raise ValueError(
            "Beemo primary scorers must be GPT2-XL, OPT-1.3B, Falcon-7B, and Qwen2-7B"
        )
    if config["scoring"].get("context_policy") != "detectllm_output_only":
        raise ValueError("Beemo single-model scorers must use output-only scoring")
    if config["binoculars"].get("context_policy") != "binoculars_output_only_512":
        raise ValueError("Beemo Binoculars must use its output-only 512-token policy")
    return config


def run_paths(workspace: str | Path, config: dict[str, Any], run_id: str) -> tuple[Path, Path]:
    root = Path(workspace).resolve()
    return root / config["paths"]["runs"] / run_id, root / config["paths"]["results"] / run_id


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def initial_manifest(
    run_id: str,
    workspace: str | Path,
    config: dict[str, Any],
    *,
    limit: int | None,
    skip_binoculars: bool,
    include_granite: bool,
    debug_only: bool,
) -> dict[str, Any]:
    root = Path(workspace).resolve()

    def git_value(*args: str) -> str | None:
        try:
            process = subprocess.run(
                ["git", "-c", f"safe.directory={root.as_posix()}", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            return process.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    dirty = git_value("status", "--porcelain")
    return {
        "run_id": run_id,
        "protocol": config["protocol_name"],
        "result_label": config["result_label"],
        "debug_only": bool(debug_only or limit is not None),
        "workspace": str(Path(workspace).resolve()),
        "creation_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "dirty_worktree": None if dirty is None else bool(dirty),
        "protocol_config": config,
        "limit_records": limit,
        "binoculars_included": not skip_binoculars,
        "granite_included": include_granite,
        "selected_scorers": [
            {"key": model["key"], "id": model["id"]}
            for model in selected_scoring_models(
                config, include_granite=include_granite
            )
        ],
        "software_versions": software_versions(),
        "accelerator": accelerator_info(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "completion_status": "running",
        "completed_stages": [],
        "outputs": {},
    }


def validate_resume_manifest(
    manifest: dict[str, Any],
    config: dict[str, Any],
    *,
    run_id: str,
    limit: int | None,
    skip_binoculars: bool,
    include_granite: bool,
) -> None:
    expected = {
        "run_id": run_id,
        "protocol": config["protocol_name"],
        "protocol_config": config,
        "limit_records": limit,
        "binoculars_included": not skip_binoculars,
        "granite_included": include_granite,
        "selected_scorers": [
            {"key": model["key"], "id": model["id"]}
            for model in selected_scoring_models(
                config, include_granite=include_granite
            )
        ],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"existing Beemo manifest disagrees on {key}; choose a new run id")


def prepare_stage(
    run_dir: Path,
    config: dict[str, Any],
    *,
    limit: int | None,
) -> dict[str, Any]:
    records, fingerprint, resolved_revision = load_official_beemo(config)
    data_path = run_dir / "data.jsonl"
    summary = prepare_from_records(
        records,
        data_path,
        config,
        dataset_fingerprint=fingerprint,
        dataset_resolved_revision=resolved_revision,
        limit=limit,
    )
    atomic_write_json(run_dir / "prepare.complete.json", summary)
    return {"data": str(data_path), "prepare_summary": summary}


def score_stage(
    run_dir: Path,
    config: dict[str, Any],
    *,
    skip_binoculars: bool,
    include_granite: bool,
) -> dict[str, Any]:
    data_path = run_dir / "data.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"prepared Beemo data is missing: {data_path}")
    score_dir = run_dir / "target_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    target_counts: dict[str, int] = {}
    target_paths: dict[str, str] = {}
    resolved_scorers: dict[str, dict[str, Any]] = {}
    for model_spec in selected_scoring_models(
        config, include_granite=include_granite
    ):
        scorer_key = str(model_spec["key"])
        target_path = score_dir / f"{scorer_key}.jsonl"
        scorer_config = {
            **config["scoring"],
            **model_spec.get("scoring", {}),
            "scorer_key": scorer_key,
        }
        target = TargetModelScorer(
            model_spec["id"],
            model_spec.get("revision"),
            model_spec.get("tokenizer_revision"),
            scorer_config,
        )
        target_count = score_jsonl(
            data_path, target_path, target, scorer_config
        )
        target_counts[scorer_key] = target_count
        target_paths[scorer_key] = str(target_path)
        resolved_scorers[scorer_key] = {
            "model_id": model_spec["id"],
            "model_revision": target.resolved_revision,
            "tokenizer_revision": target.resolved_tokenizer_revision,
            "context_policy": target.context_policy,
        }
        del target
        _release_cuda()
    binoculars_path = run_dir / "binoculars_scores.jsonl"
    binoculars_count = None
    resolved: dict[str, Any] = {}
    if not skip_binoculars:
        model_config = {"binoculars": config["binoculars"]}
        binoculars = BinocularsScorer(model_config, config["scoring"])
        binoculars_count = score_jsonl(
            data_path, binoculars_path, binoculars, config["scoring"]
        )
        resolved.update(
            {
                "binoculars_performer_revision": binoculars.performer_resolved_revision,
                "binoculars_observer_revision": binoculars.observer_resolved_revision,
            }
        )
        del binoculars
        _release_cuda()
    expected = sum(1 for _ in iter_jsonl(data_path))
    if any(count != expected for count in target_counts.values()) or (
        binoculars_count is not None and binoculars_count != expected
    ):
        raise ValueError("Beemo score counts do not match prepared data")
    marker = {
        "data_rows": expected,
        "target_score_rows": target_counts,
        "binoculars_score_rows": binoculars_count,
        "resolved_scorers": resolved_scorers,
        **resolved,
    }
    atomic_write_json(run_dir / "score.complete.json", marker)
    return {
        "target_scores": target_paths,
        "binoculars_scores": str(binoculars_path) if not skip_binoculars else None,
        "score_summary": marker,
    }


def evaluate_stage(
    run_dir: Path,
    results_dir: Path,
    config: dict[str, Any],
    *,
    bootstrap_repetitions: int | None,
    skip_binoculars: bool,
    include_granite: bool,
) -> dict[str, Any]:
    target_paths = {
        str(model["key"]): run_dir / "target_scores" / f"{model['key']}.jsonl"
        for model in selected_scoring_models(
            config, include_granite=include_granite
        )
    }
    binoculars_path = run_dir / "binoculars_scores.jsonl"
    missing_targets = [str(path) for path in target_paths.values() if not path.is_file()]
    if missing_targets:
        raise FileNotFoundError(f"Beemo target scores are missing: {missing_targets}")
    if not skip_binoculars and not binoculars_path.is_file():
        raise FileNotFoundError(f"Beemo Binoculars scores are missing: {binoculars_path}")
    summary = evaluate_beemo(
        target_paths,
        results_dir,
        config,
        binoculars_path if not skip_binoculars else None,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    atomic_write_json(results_dir / "evaluate.complete.json", summary)
    return {
        "metrics": str(results_dir / "metrics.csv"),
        "subgroup_metrics": str(results_dir / "subgroup_metrics.csv"),
        "frozen_specs": str(results_dir / "frozen_specs.json"),
        "evaluation_summary": summary,
    }


def plot_stage(results_dir: Path) -> dict[str, Any]:
    metrics_path = results_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Beemo metrics are missing: {metrics_path}")
    plots = plot_beemo(metrics_path, results_dir)
    marker = {"plots": plots}
    atomic_write_json(results_dir / "plot.complete.json", marker)
    return marker


def validate_artifacts(
    run_dir: Path,
    results_dir: Path,
    config: dict[str, Any],
    *,
    skip_binoculars: bool,
    include_granite: bool,
) -> dict[str, Any]:
    data_rows = list(iter_jsonl(run_dir / "data.jsonl"))
    data_keys = [data_row_key(row) for row in data_rows]
    if len(data_keys) != len(set(data_keys)):
        raise ValueError("Beemo prepared data contains duplicate keys")
    selected_models = selected_scoring_models(
        config, include_granite=include_granite
    )
    target_counts: dict[str, int] = {}
    for model in selected_models:
        scorer_key = str(model["key"])
        target_rows = list(
            iter_jsonl(run_dir / "target_scores" / f"{scorer_key}.jsonl")
        )
        if len(data_rows) != len(target_rows):
            raise ValueError(
                f"Beemo data and {scorer_key} score counts disagree"
            )
        if {data_row_key(row) for row in target_rows} != set(data_keys):
            raise ValueError(
                f"Beemo {scorer_key} score keys disagree with prepared data"
            )
        if {row.get("scorer_key") for row in target_rows} != {scorer_key}:
            raise ValueError(f"Beemo {scorer_key} rows have the wrong scorer key")
        if {row.get("scoring_model") for row in target_rows} != {model["id"]}:
            raise ValueError(f"Beemo {scorer_key} rows have the wrong model")
        if {
            row.get("scoring_context_policy") for row in target_rows
        } != {"detectllm_output_only"}:
            raise ValueError(f"Beemo {scorer_key} rows are not output-only")
        target_counts[scorer_key] = len(target_rows)
    variants_by_id: dict[str, set[str]] = {}
    for row in data_rows:
        variants_by_id.setdefault(str(row["sample_id"]), set()).add(
            str(row["beemo_variant"])
        )
    expected_variants = {
        "human",
        "original",
        "expert",
        "llama_p1",
        "llama_p2",
        "llama_p3",
        "gpt_p1",
        "gpt_p2",
        "gpt_p3",
    }
    if any(variants != expected_variants for variants in variants_by_id.values()):
        raise ValueError("at least one Beemo record is missing a text variant")
    if not skip_binoculars:
        binoculars_rows = list(iter_jsonl(run_dir / "binoculars_scores.jsonl"))
        if {data_row_key(row) for row in binoculars_rows} != set(data_keys):
            raise ValueError("Beemo Binoculars keys disagree with prepared data")
        if {
            row.get("scoring_context_policy") for row in binoculars_rows
        } != {"binoculars_output_only_512"}:
            raise ValueError("Beemo Binoculars rows use the wrong context policy")
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    expected_detectors = [
        detector
        for detector in config["scoring"]["detectors"]
        if not (skip_binoculars and detector == "binoculars")
    ]
    if summary["detectors"] != expected_detectors:
        raise ValueError("Beemo result detector list is incomplete")
    with (results_dir / "metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        metrics = list(csv.DictReader(handle))
    if len(metrics) != int(summary["metrics_rows"]):
        raise ValueError("Beemo metrics.csv row count disagrees with summary")
    if {row["detector"] for row in metrics} != set(expected_detectors):
        raise ValueError("Beemo metrics.csv detector set is incomplete")
    expected_configurations = {
        (str(model["key"]), detector)
        for model in selected_models
        for detector in expected_detectors
        if detector != "binoculars"
    }
    if not skip_binoculars:
        expected_configurations.add(("binoculars", "binoculars"))
    if {
        (row["scorer_key"], row["detector"]) for row in metrics
    } != expected_configurations:
        raise ValueError(
            "Beemo metrics.csv scorer-detector configurations are incomplete"
        )
    if {float(row["target_fpr"]) for row in metrics} != {0.01, 0.05}:
        raise ValueError("Beemo metrics.csv does not contain both target FPRs")
    required = [
        run_dir / "prepare.complete.json",
        run_dir / "score.complete.json",
        results_dir / "evaluate.complete.json",
        results_dir / "plot.complete.json",
        results_dir / "metrics.csv",
        results_dir / "subgroup_metrics.csv",
        results_dir / "frozen_specs.json",
        *[
            results_dir / f"beemo_tpr_{model['key']}_2x7.png"
            for model in selected_models
        ],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Beemo validation is missing artifacts: {missing}")
    report = {
        "validation_status": "pass",
        "records": len({str(row["sample_id"]) for row in data_rows}),
        "data_rows": len(data_rows),
        "target_score_rows": target_counts,
        "scorers": [str(model["key"]) for model in selected_models],
        "detectors": expected_detectors,
        "metrics_rows": summary["metrics_rows"],
    }
    atomic_write_json(results_dir / "validation_report.json", report)
    return report
