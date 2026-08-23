"""Staged orchestration and validation for the frozen RAID benchmark."""

from __future__ import annotations

import csv
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from llm_detection.data import stable_int
from llm_detection.io import atomic_write_json, iter_jsonl
from llm_detection.runtime import accelerator_info, software_versions

from RAID.raid_data import (
    RAID_ADVERSARIAL_ATTACKS,
    RAID_ATTACKS,
    RAID_DOMAINS,
    index_raid_records,
    inspect_raid_index,
    load_raid_records,
    prepare_raid_data,
    raid_row_key,
)
from RAID.raid_evaluation import evaluate_raid, write_evaluation_artifacts
from RAID.raid_scoring import (
    RAIDBinocularsScorer,
    RAIDFalconScorer,
    score_raid_jsonl,
)


STAGES = ("prepare", "score", "evaluate", "plot", "validate")
EXPECTED_DETECTORS = (
    "log_likelihood",
    "rank",
    "log_rank",
    "lrr",
    "entropy",
    "entropy_gap",
    "binoculars",
)


def load_raid_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "protocol_name",
        "result_label",
        "dataset",
        "selection",
        "models",
        "scoring",
        "evaluation",
        "paths",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"RAID config is missing sections: {missing}")
    required_attacks = tuple(config["dataset"].get("required_attacks", ()))
    if set(required_attacks) != set(RAID_ADVERSARIAL_ATTACKS):
        raise ValueError("RAID config must contain the eleven official attacks")
    fractions = config["selection"].get("split_fractions", {})
    if set(fractions) != {"clipping_tuning", "calibration", "test"}:
        raise ValueError("RAID split fractions are incomplete")
    expected_fractions = {
        "clipping_tuning": 0.4,
        "calibration": 0.2,
        "test": 0.4,
    }
    if any(abs(float(fractions[key]) - value) > 1e-12 for key, value in expected_fractions.items()):
        raise ValueError("RAID must use the frozen 40/20/40 source split")
    if tuple(config["scoring"].get("detectors", ())) != EXPECTED_DETECTORS:
        raise ValueError("RAID must use the frozen seven-detector order")
    if config["models"]["falcon"].get("id") != "tiiuae/falcon-7b":
        raise ValueError("RAID's six single-model detectors must use Falcon-7B")
    pair = config["models"]["binoculars"]
    if pair.get("observer") != "tiiuae/falcon-7b" or pair.get("performer") != "tiiuae/falcon-7b-instruct":
        raise ValueError("RAID Binoculars must use the official Falcon pair")
    if int(config["scoring"].get("max_tokens", 0)) != 512:
        raise ValueError("RAID scoring must right-truncate at 512 input tokens")
    if config["scoring"].get("truncation_side") != "right":
        raise ValueError("RAID scoring must use right truncation")
    evaluation = config["evaluation"]
    if float(evaluation.get("target_fpr", -1)) != 0.05:
        raise ValueError("RAID reports only TPR at 5% FPR")
    if [float(value) for value in evaluation.get("clipping_quantiles", ())] != [
        0.8,
        0.85,
        0.9,
        0.95,
        0.975,
        0.99,
        0.995,
    ]:
        raise ValueError("RAID clipping quantiles disagree with the frozen protocol")
    if abs(float(evaluation.get("attack_weight", -1)) - 0.8) > 1e-12 or abs(
        float(evaluation.get("clean_weight", -1)) - 0.2
    ) > 1e-12:
        raise ValueError("RAID clipping objective must use 0.8 attack and 0.2 clean weight")
    return config


def run_paths(
    workspace: str | Path, config: dict[str, Any], run_id: str
) -> tuple[Path, Path]:
    root = Path(workspace).resolve()
    return (
        root / config["paths"]["runs"] / run_id,
        root / config["paths"]["results"] / run_id,
    )


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _input_provenance(data_path: str | Path | None) -> dict[str, Any] | None:
    if data_path is None:
        return None
    path = Path(data_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RAID data file is missing: {path}")
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def initial_manifest(
    run_id: str,
    workspace: str | Path,
    config: dict[str, Any],
    *,
    data_path: str | Path | None,
    limit_sources: int | None,
    bootstrap_repetitions: int | None,
    skip_binoculars: bool,
    debug_only: bool,
    num_shards: int = 1,
    index_cache_dir: str | Path | None = None,
    reuse_index_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    dirty = _git_value(root, "status", "--porcelain")
    return {
        "run_id": run_id,
        "protocol": config["protocol_name"],
        "result_label": config["result_label"],
        "debug_only": bool(debug_only or limit_sources is not None),
        "workspace": str(root),
        "creation_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "dirty_worktree": None if dirty is None else bool(dirty),
        "protocol_config": config,
        "data_input": _input_provenance(data_path),
        "limit_sources": limit_sources,
        "bootstrap_repetitions_override": bootstrap_repetitions,
        "binoculars_included": not skip_binoculars,
        "num_score_shards": int(num_shards),
        "index_cache_dir": (
            None if index_cache_dir is None else str(Path(index_cache_dir).resolve())
        ),
        "reuse_index_path": (
            None if reuse_index_path is None else str(Path(reuse_index_path).resolve())
        ),
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
    data_path: str | Path | None,
    limit_sources: int | None,
    bootstrap_repetitions: int | None,
    skip_binoculars: bool,
    num_shards: int | None = None,
    index_cache_dir: str | Path | None = None,
    reuse_index_path: str | Path | None = None,
) -> None:
    root = Path(manifest.get("workspace", ".")).resolve()
    dirty = _git_value(root, "status", "--porcelain")
    expected = {
        "run_id": run_id,
        "protocol": config["protocol_name"],
        "protocol_config": config,
        "data_input": (
            manifest.get("data_input")
            if data_path is None
            else _input_provenance(data_path)
        ),
        "limit_sources": limit_sources,
        "bootstrap_repetitions_override": bootstrap_repetitions,
        "binoculars_included": not skip_binoculars,
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "dirty_worktree": None if dirty is None else bool(dirty),
        "num_score_shards": (
            int(manifest.get("num_score_shards", 1))
            if num_shards is None
            else int(num_shards)
        ),
        "index_cache_dir": (
            manifest.get("index_cache_dir")
            if index_cache_dir is None
            else str(Path(index_cache_dir).resolve())
        ),
        "reuse_index_path": (
            manifest.get("reuse_index_path")
            if reuse_index_path is None
            else str(Path(reuse_index_path).resolve())
        ),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"existing RAID manifest disagrees on {key}; choose a new run ID"
            )


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def prepare_stage(
    run_dir: Path,
    config: dict[str, Any],
    *,
    data_path: str | Path | None,
    limit_sources: int | None,
    reset_index: bool,
    data_provenance: dict[str, Any] | None = None,
    num_shards: int = 1,
    index_cache_dir: str | Path | None = None,
    reuse_index_path: str | Path | None = None,
) -> dict[str, Any]:
    if data_path is not None:
        source: Any = Path(data_path).resolve()
    else:
        source = load_raid_records(config)
    provenance = data_provenance or _input_provenance(data_path)
    if provenance is None:
        index_path = run_dir / "raid_index.sqlite3"
        reuse_index = False
        cache_summary: dict[str, Any] = {"mode": "run_local"}
    elif reuse_index_path is not None:
        if limit_sources is None:
            raise ValueError(
                "explicit run-index reuse is debug-only; full runs require a completed checksum cache"
            )
        index_path = Path(reuse_index_path).resolve()
        source_manifest_path = index_path.parent / "manifest.json"
        if not source_manifest_path.is_file():
            raise ValueError(
                "an explicitly reused RAID index must sit beside its source manifest.json"
            )
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest.get("data_input") != provenance:
            raise ValueError("reused RAID index source manifest has different input data")
        inspection = inspect_raid_index(index_path)
        reuse_index = True
        cache_summary = {
            "mode": "explicit_reuse",
            "index_path": str(index_path),
            "source_run_id": source_manifest.get("run_id"),
            **inspection,
        }
    elif index_cache_dir is not None:
        cache_root = Path(index_cache_dir).resolve() / str(provenance["sha256"])
        cache_root.mkdir(parents=True, exist_ok=True)
        index_path = cache_root / "raid_index.sqlite3"
        marker_path = cache_root / "index.complete.json"
        marker = (
            json.loads(marker_path.read_text(encoding="utf-8"))
            if marker_path.is_file()
            else None
        )
        expected_marker = {
            "index_schema": "raid-sqlite-index-v2",
            "data_size_bytes": int(provenance["size_bytes"]),
            "data_sha256": str(provenance["sha256"]),
        }
        if marker is not None:
            if any(marker.get(key) != value for key, value in expected_marker.items()):
                raise ValueError("RAID cache marker disagrees with the input data")
            inspection = inspect_raid_index(index_path)
            if any(int(marker.get(key, -1)) != value for key, value in inspection.items()):
                raise ValueError("RAID cache marker row counts disagree with its index")
            reuse_index = True
            cache_summary = {"mode": "cache_hit", "index_path": str(index_path), **inspection}
        else:
            build = index_raid_records(source, index_path, reset=True)
            inspection = inspect_raid_index(index_path)
            marker = {**expected_marker, **inspection}
            atomic_write_json(marker_path, marker)
            reuse_index = True
            cache_summary = {
                "mode": "cache_built",
                "index_path": str(index_path),
                **build,
                **inspection,
            }
    else:
        index_path = run_dir / "raid_index.sqlite3"
        reuse_index = False
        cache_summary = {"mode": "run_local"}

    summary = prepare_raid_data(
        source,
        run_dir / "data.jsonl",
        run_dir / "selected_sources.jsonl",
        run_dir / "excluded_sources.json",
        index_path,
        selection_seed=int(config["selection"]["selection_seed"]),
        split_seed=int(config["selection"]["split_seed"]),
        split_fractions=config["selection"]["split_fractions"],
        limit_sources=limit_sources,
        dataset_revision=config["dataset"].get("revision"),
        dataset_fingerprint=config["dataset"].get("fingerprint"),
        reset_index=reset_index,
        reuse_index=reuse_index,
    )
    summary["index_cache"] = cache_summary
    summary["expected_paper_sources"] = config["dataset"].get(
        "expected_paper_sources"
    )
    summary["sample_scope"] = (
        "bounded_debug"
        if limit_sources is not None
        else "complete_available_labeled_release"
    )
    atomic_write_json(run_dir / "prepare.complete.json", summary)
    shard_summary = write_prepared_shards(run_dir, int(num_shards))
    summary["score_shards"] = shard_summary
    atomic_write_json(run_dir / "prepare.complete.json", summary)
    return {
        "data": str(run_dir / "data.jsonl"),
        "selected_sources": str(run_dir / "selected_sources.jsonl"),
        "prepare_summary": summary,
    }


def score_shard_index(source_id: str, num_shards: int) -> int:
    if int(num_shards) < 1:
        raise ValueError("num_shards must be positive")
    return stable_int("raid-score-shard-v1", str(source_id)) % int(num_shards)


def shard_data_path(run_dir: Path, shard_index: int, num_shards: int) -> Path:
    return run_dir / "data_shards" / f"part-{int(shard_index):05d}-of-{int(num_shards):05d}.jsonl"


def write_prepared_shards(run_dir: Path, num_shards: int) -> dict[str, Any]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    data_path = run_dir / "data.jsonl"
    shard_dir = run_dir / "data_shards"
    temporary = run_dir / "data_shards.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    handles = [
        (temporary / path.name).open("w", encoding="utf-8", newline="")
        for path in (shard_data_path(run_dir, index, num_shards) for index in range(num_shards))
    ]
    counts = [0] * num_shards
    sources: list[set[str]] = [set() for _ in range(num_shards)]
    try:
        for row in iter_jsonl(data_path):
            source_id = str(row["source_id"])
            index = score_shard_index(source_id, num_shards)
            handles[index].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts[index] += 1
            sources[index].add(source_id)
    finally:
        for handle in handles:
            handle.close()
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    os.replace(temporary, shard_dir)
    if any(count == 0 for count in counts):
        raise ValueError("RAID score sharding produced an empty shard")
    if any(count != len(source_ids) * (1 + len(RAID_ATTACKS)) for count, source_ids in zip(counts, sources)):
        raise ValueError("RAID score sharding split a source family")
    marker = {
        "assignment": "stable_int('raid-score-shard-v1', source_id) % num_shards",
        "num_shards": num_shards,
        "rows_per_shard": counts,
        "sources_per_shard": [len(values) for values in sources],
    }
    atomic_write_json(run_dir / "data_shards.complete.json", marker)
    return marker


def _scoring_adjustments(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    boundary = 0
    truncated = 0
    removed = 0
    maximum_original = 0
    for row in rows:
        boundary += int(bool(row.get("boundary_token_added")))
        count = int(row.get("truncated_token_count", 0))
        truncated += int(count > 0)
        removed += count
        maximum_original = max(maximum_original, int(row["original_num_input_tokens"]))
    return {
        "boundary_token_rows": boundary,
        "truncated_rows": truncated,
        "truncated_tokens": removed,
        "maximum_original_input_tokens": maximum_original,
    }


def score_shard_output_path(
    run_dir: Path, scorer_name: str, shard_index: int, num_shards: int
) -> Path:
    return (
        run_dir
        / "score_shards"
        / f"{scorer_name}-part-{int(shard_index):05d}-of-{int(num_shards):05d}.jsonl"
    )


def _falcon_scorer(config: dict[str, Any]) -> RAIDFalconScorer:
    scoring = config["scoring"]
    model = config["models"]["falcon"]
    return RAIDFalconScorer(
        {
            **scoring,
            "model_id": model["id"],
            "revision": model.get("revision"),
            "tokenizer_revision": model.get("tokenizer_revision"),
            "context_policy": "raid_output_only_512",
        }
    )


def _binoculars_scorer(config: dict[str, Any]) -> RAIDBinocularsScorer:
    return RAIDBinocularsScorer(
        {
            **config["scoring"],
            **config["models"]["binoculars"],
            "context_policy": "binoculars_official_output_only_512",
        }
    )


def score_shard_stage(
    run_dir: Path,
    config: dict[str, Any],
    *,
    scorer_name: str,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    """Score one immutable prepared source shard without touching the manifest."""
    if scorer_name not in {"falcon", "binoculars"}:
        raise ValueError("RAID shard scorer must be falcon or binoculars")
    if not 0 <= int(shard_index) < int(num_shards):
        raise ValueError("RAID shard index is outside num_shards")
    marker = json.loads(
        (run_dir / "data_shards.complete.json").read_text(encoding="utf-8")
    )
    if int(marker.get("num_shards", -1)) != int(num_shards):
        raise ValueError("prepared RAID shard count disagrees with the scorer job")
    input_path = shard_data_path(run_dir, shard_index, num_shards)
    if not input_path.is_file():
        raise FileNotFoundError(f"prepared RAID score shard is missing: {input_path}")
    output_path = score_shard_output_path(
        run_dir, scorer_name, shard_index, num_shards
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scorer = _falcon_scorer(config) if scorer_name == "falcon" else _binoculars_scorer(config)
    count = score_raid_jsonl(input_path, output_path, scorer, config["scoring"])
    resolved = (
        {
            "model": scorer.model_id,
            "model_revision": scorer.resolved_revision,
            "tokenizer_revision": scorer.resolved_tokenizer_revision,
            "dtype": scorer.resolved_dtype,
            "device": scorer.resolved_device,
        }
        if scorer_name == "falcon"
        else {
            "observer": scorer.observer_id,
            "observer_revision": scorer.observer_resolved_revision,
            "observer_tokenizer_revision": scorer.resolved_tokenizer_revision,
            "performer": scorer.performer_id,
            "performer_revision": scorer.performer_resolved_revision,
            "performer_tokenizer_revision": scorer.performer_resolved_tokenizer_revision,
            "observer_dtype": scorer.observer_dtype,
            "performer_dtype": scorer.performer_dtype,
        }
    )
    del scorer
    _release_cuda()
    expected = int(marker["rows_per_shard"][int(shard_index)])
    if count != expected:
        raise ValueError(
            f"RAID {scorer_name} shard produced {count} rows; expected {expected}"
        )
    complete = {
        "scorer": scorer_name,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "score_rows": count,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "resolved": resolved,
        "adjustments": _scoring_adjustments(iter_jsonl(output_path)),
    }
    atomic_write_json(output_path.with_suffix(".complete.json"), complete)
    return complete


def _merge_one_scorer(
    run_dir: Path, scorer_name: str, num_shards: int
) -> tuple[Path, int, list[dict[str, Any]]]:
    expected_keys = {raid_row_key(row) for row in iter_jsonl(run_dir / "data.jsonl")}
    seen: set[tuple[str, str, str]] = set()
    final_path = run_dir / f"{scorer_name}_scores.jsonl"
    temporary = final_path.with_suffix(".jsonl.tmp")
    markers: list[dict[str, Any]] = []
    schema = (
        "raid-falcon-token-features-v1"
        if scorer_name == "falcon"
        else "raid-binoculars-token-features-v1"
    )
    context = (
        "raid_output_only_512"
        if scorer_name == "falcon"
        else "binoculars_official_output_only_512"
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for index in range(num_shards):
            shard = score_shard_output_path(run_dir, scorer_name, index, num_shards)
            marker_path = shard.with_suffix(".complete.json")
            if not shard.is_file() or not marker_path.is_file():
                raise FileNotFoundError(
                    f"RAID {scorer_name} shard {index}/{num_shards} is incomplete"
                )
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                marker.get("scorer") != scorer_name
                or int(marker.get("shard_index", -1)) != index
                or int(marker.get("num_shards", -1)) != num_shards
            ):
                raise ValueError(f"RAID {scorer_name} shard marker is inconsistent")
            rows = 0
            for row in iter_jsonl(shard):
                key = raid_row_key(row)
                if key not in expected_keys or key in seen:
                    raise ValueError(
                        f"RAID {scorer_name} shard has an unexpected or duplicate key {key!r}"
                    )
                if row.get("scoring_feature_schema") != schema:
                    raise ValueError(f"RAID {scorer_name} shard schema is inconsistent")
                if row.get("scoring_context_policy") != context or int(
                    row.get("scoring_max_tokens", -1)
                ) != 512:
                    raise ValueError(f"RAID {scorer_name} shard context is inconsistent")
                seen.add(key)
                rows += 1
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if rows != int(marker.get("score_rows", -1)):
                raise ValueError(f"RAID {scorer_name} shard marker row count is wrong")
            markers.append(marker)
    if seen != expected_keys:
        raise ValueError(
            f"RAID {scorer_name} merged keys are incomplete: {len(seen)}/{len(expected_keys)}"
        )
    os.replace(temporary, final_path)
    return final_path, len(seen), markers


def merge_score_shards(run_dir: Path, num_shards: int) -> dict[str, Any]:
    falcon_path, falcon_count, falcon_markers = _merge_one_scorer(
        run_dir, "falcon", num_shards
    )
    binoculars_path, binoculars_count, binoculars_markers = _merge_one_scorer(
        run_dir, "binoculars", num_shards
    )
    if falcon_count != binoculars_count:
        raise ValueError("merged RAID scorer packs have different row counts")
    for scorer_name, markers in (
        ("falcon", falcon_markers),
        ("binoculars", binoculars_markers),
    ):
        resolved = {json.dumps(marker.get("resolved"), sort_keys=True) for marker in markers}
        if len(resolved) != 1:
            raise ValueError(f"RAID {scorer_name} shards used different model provenance")
    marker = {
        "data_rows": falcon_count,
        "falcon_score_rows": falcon_count,
        "binoculars_score_rows": binoculars_count,
        "num_score_shards": int(num_shards),
        "falcon_shards": falcon_markers,
        "binoculars_shards": binoculars_markers,
        "falcon_adjustments": _scoring_adjustments(iter_jsonl(falcon_path)),
        "binoculars_adjustments": _scoring_adjustments(iter_jsonl(binoculars_path)),
    }
    atomic_write_json(run_dir / "score.complete.json", marker)
    return {
        "falcon_scores": str(falcon_path),
        "binoculars_scores": str(binoculars_path),
        "score_summary": marker,
    }


def score_stage(
    run_dir: Path,
    config: dict[str, Any],
    *,
    skip_binoculars: bool,
) -> dict[str, Any]:
    data_path = run_dir / "data.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"prepared RAID data is missing: {data_path}")
    shard_marker = run_dir / "data_shards.complete.json"
    if shard_marker.is_file():
        num_shards = int(
            json.loads(shard_marker.read_text(encoding="utf-8"))["num_shards"]
        )
        if num_shards > 1:
            if skip_binoculars:
                raise ValueError("sharded RAID merge requires Binoculars")
            return merge_score_shards(run_dir, num_shards)
    scoring = config["scoring"]
    falcon_model = config["models"]["falcon"]
    falcon_config = {
        **scoring,
        "model_id": falcon_model["id"],
        "revision": falcon_model.get("revision"),
        "tokenizer_revision": falcon_model.get("tokenizer_revision"),
        "context_policy": "raid_output_only_512",
    }
    falcon_path = run_dir / "falcon_scores.jsonl"
    falcon = RAIDFalconScorer(falcon_config)
    falcon_count = score_raid_jsonl(data_path, falcon_path, falcon, scoring)
    falcon_resolved = {
        "model": falcon.model_id,
        "model_revision": falcon.resolved_revision,
        "tokenizer_revision": falcon.resolved_tokenizer_revision,
        "dtype": falcon.resolved_dtype,
        "device": falcon.resolved_device,
        "context_policy": falcon.context_policy,
        "max_tokens": falcon.max_tokens,
    }
    del falcon
    _release_cuda()

    binoculars_path = run_dir / "binoculars_scores.jsonl"
    binoculars_count: int | None = None
    binoculars_resolved: dict[str, Any] | None = None
    if not skip_binoculars:
        pair = config["models"]["binoculars"]
        binoculars_config = {
            **scoring,
            **pair,
            "context_policy": "binoculars_official_output_only_512",
        }
        binoculars = RAIDBinocularsScorer(binoculars_config)
        binoculars_count = score_raid_jsonl(
            data_path, binoculars_path, binoculars, scoring
        )
        binoculars_resolved = {
            "observer": binoculars.observer_id,
            "observer_revision": binoculars.observer_resolved_revision,
            "observer_tokenizer_revision": binoculars.resolved_tokenizer_revision,
            "observer_dtype": binoculars.observer_dtype,
            "observer_device": binoculars.resolved_observer_device,
            "performer": binoculars.performer_id,
            "performer_revision": binoculars.performer_resolved_revision,
            "performer_tokenizer_revision": binoculars.performer_resolved_tokenizer_revision,
            "performer_dtype": binoculars.performer_dtype,
            "performer_device": binoculars.resolved_performer_device,
            "context_policy": binoculars.context_policy,
            "max_tokens": binoculars.max_tokens,
        }
        del binoculars
        _release_cuda()

    data_count = sum(1 for _ in iter_jsonl(data_path))
    if falcon_count != data_count or (
        binoculars_count is not None and binoculars_count != data_count
    ):
        raise ValueError("RAID score counts do not match prepared data")
    marker = {
        "data_rows": data_count,
        "falcon_score_rows": falcon_count,
        "binoculars_score_rows": binoculars_count,
        "falcon_adjustments": _scoring_adjustments(iter_jsonl(falcon_path)),
        "binoculars_adjustments": (
            _scoring_adjustments(iter_jsonl(binoculars_path))
            if binoculars_count is not None
            else None
        ),
        "falcon_resolved": falcon_resolved,
        "binoculars_resolved": binoculars_resolved,
    }
    atomic_write_json(run_dir / "score.complete.json", marker)
    return {
        "falcon_scores": str(falcon_path),
        "binoculars_scores": None if skip_binoculars else str(binoculars_path),
        "score_summary": marker,
    }


def evaluate_stage(
    run_dir: Path,
    results_dir: Path,
    config: dict[str, Any],
    *,
    bootstrap_repetitions: int | None,
    skip_binoculars: bool,
) -> dict[str, Any]:
    if skip_binoculars:
        raise ValueError(
            "RAID's frozen seven-detector evaluation requires Binoculars; "
            "--skip-binoculars is score-stage debugging only"
        )
    falcon_path = run_dir / "falcon_scores.jsonl"
    binoculars_path = run_dir / "binoculars_scores.jsonl"
    if not falcon_path.is_file() or not binoculars_path.is_file():
        raise FileNotFoundError("RAID Falcon and Binoculars score packs are required")
    evaluation_config = {
        **config,
        "expected_attacks": list(config["dataset"]["required_attacks"]),
        "evaluation": {
            **config["evaluation"],
            "bootstrap_repetitions": (
                int(bootstrap_repetitions)
                if bootstrap_repetitions is not None
                else int(config["evaluation"]["bootstrap_repetitions"])
            ),
        },
    }
    result = evaluate_raid(
        list(iter_jsonl(falcon_path)),
        evaluation_config,
        list(iter_jsonl(binoculars_path)),
        config["evaluation"].get("published_binoculars_tpr"),
    )
    paths = write_evaluation_artifacts(result, results_dir)
    marker = {
        **result.validation_counts,
        "contamination_cutpoints": result.contamination_cutpoints,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    atomic_write_json(results_dir / "evaluate.complete.json", marker)
    return {
        **{name: str(path) for name, path in paths.items()},
        "evaluation_summary": marker,
    }


def plot_stage(results_dir: Path) -> dict[str, Any]:
    from RAID.raid_plot import plot_raid

    paths = plot_raid(results_dir)
    marker = {"plots": [str(path) for path in paths]}
    atomic_write_json(results_dir / "plot.complete.json", marker)
    return marker


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_prepared_rows(
    rows: list[dict[str, Any]], required_attacks: set[str]
) -> dict[str, Any]:
    keys = [raid_row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("prepared RAID data contains duplicate stable keys")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_id"])].append(row)
    split_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    for source_id, family in grouped.items():
        splits = {str(row["split"]) for row in family}
        if len(splits) != 1:
            raise ValueError(f"RAID source leaks across splits: {source_id}")
        split_counts.update(splits)
        domain_counts.update({str(family[0]["domain"])})
        human_rows = [row for row in family if row["label"] == "human"]
        machine_rows = [row for row in family if row["label"] == "llm"]
        if len(human_rows) != 1 or len(machine_rows) != len(RAID_ATTACKS):
            raise ValueError(f"RAID source has an incomplete family: {source_id}")
        conditions = {str(row["attack"]) for row in machine_rows}
        if conditions != {"none", *required_attacks}:
            raise ValueError(f"RAID source has the wrong attacks: {source_id}")
        generation_ids = {str(row["base_generation_id"]) for row in machine_rows}
        if len(generation_ids) != 1:
            raise ValueError(f"RAID source has mixed machine families: {source_id}")
    if set(split_counts) != {"clipping_tuning", "calibration", "test"}:
        raise ValueError("prepared RAID data does not contain all frozen splits")
    return {
        "sources": len(grouped),
        "data_rows": len(rows),
        "split_sources": dict(split_counts),
        "domain_sources": dict(domain_counts),
    }


def validate_artifacts(
    run_dir: Path,
    results_dir: Path,
    config: dict[str, Any],
    *,
    limit_sources: int | None,
    bootstrap_repetitions: int | None,
    skip_binoculars: bool,
) -> dict[str, Any]:
    if skip_binoculars:
        raise ValueError("a scientifically complete RAID run requires Binoculars")
    required_files = [
        run_dir / "selected_sources.jsonl",
        run_dir / "data.jsonl",
        run_dir / "falcon_scores.jsonl",
        run_dir / "binoculars_scores.jsonl",
        run_dir / "prepare.complete.json",
        run_dir / "score.complete.json",
        results_dir / "frozen_specs.json",
        results_dir / "metrics.csv",
        results_dir / "attack_summary.csv",
        results_dir / "contamination_summary.csv",
        results_dir / "binoculars_sanity.csv",
        results_dir / "calibration_summary.csv",
        results_dir / "contamination_records.csv",
        results_dir / "evaluation_counts.json",
        results_dir / "evaluate.complete.json",
        results_dir / "plot.complete.json",
        results_dir / "plots" / "raid_attack_tpr.png",
        results_dir / "plots" / "raid_contamination_tpr.png",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"RAID validation is missing artifacts: {missing}")
    data_rows = list(iter_jsonl(run_dir / "data.jsonl"))
    required_attacks = set(config["dataset"]["required_attacks"])
    prepared = _validate_prepared_rows(data_rows, required_attacks)
    if set(prepared["domain_sources"]) != set(RAID_DOMAINS):
        raise ValueError("RAID prepared data does not cover all eight core domains")
    recognized_counts = {
        int(value)
        for value in config["dataset"].get("recognized_full_source_counts", ())
    }
    if limit_sources is None and recognized_counts and prepared["sources"] not in recognized_counts:
        raise ValueError(
            "RAID full source count is not a recognized labeled release or paper snapshot"
        )
    data_keys = {raid_row_key(row) for row in data_rows}
    score_counts: dict[str, int] = {}
    for name, path, schema, context in (
        (
            "falcon",
            run_dir / "falcon_scores.jsonl",
            "raid-falcon-token-features-v1",
            "raid_output_only_512",
        ),
        (
            "binoculars",
            run_dir / "binoculars_scores.jsonl",
            "raid-binoculars-token-features-v1",
            "binoculars_official_output_only_512",
        ),
    ):
        rows = list(iter_jsonl(path))
        keys = {raid_row_key(row) for row in rows}
        if len(rows) != len(data_rows) or keys != data_keys:
            raise ValueError(f"RAID {name} score keys disagree with prepared data")
        if {row.get("scoring_feature_schema") for row in rows} != {schema}:
            raise ValueError(f"RAID {name} score schema is not frozen")
        if {row.get("scoring_context_policy") for row in rows} != {context}:
            raise ValueError(f"RAID {name} scoring context is not frozen")
        if {int(row.get("scoring_max_tokens", -1)) for row in rows} != {512}:
            raise ValueError(f"RAID {name} scoring limit is not 512")
        score_counts[name] = len(rows)

    score_marker = json.loads(
        (run_dir / "score.complete.json").read_text(encoding="utf-8")
    )
    num_score_shards = int(score_marker.get("num_score_shards", 1))
    if num_score_shards > 1:
        prepared_shards = json.loads(
            (run_dir / "data_shards.complete.json").read_text(encoding="utf-8")
        )
        if int(prepared_shards.get("num_shards", -1)) != num_score_shards:
            raise ValueError("RAID prepared and scored shard counts disagree")
        for scorer_name in ("falcon", "binoculars"):
            shard_markers = score_marker.get(f"{scorer_name}_shards", [])
            if len(shard_markers) != num_score_shards or {
                int(marker.get("shard_index", -1)) for marker in shard_markers
            } != set(range(num_score_shards)):
                raise ValueError(
                    f"RAID {scorer_name} completion does not cover every score shard"
                )

    evaluation_counts = json.loads(
        (results_dir / "evaluation_counts.json").read_text(encoding="utf-8")
    )
    repetitions = (
        int(bootstrap_repetitions)
        if bootstrap_repetitions is not None
        else int(config["evaluation"]["bootstrap_repetitions"])
    )
    if int(evaluation_counts.get("bootstrap_repetitions", -1)) != repetitions:
        raise ValueError("RAID evaluation used the wrong bootstrap repetitions")
    if evaluation_counts.get("detectors") != list(EXPECTED_DETECTORS):
        raise ValueError("RAID evaluation detector list is incomplete")
    if evaluation_counts.get("target_fprs") != [0.05]:
        raise ValueError("RAID evaluation must contain only the 5% FPR target")
    if int(evaluation_counts.get("universal_specs", -1)) != 7 or int(
        evaluation_counts.get("rate_adaptive_specs", -1)
    ) != 0 or int(evaluation_counts.get("attack_specific_specs", -1)) != 0:
        raise ValueError("RAID evaluation did not use one universal spec per detector")
    contamination_records = _csv_rows(results_dir / "contamination_records.csv")
    expected_machine_rows = prepared["sources"] * len(RAID_ATTACKS)
    if len(contamination_records) != expected_machine_rows:
        raise ValueError("RAID contamination record count is incomplete")
    contamination_required = {
        "levenshtein_distance",
        "substitution_count",
        "deletion_count",
        "insertion_count",
        "realized_contamination_rate",
        "native_attack_rate",
        "full_text_edit_audited",
        "truncation_hid_edits",
    }
    if any(
        not contamination_required.issubset(row)
        or row["full_text_edit_audited"].lower() != "true"
        for row in contamination_records
    ):
        raise ValueError("RAID contamination records lack the frozen edit audit")

    attack_rows = _csv_rows(results_dir / "attack_summary.csv")
    if len(attack_rows) != 7 * len(RAID_ATTACKS):
        raise ValueError("RAID attack summary has the wrong number of rows")
    if {row["detector"] for row in attack_rows} != set(EXPECTED_DETECTORS):
        raise ValueError("RAID attack summary is missing detectors")
    if {row["condition"] for row in attack_rows} != set(RAID_ATTACKS):
        raise ValueError("RAID attack summary is missing conditions")
    if {float(row["target_fpr"]) for row in attack_rows} != {0.05}:
        raise ValueError("RAID attack summary contains a non-5% FPR row")
    ci_columns = {
        "raw_tpr_ci_low",
        "raw_tpr_ci_high",
        "clipped_tpr_ci_low",
        "clipped_tpr_ci_high",
        "paired_tpr_difference_ci_low",
        "paired_tpr_difference_ci_high",
    }
    if repetitions > 0 and any(
        not ci_columns.issubset(row) or any(row[name] == "" for name in ci_columns)
        for row in attack_rows
    ):
        raise ValueError("RAID attack summary lacks bootstrap confidence intervals")

    metrics = _csv_rows(results_dir / "metrics.csv")
    if {row["detector"] for row in metrics} != set(EXPECTED_DETECTORS):
        raise ValueError("RAID metrics are missing detectors")
    if {row["aggregation"] for row in metrics} != {"raw", "clipped"}:
        raise ValueError("RAID metrics must contain paired raw and clipped rows")
    if {float(row["target_fpr"]) for row in metrics} != {0.05}:
        raise ValueError("RAID metrics contain a non-5% FPR row")
    sanity = _csv_rows(results_dir / "binoculars_sanity.csv")
    published_conditions = set(config["evaluation"]["published_binoculars_tpr"])
    if {row["condition"] for row in sanity} != published_conditions:
        raise ValueError("RAID Binoculars sanity table is incomplete")

    frozen = json.loads(
        (results_dir / "frozen_specs.json").read_text(encoding="utf-8")
    )
    if frozen.get("rate_adaptive_clipping") is not False or frozen.get(
        "attack_specific_clipping"
    ) is not False:
        raise ValueError("RAID frozen specs unexpectedly contain adaptive clipping")
    if set(frozen.get("detectors", {})) != set(EXPECTED_DETECTORS):
        raise ValueError("RAID frozen specs are missing detectors")

    report = {
        "validation_status": "pass",
        "debug_only": limit_sources is not None,
        **prepared,
        "score_rows": score_counts,
        "num_score_shards": num_score_shards,
        "detectors": list(EXPECTED_DETECTORS),
        "target_fpr": 0.05,
        "bootstrap_repetitions": repetitions,
        "metrics_rows": len(metrics),
        "attack_summary_rows": len(attack_rows),
        "contamination_record_rows": len(contamination_records),
        "binoculars_sanity_rows": len(sanity),
        "contamination_cutpoints": frozen.get("contamination_cutpoints", []),
    }
    atomic_write_json(results_dir / "validation_report.json", report)
    return report
