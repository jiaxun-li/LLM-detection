"""End-to-end experiment stages used by local smoke and Delta jobs."""

from __future__ import annotations

import heapq
import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .config import resolved_run_config, total_examples
from .data import (
    TailCandidate,
    assign_splits_and_generation_seeds,
    contamination_counts,
    corruption_seed,
    data_row_key,
    expected_rows_per_source,
    random_token_contamination,
    rank_tail_candidates_once,
    sentence_spans,
    stable_int,
    tail_token_contamination,
    validate_disjoint_splits,
)
from .evaluation import evaluate
from .generation import (
    GenerationRequest,
    TransformersBackend,
    make_generation_backend,
)
from .io import (
    AppendSafeJsonlWriter,
    append_jsonl,
    atomic_write_text,
    completed_keys,
    iter_jsonl,
    validate_unique_jsonl,
    write_completion_marker,
)
from .runtime import Throughput, peak_gpu_memory_bytes
from .scoring import BinocularsScorer, TargetModelScorer, score_jsonl


def source_manifest_path(config: dict[str, Any], dataset: str, workspace: Path) -> Path:
    count = total_examples(config)
    seed = config["splits"]["selection_seed"]
    spec = config["datasets"][dataset]
    source_signature = stable_int(
        json.dumps(spec, sort_keys=True),
        json.dumps(config["splits"], sort_keys=True),
        json.dumps(config["generation"]["seeds"]),
        bits=31,
    )
    return (
        workspace
        / config["paths"]["source_manifests"]
        / f"{dataset}-{spec['split']}-{count}-seed{seed}-{source_signature:08x}.jsonl"
    )


def _source_text(row: dict[str, Any], fields: Sequence[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str):
            normalized = re.sub(r"\s+", " ", value).strip()
            if len(normalized.split()) >= 20:
                return normalized
    return None


def _source_id(row: dict[str, Any], fields: Sequence[str], text: str) -> str:
    # Hashing normalized source text deduplicates repeated SQuAD contexts.
    text_hash = f"sha256:{stable_int(text, bits=63):016x}"
    for field in fields:
        value = row.get(field)
        if value is not None:
            return f"{field}:{value}:{text_hash}"
    return text_hash


def select_source_manifest(
    config: dict[str, Any],
    dataset_name: str,
    path: str | Path,
) -> Path:
    """Create the model-independent source manifest exactly once."""
    target = Path(path)
    if target.exists():
        rows = list(iter_jsonl(target))
        validate_disjoint_splits(rows, config["splits"])
        return target
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "source selection requires datasets; install requirements.txt on the run host"
        ) from exc

    spec = config["datasets"][dataset_name]
    resolved_revision = HfApi().dataset_info(
        spec["id"], revision=spec.get("revision")
    ).sha
    kwargs: dict[str, Any] = {
        "path": spec["id"],
        "split": spec["split"],
        "revision": spec.get("revision"),
        "streaming": True,
    }
    if spec.get("config") is not None:
        kwargs["name"] = spec["config"]
    dataset = load_dataset(**kwargs)
    needed = total_examples(config)
    # Keep only the smallest deterministic hashes, bounding source-selection memory.
    heap: list[tuple[int, str, dict[str, Any]]] = []
    seen_text_hashes: set[int] = set()
    packed_records: list[dict[str, Any]] = []
    packed_word_count = 0
    minimum_words = int(spec.get("minimum_source_words", 260))

    def consider(record: dict[str, Any]) -> None:
        source_id = record["source_id"]
        priority = stable_int(
            config["splits"]["selection_seed"], dataset_name, source_id
        )
        item = (-priority, source_id, record)
        if len(heap) < needed:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    for index, raw in enumerate(dataset):
        text = _source_text(raw, spec["text_fields"])
        if text is None:
            continue
        text_hash = stable_int(text)
        if text_hash in seen_text_hashes:
            continue
        seen_text_hashes.add(text_hash)
        source_id = _source_id(raw, spec["id_fields"], text)
        record = {
            "dataset": dataset_name,
            "dataset_id": spec["id"],
            "dataset_config": spec.get("config"),
            "dataset_split": spec["split"],
            "dataset_revision": spec.get("revision"),
            "dataset_resolved_revision": resolved_revision,
            "source_id": source_id,
            "source_index": index,
            "source_text": text,
        }
        word_count = len(text.split())
        if word_count >= minimum_words:
            consider(record)
        elif spec.get("pack_short_documents", False):
            packed_records.append(record)
            packed_word_count += word_count
            if packed_word_count >= minimum_words:
                component_ids = [item["source_id"] for item in packed_records]
                combined_text = "\n\n".join(
                    item["source_text"] for item in packed_records
                )
                consider(
                    {
                        "dataset": dataset_name,
                        "dataset_id": spec["id"],
                        "dataset_config": spec.get("config"),
                        "dataset_split": spec["split"],
                        "dataset_revision": spec.get("revision"),
                        "dataset_resolved_revision": resolved_revision,
                        "source_id": "pack:"
                        + f"{stable_int(*component_ids, bits=63):016x}",
                        "source_index": [
                            item["source_index"] for item in packed_records
                        ],
                        "source_component_ids": component_ids,
                        "source_text": combined_text,
                    }
                )
                packed_records = []
                packed_word_count = 0
    candidates = [item[2] for item in heap]
    selected = assign_splits_and_generation_seeds(
        candidates,
        config["splits"],
        config["generation"]["seeds"],
        config["splits"]["selection_seed"],
    )
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected
    )
    atomic_write_text(target, payload)
    return target


def _base_key(row: dict[str, Any]) -> tuple[str]:
    return (str(row["sample_id"]),)


def _tail_key(row: dict[str, Any]) -> tuple[str]:
    return (str(row["sample_id"]),)


def _decode_visible(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _stable_visible_tokenization(
    tokenizer: Any, token_ids: Sequence[int], max_iterations: int = 12
) -> tuple[str, list[int]]:
    """Return visible text and token IDs that reproduce one another exactly."""
    current = list(token_ids)
    for _ in range(max_iterations):
        text = _decode_visible(tokenizer, current)
        encoded = list(tokenizer.encode(text, add_special_tokens=False))
        if encoded == current:
            return text, current
        current = encoded
    raise ValueError("visible decode/re-tokenize sequence did not stabilize")


def _canonical_length_matched_pair(
    tokenizer: Any,
    human_ids: Sequence[int],
    llm_ids: Sequence[int],
    maximum_length: int,
) -> tuple[str, list[int], str, list[int]]:
    """Canonicalize a pathological pair and retain a common visible length."""
    _, canonical_human = _stable_visible_tokenization(tokenizer, human_ids)
    _, canonical_llm = _stable_visible_tokenization(tokenizer, llm_ids)
    for _ in range(12):
        length = min(len(canonical_human), len(canonical_llm), maximum_length)
        if length <= 0:
            raise ValueError("canonical continuation is empty")
        human_text, next_human = _stable_visible_tokenization(
            tokenizer, canonical_human[:length]
        )
        llm_text, next_llm = _stable_visible_tokenization(
            tokenizer, canonical_llm[:length]
        )
        if len(next_human) == len(next_llm) == length:
            return human_text, next_human, llm_text, next_llm
        canonical_human = next_human
        canonical_llm = next_llm
    raise ValueError("canonical human/LLM continuation lengths did not converge")


def generate_base_examples(
    run_config: dict[str, Any],
    source_path: str | Path,
    base_path: str | Path,
    backend: Any,
) -> int:
    """Generate and checkpoint uncontaminated continuations once."""
    completed = completed_keys(base_path, _base_key)
    sources = [row for row in iter_jsonl(source_path) if _base_key(row) not in completed]
    progress = Throughput(len(sources) + len(completed), "generation")
    progress.examples = len(completed)
    tokenizer = backend.tokenizer
    generation = run_config["generation"]
    requests: list[GenerationRequest] = []
    prepared: dict[
        str,
        tuple[dict[str, Any], list[int], list[int], str, list[int], int],
    ] = {}
    for source in sources:
        token_ids = tokenizer.encode(source["source_text"], add_special_tokens=False)
        required = int(generation["prompt_tokens"]) + int(
            generation["continuation_tokens"]
        )
        if len(token_ids) < required:
            raise ValueError(
                f"model-independent source {source['sample_id']} has {len(token_ids)} "
                f"target tokens, fewer than required {required}; rebuild the source "
                "manifest with a stricter target-independent word filter"
            )
        prompt_ids = token_ids[: int(generation["prompt_tokens"])]
        human_ids = token_ids[int(generation["prompt_tokens"]) : required]
        prompt = tokenizer.decode(
            prompt_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_delta = len(encoded_prompt) - int(generation["prompt_tokens"])
        tolerance = int(
            run_config["contamination"].get("max_length_delta_tokens", 4)
        )
        if abs(prompt_delta) > tolerance:
            raise ValueError(
                f"prompt round-trip for {source['sample_id']} produced "
                f"{len(encoded_prompt)} tokens instead of "
                f"{generation['prompt_tokens']}; drift exceeds configured "
                f"tolerance {tolerance}"
            )
        key = str(source["sample_id"])
        requests.append(GenerationRequest(key, prompt, int(source["generation_seed"])))
        prepared[key] = (
            source,
            prompt_ids,
            human_ids,
            prompt,
            list(encoded_prompt),
            prompt_delta,
        )

    with AppendSafeJsonlWriter(base_path) as writer:
        checkpoint_examples = max(int(generation.get("checkpoint_examples", 96)), 1)
        for checkpoint_start in range(0, len(requests), checkpoint_examples):
            checkpoint_requests = requests[
                checkpoint_start : checkpoint_start + checkpoint_examples
            ]
            generated = backend.generate(checkpoint_requests)
            for request in checkpoint_requests:
                (
                    source,
                    prompt_ids,
                    human_ids,
                    prompt,
                    prompt_input_ids,
                    prompt_delta,
                ) = prepared[request.key]
                llm_ids = list(generated[request.key])
                generated_llm_count = len(llm_ids)
                minimum = int(
                    generation.get(
                        "min_new_tokens", generation["continuation_tokens"]
                    )
                )
                if len(llm_ids) < minimum:
                    raise ValueError(
                        f"generation for {request.key} stopped at {len(llm_ids)} tokens; "
                        f"minimum is {minimum}"
                    )
                length = min(
                    len(llm_ids),
                    int(generation["continuation_tokens"]),
                    len(human_ids),
                )
                llm_ids = llm_ids[:length]
                human_ids = human_ids[:length]
                human_text = _decode_visible(tokenizer, human_ids)
                llm_text = _decode_visible(tokenizer, llm_ids)
                tolerance = int(
                    run_config["contamination"].get("max_length_delta_tokens", 4)
                )
                human_roundtrip_count = len(
                    tokenizer.encode(human_text, add_special_tokens=False)
                )
                llm_roundtrip_count = len(
                    tokenizer.encode(llm_text, add_special_tokens=False)
                )
                human_delta = human_roundtrip_count - length
                llm_delta = llm_roundtrip_count - length
                normalized = abs(human_delta) > tolerance or abs(llm_delta) > tolerance
                if normalized:
                    human_text, human_ids, llm_text, llm_ids = (
                        _canonical_length_matched_pair(
                            tokenizer,
                            human_ids,
                            llm_ids,
                            int(generation["continuation_tokens"]),
                        )
                    )
                    length = len(llm_ids)
                    if len(human_ids) != length:
                        raise ValueError("canonical continuation lengths disagree")
                    if tokenizer.encode(
                        human_text, add_special_tokens=False
                    ) != human_ids or tokenizer.encode(
                        llm_text, add_special_tokens=False
                    ) != llm_ids:
                        raise ValueError(
                            "canonical continuation text/token IDs disagree"
                        )
                row = {
                    **source,
                    "target_model": run_config["target_model"],
                    "target_model_revision": run_config.get("target_model_revision"),
                    "target_model_resolved_revision": backend.resolved_model_revision,
                    "target_tokenizer_revision": run_config.get(
                        "target_tokenizer_revision"
                    ),
                    "target_tokenizer_resolved_revision": backend.resolved_tokenizer_revision,
                    "prompt": prompt,
                    "prompt_token_ids": prompt_ids,
                    "prompt_input_token_ids": prompt_input_ids,
                    "requested_prompt_token_count": int(generation["prompt_tokens"]),
                    "prompt_input_token_count": len(prompt_input_ids),
                    "prompt_roundtrip_length_delta_tokens": prompt_delta,
                    "human_continuation": human_text,
                    "human_token_ids": human_ids,
                    "llm_continuation": llm_text,
                    "llm_token_ids": llm_ids,
                    "continuation_token_count": length,
                    "generated_llm_token_count": generated_llm_count,
                    "initial_human_roundtrip_token_count": human_roundtrip_count,
                    "initial_llm_roundtrip_token_count": llm_roundtrip_count,
                    "initial_human_roundtrip_length_delta_tokens": human_delta,
                    "initial_llm_roundtrip_length_delta_tokens": llm_delta,
                    "continuation_roundtrip_normalized": normalized,
                    "generation_backend": generation["backend"],
                    "generation_temperature": generation["temperature"],
                    "generation_top_p": generation["top_p"],
                    "generation_max_new_tokens": generation["max_new_tokens"],
                }
                writer.write(row)
                progress.update(1, length, peak_gpu_memory_bytes())
            writer.checkpoint()
    return validate_unique_jsonl(base_path, _base_key, total_examples(run_config))


def build_tail_cache(
    run_config: dict[str, Any],
    base_path: str | Path,
    cache_path: str | Path,
    scorer_backend: Any,
) -> int:
    """Score/order candidate spans once per source and persist the result."""
    completed = completed_keys(cache_path, _tail_key)
    tokenizer = scorer_backend.tokenizer
    pending: list[tuple[dict[str, Any], list[list[int]]]] = []
    for base in iter_jsonl(base_path):
        if _tail_key(base) in completed:
            continue
        spans = sentence_spans(tokenizer, base["human_continuation"])
        pending.append((base, spans))
    with AppendSafeJsonlWriter(cache_path) as writer:
        checkpoint_sources = max(
            int(run_config["generation"].get("checkpoint_examples", 96)), 1
        )
        for checkpoint_start in range(0, len(pending), checkpoint_sources):
            checkpoint = pending[
                checkpoint_start : checkpoint_start + checkpoint_sources
            ]
            requests = [
                (f"{base['sample_id']}:{index}", base["prompt"], span)
                for base, spans in checkpoint
                for index, span in enumerate(spans)
            ]
            scored = scorer_backend.candidate_nll_batch(requests)
            for base, spans in checkpoint:
                nlls = [
                    scored[f"{base['sample_id']}:{index}"]
                    for index in range(len(spans))
                ]
                iterator = iter(nlls)
                ranked = rank_tail_candidates_once(
                    spans, lambda _span: next(iterator)
                )
                writer.write(
                    {
                        "dataset": run_config["dataset"],
                        "sample_id": base["sample_id"],
                        "tail_selection_model": run_config["target_model"],
                        "tail_selection_model_revision": scorer_backend.resolved_model_revision,
                        "tail_selection_tokenizer_revision": scorer_backend.resolved_tokenizer_revision,
                        "candidate_score_count": len(ranked),
                        "ranked_candidates": [
                            {
                                "candidate_id": item.candidate_id,
                                "token_ids": list(item.token_ids),
                                "nll": item.nll,
                            }
                            for item in ranked
                        ],
                    }
                )
            writer.checkpoint()
    return validate_unique_jsonl(cache_path, _tail_key, total_examples(run_config))


def _base_clean_row(
    run_config: dict[str, Any],
    base: dict[str, Any],
    label: str,
    text: str,
    token_count: int,
) -> dict[str, Any]:
    return {
        "run_id": run_config["run_id"],
        "dataset": run_config["dataset"],
        "dataset_id": base["dataset_id"],
        "dataset_config": base.get("dataset_config"),
        "dataset_revision": base.get("dataset_revision"),
        "dataset_resolved_revision": base.get("dataset_resolved_revision"),
        "source_id": base["source_id"],
        "sample_id": base["sample_id"],
        "split": base["split"],
        "generation_seed": base["generation_seed"],
        "target_model": run_config["target_model"],
        "target_model_revision": run_config.get("target_model_revision"),
        "target_model_resolved_revision": base.get("target_model_resolved_revision"),
        "target_tokenizer_revision": run_config.get("target_tokenizer_revision"),
        "target_tokenizer_resolved_revision": base.get(
            "target_tokenizer_resolved_revision"
        ),
        "prompt": base["prompt"],
        "text": text,
        "label": label,
        "contamination_mode": "none",
        "corruption_seed": None,
        "corruption_draw_id": 0,
        **contamination_counts(0.0, token_count, 0, 0),
    }


def construct_contaminated_data(
    run_config: dict[str, Any],
    base_path: str | Path,
    tail_cache_path: str | Path,
    output_path: str | Path,
    tokenizer: Any,
) -> int:
    """Construct fixed-length rows, resuming at the full provenance row key."""
    completed = completed_keys(output_path, data_row_key)
    tail_cache = {
        str(row["sample_id"]): row for row in iter_jsonl(tail_cache_path)
    }
    ratios = [
        float(ratio)
        for ratio in run_config["contamination"]["ratios"]
        if float(ratio) > 0
    ]
    random_draws = int(run_config["contamination"]["random_draws"])

    def decode_and_measure(token_ids: Sequence[int], counts: dict[str, Any]) -> str:
        text = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        final_count = len(tokenizer.encode(text, add_special_tokens=False))
        counts["final_token_count"] = final_count
        counts["length_delta_tokens"] = final_count - counts["original_token_count"]
        counts["realized_contamination_ratio"] = (
            counts["human_token_count"] / final_count if final_count else 0.0
        )
        tolerance = int(
            run_config["contamination"].get(
                "max_constructed_length_delta_tokens",
                run_config["contamination"].get("max_length_delta_tokens", 4),
            )
        )
        if abs(counts["length_delta_tokens"]) > tolerance:
            raise ValueError(
                f"decode/re-tokenize length drift {counts['length_delta_tokens']} "
                f"exceeds configured tolerance {tolerance}"
            )
        return text

    with AppendSafeJsonlWriter(output_path) as writer:
        for base in iter_jsonl(base_path):
            clean_rows = [
                _base_clean_row(
                    run_config,
                    base,
                    "human",
                    base["human_continuation"],
                    len(base["human_token_ids"]),
                ),
                _base_clean_row(
                    run_config,
                    base,
                    "llm",
                    base["llm_continuation"],
                    len(base["llm_token_ids"]),
                ),
            ]
            for row in clean_rows:
                if data_row_key(row) not in completed:
                    writer.write(row)
                    completed.add(data_row_key(row))

            human_spans = sentence_spans(tokenizer, base["human_continuation"])
            cached = tail_cache[str(base["sample_id"])]
            ranked = [
                TailCandidate(
                    int(item["candidate_id"]),
                    tuple(item["token_ids"]),
                    float(item["nll"]),
                )
                for item in cached["ranked_candidates"]
            ]
            for ratio in ratios:
                for draw_id in range(random_draws):
                    seed = corruption_seed(
                        int(run_config["contamination"]["corruption_seed"]),
                        run_config["dataset"],
                        run_config["target_model"],
                        str(base["sample_id"]),
                        draw_id,
                    )
                    mixed_ids, counts = random_token_contamination(
                        base["llm_token_ids"], human_spans, ratio, seed
                    )
                    mixed_text = decode_and_measure(mixed_ids, counts)
                    row = {
                        **clean_rows[1],
                        "text": mixed_text,
                        "contamination_mode": "random",
                        "corruption_seed": seed,
                        "corruption_draw_id": draw_id,
                        **counts,
                    }
                    if data_row_key(row) not in completed:
                        writer.write(row)
                        completed.add(data_row_key(row))

                mixed_ids, counts = tail_token_contamination(
                    base["llm_token_ids"], ranked, ratio
                )
                mixed_text = decode_and_measure(mixed_ids, counts)
                row = {
                    **clean_rows[1],
                    "text": mixed_text,
                    "contamination_mode": "tail",
                    "corruption_seed": None,
                    "corruption_draw_id": 0,
                    "tail_selection_model": cached["tail_selection_model"],
                    "tail_selection_model_revision": cached[
                        "tail_selection_model_revision"
                    ],
                    "tail_selection_tokenizer_revision": cached[
                        "tail_selection_tokenizer_revision"
                    ],
                    **counts,
                }
                if data_row_key(row) not in completed:
                    writer.write(row)
                    completed.add(data_row_key(row))

    expected = total_examples(run_config) * expected_rows_per_source(
        run_config["contamination"]["ratios"], random_draws
    )
    return validate_unique_jsonl(output_path, data_row_key, expected)


def prepare_run_data(
    run_config: dict[str, Any],
    source_path: str | Path,
    run_dir: str | Path,
) -> dict[str, str]:
    directory = Path(run_dir)
    base_path = directory / "base_generations.jsonl"
    tail_path = directory / "tail_candidate_cache.jsonl"
    data_path = directory / "data.jsonl"
    backend = make_generation_backend(
        run_config["target_model"],
        run_config.get("target_model_revision"),
        run_config.get("target_tokenizer_revision"),
        run_config["generation"],
    )
    generate_base_examples(run_config, source_path, base_path, backend)
    tail_backend = backend
    if run_config["generation"]["backend"] == "vllm":
        # Generation remains vLLM; the dependable exact Transformers model is
        # loaded only for the white-box tail scoring stage.
        del backend
        tail_config = dict(run_config["generation"])
        tail_config["backend"] = "transformers"
        tail_backend = TransformersBackend(
            run_config["target_model"],
            run_config.get("target_model_revision"),
            run_config.get("target_tokenizer_revision"),
            tail_config,
        )
    build_tail_cache(run_config, base_path, tail_path, tail_backend)
    construct_contaminated_data(
        run_config, base_path, tail_path, data_path, tail_backend.tokenizer
    )
    write_completion_marker(
        directory / "prepare.complete.json",
        {
            "base_rows": total_examples(run_config),
            "data_rows": sum(1 for _ in iter_jsonl(data_path)),
            "tail_cache_rows": total_examples(run_config),
        },
    )
    return {
        "base_generations": str(base_path),
        "tail_candidate_cache": str(tail_path),
        "data": str(data_path),
    }


def score_run(
    run_config: dict[str, Any],
    data_path: str | Path,
    run_dir: str | Path,
    include_binoculars: bool = True,
) -> dict[str, str]:
    directory = Path(run_dir)
    target_path = directory / "target_scores.jsonl"
    target_scorer = TargetModelScorer(
        run_config["target_model"],
        run_config.get("target_model_revision"),
        run_config.get("target_tokenizer_revision"),
        run_config["scoring"],
    )
    target_count = score_jsonl(
        data_path, target_path, target_scorer, run_config["scoring"]
    )
    del target_scorer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    outputs = {"target_scores": str(target_path)}
    binoculars_count = None
    if include_binoculars:
        binoculars_path = directory / "binoculars_scores.jsonl"
        pair_scorer = BinocularsScorer(run_config["models"], run_config["scoring"])
        binoculars_count = score_jsonl(
            data_path, binoculars_path, pair_scorer, run_config["scoring"]
        )
        outputs["binoculars_scores"] = str(binoculars_path)
    write_completion_marker(
        directory / "score.complete.json",
        {
            "target_score_rows": target_count,
            "binoculars_score_rows": binoculars_count,
        },
    )
    return outputs


def evaluate_run(
    run_config: dict[str, Any],
    run_dir: str | Path,
    results_dir: str | Path,
) -> str:
    directory = Path(run_dir)
    output = Path(results_dir) / "metrics.csv"
    binoculars = directory / "binoculars_scores.jsonl"
    rows = evaluate(
        directory / "target_scores.jsonl",
        output,
        run_config,
        binoculars if binoculars.exists() else None,
    )
    write_completion_marker(
        Path(results_dir) / "evaluate.complete.json",
        {"metric_rows": len(rows), "metrics_csv": str(output.resolve())},
    )
    return str(output)
