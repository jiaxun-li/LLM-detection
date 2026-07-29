"""Batched exact detector scoring with resumable streaming output."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .data import data_row_key
from .generation import dtype_from_name, length_bucketed
from .io import AppendSafeJsonlWriter, completed_keys, iter_jsonl
from .runtime import Throughput, peak_gpu_memory_bytes


EPS = 1e-12


def numpy_exact_token_features(logits: np.ndarray, target_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Reference implementation used by unit tests.

    Ties receive the same one-based competition rank: ``1 + count(logit >
    target_logit)``.
    """
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(target_ids, dtype=np.int64)
    target_logits = values[np.arange(len(targets)), targets]
    maxima = values.max(axis=-1, keepdims=True)
    exponentials = np.exp(values - maxima)
    normalizers = exponentials.sum(axis=-1, keepdims=True)
    probabilities = exponentials / normalizers
    log_z = maxima[:, 0] + np.log(normalizers[:, 0])
    logp = target_logits - log_z
    rank = 1 + (values > target_logits[:, None]).sum(axis=-1)
    entropy = -(probabilities * np.log(probabilities)).sum(axis=-1)
    return {
        "logp": logp,
        "rank": rank,
        "log_rank": np.log(rank),
        "entropy": entropy,
    }


def exact_token_features(
    logits: Any,
    target_ids: Any,
    vocab_chunk_size: int,
) -> dict[str, Any]:
    """Compute log-probability, exact rank, log-rank, and entropy.

    Vocabulary chunks bound temporary memory without approximate top-k ranks or
    truncated entropy. The model's logits are reused for every target detector.
    """
    torch = _torch()
    targets = target_ids.long()
    target_logits_raw = logits.gather(-1, targets[:, None]).squeeze(-1)
    target_logits = target_logits_raw.float()
    log_z = torch.full_like(target_logits, -torch.inf)
    rank = torch.ones_like(targets, dtype=torch.long)
    width = max(int(vocab_chunk_size), 1)
    for start in range(0, logits.shape[-1], width):
        raw_chunk = logits[:, start : start + width]
        chunk = raw_chunk.float()
        rank += (raw_chunk > target_logits_raw[:, None]).sum(dim=-1)
        log_z = torch.logaddexp(log_z, torch.logsumexp(chunk, dim=-1))
    expected_logit = torch.zeros_like(log_z)
    for start in range(0, logits.shape[-1], width):
        chunk = logits[:, start : start + width].float()
        expected_logit += (torch.exp(chunk - log_z[:, None]) * chunk).sum(dim=-1)
    return {
        "logp": target_logits - log_z,
        "rank": rank,
        "log_rank": torch.log(rank.float()),
        "entropy": log_z - expected_logit,
    }


def binoculars_score(mean_performer_nll: float, mean_observer_to_performer_xent: float) -> float:
    """Upstream-style Binoculars ratio.

    numerator   = exp(mean NLL under performer/instruct model)
    denominator = exp(mean H(observer/base distribution, performer distribution))
    score       = numerator / denominator
    """
    return float(math.exp(mean_performer_nll - mean_observer_to_performer_xent))


def numpy_cross_entropy(
    observer_logits: np.ndarray,
    performer_logits: np.ndarray,
) -> np.ndarray:
    observer = np.asarray(observer_logits, dtype=np.float64)
    performer = np.asarray(performer_logits, dtype=np.float64)
    observer_max = observer.max(axis=-1, keepdims=True)
    observer_exp = np.exp(observer - observer_max)
    observer_p = observer_exp / observer_exp.sum(axis=-1, keepdims=True)
    performer_max = performer.max(axis=-1, keepdims=True)
    performer_log_z = performer_max + np.log(
        np.exp(performer - performer_max).sum(axis=-1, keepdims=True)
    )
    performer_log_p = performer - performer_log_z
    return -(observer_p * performer_log_p).sum(axis=-1)


def exact_observer_to_performer_cross_entropy(
    observer_logits: Any,
    performer_logits: Any,
    vocab_chunk_size: int,
) -> Any:
    torch = _torch()
    observer_log_z = torch.full(
        observer_logits.shape[:-1],
        -torch.inf,
        dtype=torch.float32,
        device=observer_logits.device,
    )
    performer_log_z = torch.full_like(observer_log_z, -torch.inf)
    width = max(int(vocab_chunk_size), 1)
    for start in range(0, observer_logits.shape[-1], width):
        observer_chunk = observer_logits[:, start : start + width].float()
        performer_chunk = performer_logits[:, start : start + width].float()
        observer_log_z = torch.logaddexp(
            observer_log_z, torch.logsumexp(observer_chunk, dim=-1)
        )
        performer_log_z = torch.logaddexp(
            performer_log_z, torch.logsumexp(performer_chunk, dim=-1)
        )
    cross_entropy = torch.zeros_like(observer_log_z)
    for start in range(0, observer_logits.shape[-1], width):
        obs_chunk = observer_logits[:, start : start + width].float()
        perf_chunk = performer_logits[:, start : start + width].float()
        probabilities = torch.exp(obs_chunk - observer_log_z[:, None])
        performer_logp = perf_chunk - performer_log_z[:, None]
        cross_entropy -= (probabilities * performer_logp).sum(dim=-1)
    return cross_entropy


def single_model_doc_scores(features: dict[str, Sequence[float]]) -> dict[str, float]:
    logp = np.asarray(features["logp"], dtype=float)
    rank = np.asarray(features["rank"], dtype=float)
    log_rank = np.asarray(features["log_rank"], dtype=float)
    entropy = np.asarray(features["entropy"], dtype=float)
    nll = -logp
    return {
        "log_likelihood": float(logp.mean()),
        "rank": float(rank.mean()),
        "log_rank": float(log_rank.mean()),
        "lrr": float(nll.mean() / (log_rank.mean() + EPS)),
        "entropy": float(entropy.mean()),
        "entropy_gap": float((nll - entropy).mean()),
    }


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("scoring requires torch on the execution host") from exc
    return torch


def _transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("scoring requires transformers on the execution host") from exc
    return AutoModelForCausalLM, AutoTokenizer


def _load_model(
    model_id: str,
    revision: str | None,
    tokenizer_revision: str | None,
    dtype: str,
    device: str,
    device_map: Any,
) -> tuple[Any, Any, Any]:
    torch = _torch()
    AutoModelForCausalLM, AutoTokenizer = _transformers()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=tokenizer_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs: dict[str, Any] = {
        "revision": revision,
        "torch_dtype": dtype_from_name(torch, dtype),
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device_map is None:
        model.to(device)
    model.eval()
    return tokenizer, model, next(model.parameters()).device


def _compatible_tokenizers(first: Any, second: Any) -> None:
    probes = [
        "The quick brown fox jumps over the lazy dog.",
        "Numbers 10, 20, 30; symbols !?;:",
        "Unicode café 北京",
    ]
    for probe in probes:
        if first.encode(probe, add_special_tokens=False) != second.encode(
            probe, add_special_tokens=False
        ):
            raise ValueError("Binoculars performer and observer tokenizers are incompatible")


def _encode_row(tokenizer: Any, row: dict[str, Any], max_tokens: int) -> tuple[list[int], int, int]:
    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
    text_ids = tokenizer.encode(row["text"], add_special_tokens=False)
    full = prompt_ids + text_ids
    if len(full) > max_tokens:
        allowed_text = max_tokens - len(prompt_ids)
        if allowed_text <= 0:
            raise ValueError(f"prompt exceeds scoring.max_tokens for {row['sample_id']}")
        text_ids = text_ids[:allowed_text]
        full = prompt_ids + text_ids
    if not prompt_ids or not text_ids:
        raise ValueError(f"empty prompt/continuation for {row['sample_id']}")
    start = len(prompt_ids) - 1
    return full, start, len(text_ids)


def _padded_batch(tokenizer: Any, encoded: Sequence[tuple[list[int], int, int]], device: Any) -> tuple[Any, Any]:
    torch = _torch()
    maximum = max(len(item[0]) for item in encoded)
    input_ids = torch.full(
        (len(encoded), maximum),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros_like(input_ids)
    for index, (token_ids, _, _) in enumerate(encoded):
        input_ids[index, : len(token_ids)] = torch.tensor(token_ids, device=device)
        attention[index, : len(token_ids)] = 1
    return input_ids, attention


class TargetModelScorer:
    """One target-model forward pass supplies all six single-model detectors."""

    def __init__(self, model_id: str, revision: str | None, tokenizer_revision: str | None, config: dict[str, Any]):
        self.config = config
        self.model_id = model_id
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.tokenizer, self.model, self.device = _load_model(
            model_id,
            revision,
            tokenizer_revision,
            config["dtype"],
            config["device"],
            config.get("device_map"),
        )
        self.resolved_revision = (
            getattr(self.model.config, "_commit_hash", None) or revision
        )
        self.resolved_tokenizer_revision = (
            self.tokenizer.init_kwargs.get("_commit_hash")
            or tokenizer_revision
            or self.resolved_revision
        )

    def token_count(self, row: dict[str, Any]) -> int:
        return len(self.tokenizer.encode(row["prompt"] + row["text"], add_special_tokens=False))

    def score_batch(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        torch = _torch()
        encoded = [
            _encode_row(self.tokenizer, row, int(self.config["max_tokens"]))
            for row in rows
        ]
        input_ids, attention = _padded_batch(self.tokenizer, encoded, self.device)
        with torch.inference_mode():
            # This is the sole target-model forward for all single-model features.
            batch_logits = self.model(input_ids=input_ids, attention_mask=attention).logits[:, :-1, :]
        outputs = []
        for index, (row, (_, start, length)) in enumerate(zip(rows, encoded)):
            logits = batch_logits[index, start : start + length]
            targets = input_ids[index, start + 1 : start + 1 + length]
            tensor_features = exact_token_features(
                logits,
                targets,
                int(self.config["vocab_chunk_size"]),
            )
            features = {
                name: values.detach().cpu().tolist()
                for name, values in tensor_features.items()
            }
            outputs.append(
                {
                    **row,
                    "scoring_model": self.model_id,
                    "scoring_model_revision": self.resolved_revision,
                    "scoring_tokenizer_revision": self.resolved_tokenizer_revision,
                    "num_scored_tokens": length,
                    "token_features": features,
                    "doc_scores": single_model_doc_scores(features),
                }
            )
        del batch_logits
        return outputs


class BinocularsScorer:
    """Falcon pair scorer with explicit roles on independently configured GPUs."""

    def __init__(self, model_config: dict[str, Any], scoring_config: dict[str, Any]):
        pair = model_config["binoculars"]
        self.config = scoring_config
        self.performer_id = pair["performer"]
        self.observer_id = pair["observer"]
        self.performer_revision = pair.get("performer_revision")
        self.observer_revision = pair.get("observer_revision")
        self.tokenizer, self.performer, self.performer_device = _load_model(
            self.performer_id,
            self.performer_revision,
            self.performer_revision,
            scoring_config["dtype"],
            pair["performer_device"],
            None,
        )
        observer_tokenizer, self.observer, self.observer_device = _load_model(
            self.observer_id,
            self.observer_revision,
            self.observer_revision,
            scoring_config["dtype"],
            pair["observer_device"],
            None,
        )
        _compatible_tokenizers(self.tokenizer, observer_tokenizer)
        self.performer_resolved_revision = (
            getattr(self.performer.config, "_commit_hash", None)
            or self.performer_revision
        )
        self.observer_resolved_revision = (
            getattr(self.observer.config, "_commit_hash", None)
            or self.observer_revision
        )

    def token_count(self, row: dict[str, Any]) -> int:
        return len(self.tokenizer.encode(row["prompt"] + row["text"], add_special_tokens=False))

    def score_batch(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        torch = _torch()
        encoded = [
            _encode_row(self.tokenizer, row, int(self.config["max_tokens"]))
            for row in rows
        ]
        performer_ids, performer_attention = _padded_batch(
            self.tokenizer, encoded, self.performer_device
        )
        observer_ids = performer_ids.to(self.observer_device)
        observer_attention = performer_attention.to(self.observer_device)
        with torch.inference_mode():
            performer_batch = self.performer(
                input_ids=performer_ids, attention_mask=performer_attention
            ).logits[:, :-1, :]
            observer_batch = self.observer(
                input_ids=observer_ids, attention_mask=observer_attention
            ).logits[:, :-1, :]
        outputs = []
        for index, (row, (_, start, length)) in enumerate(zip(rows, encoded)):
            performer_logits = performer_batch[index, start : start + length].float()
            observer_logits = observer_batch[index, start : start + length].to(
                self.performer_device
            ).float()
            targets = performer_ids[index, start + 1 : start + 1 + length]
            target_logits = performer_logits.gather(-1, targets[:, None]).squeeze(-1)
            performer_nll = -(target_logits - torch.logsumexp(performer_logits, dim=-1))
            cross_entropy = exact_observer_to_performer_cross_entropy(
                observer_logits,
                performer_logits,
                int(self.config["vocab_chunk_size"]),
            )
            mean_nll = float(performer_nll.mean().item())
            mean_xent = float(cross_entropy.mean().item())
            outputs.append(
                {
                    **row,
                    "binoculars_performer_model": self.performer_id,
                    "binoculars_performer_revision": self.performer_resolved_revision,
                    "binoculars_observer_model": self.observer_id,
                    "binoculars_observer_revision": self.observer_resolved_revision,
                    "binoculars_formula": "exp(mean_performer_nll) / exp(mean_H(observer, performer))",
                    "num_scored_tokens": length,
                    "token_features": {
                        "performer_nll": performer_nll.detach().cpu().tolist(),
                        "observer_to_performer_cross_entropy": cross_entropy.detach()
                        .cpu()
                        .tolist(),
                    },
                    "doc_scores": {
                        "binoculars": binoculars_score(mean_nll, mean_xent),
                        "binoculars_numerator_perplexity": math.exp(mean_nll),
                        "binoculars_denominator_cross_perplexity": math.exp(mean_xent),
                    },
                }
            )
        del performer_batch, observer_batch
        return outputs


def score_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    scorer: TargetModelScorer | BinocularsScorer,
    scoring_config: dict[str, Any],
) -> int:
    """Stream rows, length-bucket them, microbatch, and resume by complete row key."""
    completed = completed_keys(output_path, data_row_key)
    pending = (row for row in iter_jsonl(input_path) if data_row_key(row) not in completed)
    total = sum(1 for _ in iter_jsonl(input_path))
    progress = Throughput(total, "scoring")
    progress.examples = len(completed)
    written = len(completed)
    with AppendSafeJsonlWriter(output_path) as writer:
        for batch in length_bucketed(
            pending,
            scorer.token_count,
            int(scoring_config["batch_size"]),
            int(scoring_config["length_bucket_width"]),
        ):
            micro = max(int(scoring_config["microbatch_size"]), 1)
            for start in range(0, len(batch), micro):
                part = batch[start : start + micro]
                scored = scorer.score_batch(part)
                for row in scored:
                    writer.write(row)
                    written += 1
                progress.update(
                    len(scored),
                    sum(int(row["num_scored_tokens"]) for row in scored),
                    peak_gpu_memory_bytes(),
                )
    if written != total:
        raise ValueError(f"scoring produced {written} rows; expected {total}")
    return written
