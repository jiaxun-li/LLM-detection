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


def numpy_exact_token_features(
    logits: np.ndarray,
    target_ids: np.ndarray,
    saved_top_k: int = 0,
) -> dict[str, np.ndarray]:
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
    result = {
        "logp": logp,
        "rank": rank,
        "log_rank": np.log(rank),
        "entropy": entropy,
    }
    if saved_top_k:
        top_k = min(max(int(saved_top_k), 1), values.shape[-1])
        if top_k < 2:
            raise ValueError("saved_top_k requires a model vocabulary of at least 2")
        top_ids = np.argsort(-values, axis=-1, kind="stable")[:, :top_k]
        top_logits = np.take_along_axis(values, top_ids, axis=-1)
        result.update(
            {
                "top_k_token_ids": top_ids,
                "top_k_logprobs": top_logits - log_z[:, None],
                "top1_top2_logprob_margin": top_logits[:, 0] - top_logits[:, 1],
                # Zero means the observed token was a top-1 prediction. More
                # negative values indicate a larger gap below the top choice.
                "target_top1_logprob_margin": target_logits - top_logits[:, 0],
            }
        )
    return result


def exact_token_features(
    logits: Any,
    target_ids: Any,
    vocab_chunk_size: int,
    saved_top_k: int = 0,
) -> dict[str, Any]:
    """Compute exact token statistics and an optional compact top-k feature pack.

    Vocabulary chunks bound temporary memory without approximate top-k ranks or
    truncated entropy. The model's logits are reused for every target detector.
    ``target_top1_logprob_margin`` is target log-probability minus top-1
    log-probability, so it is non-positive and equals zero for a top-1 target.
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
    result = {
        "logp": target_logits - log_z,
        "rank": rank,
        "log_rank": torch.log(rank.float()),
        "entropy": log_z - expected_logit,
    }
    if saved_top_k:
        top_k = min(max(int(saved_top_k), 1), int(logits.shape[-1]))
        if top_k < 2:
            raise ValueError("saved_top_k requires a model vocabulary of at least 2")
        top_logits, top_ids = torch.topk(
            logits.float(), k=top_k, dim=-1, largest=True, sorted=True
        )
        result.update(
            {
                "top_k_token_ids": top_ids,
                "top_k_logprobs": top_logits - log_z[:, None],
                "top1_top2_logprob_margin": top_logits[:, 0] - top_logits[:, 1],
                "target_top1_logprob_margin": target_logits - top_logits[:, 0],
            }
        )
    return result


def binoculars_score(mean_performer_nll: float, mean_observer_to_performer_xent: float) -> float:
    """Legacy binocular-gap score (NOT the published Binoculars ratio).

    numerator   = exp(mean NLL under performer/instruct model)
    denominator = exp(mean H(observer/base distribution, performer distribution))
    score       = numerator / denominator

    Kept unchanged for historical score packs. The published score divides
    mean NLL by mean cross-entropy, without either exponential.
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
    *,
    tokenizer_use_fast: bool | None = True,
    trust_remote_code: bool = False,
) -> tuple[Any, Any, Any]:
    torch = _torch()
    AutoModelForCausalLM, AutoTokenizer = _transformers()
    tokenizer_kwargs: dict[str, Any] = {
        "revision": tokenizer_revision,
        "trust_remote_code": trust_remote_code,
    }
    if tokenizer_use_fast is not None:
        tokenizer_kwargs["use_fast"] = bool(tokenizer_use_fast)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs: dict[str, Any] = {
        "revision": revision,
        "torch_dtype": dtype_from_name(torch, dtype),
        "trust_remote_code": trust_remote_code,
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


def _encode_row(
    tokenizer: Any,
    row: dict[str, Any],
    max_tokens: int | None,
    context_policy: str = "prompt_conditioned_response_only",
) -> tuple[list[int], int, int, dict[str, Any]]:
    """Encode one scoring row and identify its shifted-token evidence window.

    The repository's synthetic study keeps its historical prompt-conditioned
    policy. Beemo's DetectLLM-style baseline instead tokenizes only the released
    response with tokenizer-default special-token behavior. Binoculars uses the
    same output-only input but applies its upstream 512-token truncation.
    """
    if context_policy == "prompt_conditioned_response_only":
        if max_tokens is None:
            raise ValueError("prompt-conditioned scoring requires scoring.max_tokens")
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        text_ids = tokenizer.encode(row["text"], add_special_tokens=False)
        full = prompt_ids + text_ids
        original_num_input_tokens = len(full)
        if len(full) > int(max_tokens):
            allowed_text = int(max_tokens) - len(prompt_ids)
            if allowed_text <= 0:
                raise ValueError(f"prompt exceeds scoring.max_tokens for {row['sample_id']}")
            text_ids = text_ids[:allowed_text]
            full = prompt_ids + text_ids
        if not prompt_ids or not text_ids:
            raise ValueError(f"empty prompt/continuation for {row['sample_id']}")
        return full, len(prompt_ids) - 1, len(text_ids), {
            "original_num_input_tokens": original_num_input_tokens,
            "boundary_token_added": False,
            "truncated_token_count": original_num_input_tokens - len(full),
        }

    if context_policy == "detectllm_output_only":
        # Deliberately omit add_special_tokens so each model retains its native
        # tokenizer default, as in the cited DetectLLM baseline implementation.
        token_ids = tokenizer.encode(row["text"])
        original_num_input_tokens = len(token_ids)
        if max_tokens is not None and len(token_ids) > int(max_tokens):
            if int(max_tokens) < 2:
                raise ValueError("output-only scoring.max_tokens must be at least 2")
            # Keep the beginning of the released response. This is explicit,
            # model-specific right truncation, never prompt concatenation.
            token_ids = token_ids[: int(max_tokens)]
    elif context_policy == "binoculars_output_only_512":
        limit = 512 if max_tokens is None else int(max_tokens)
        if limit < 2:
            raise ValueError("Binoculars max_tokens must be at least 2")
        token_ids = tokenizer.encode(row["text"])
        original_num_input_tokens = len(token_ids)
        token_ids = token_ids[:limit]
    else:
        raise ValueError(f"unknown scoring context policy: {context_policy}")

    if not token_ids:
        raise ValueError(f"zero output tokens for {row['sample_id']}")
    boundary_token_added = False
    if len(token_ids) == 1:
        boundary_token_id = getattr(tokenizer, "bos_token_id", None)
        if boundary_token_id is None:
            boundary_token_id = getattr(tokenizer, "eos_token_id", None)
        if boundary_token_id is None:
            raise ValueError(
                f"one output token and no BOS/EOS boundary token for {row['sample_id']}"
            )
        token_ids = [int(boundary_token_id), *token_ids]
        boundary_token_added = True
    # Causal shifting scores token_ids[1:] from logits at positions [:-1]. If a
    # tokenizer inserts BOS, the first text token is therefore scored; otherwise
    # the first text token is the unscored context token, matching upstream.
    return token_ids, 0, len(token_ids) - 1, {
        "original_num_input_tokens": original_num_input_tokens,
        "boundary_token_added": boundary_token_added,
        "truncated_token_count": max(
            original_num_input_tokens - (len(token_ids) - int(boundary_token_added)),
            0,
        ),
    }


def _padded_batch(
    tokenizer: Any,
    encoded: Sequence[tuple[list[int], int, int, dict[str, Any]]],
    device: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    maximum = max(len(item[0]) for item in encoded)
    input_ids = torch.full(
        (len(encoded), maximum),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros_like(input_ids)
    for index, (token_ids, _, _, _) in enumerate(encoded):
        input_ids[index, : len(token_ids)] = torch.tensor(token_ids, device=device)
        attention[index, : len(token_ids)] = 1
    return input_ids, attention


class TargetModelScorer:
    """One target-model forward pass supplies all six single-model detectors."""

    def __init__(self, model_id: str, revision: str | None, tokenizer_revision: str | None, config: dict[str, Any]):
        self.config = config
        self.model_id = model_id
        self.scorer_key = str(config.get("scorer_key", model_id))
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.context_policy = config.get(
            "context_policy", "prompt_conditioned_response_only"
        )
        self.max_tokens = config.get("max_tokens")
        self.feature_schema = (
            "target-token-features-v4-output-only-boundary-truncation"
            if self.context_policy == "detectllm_output_only"
            else "target-token-features-v2"
        )
        self.tokenizer, self.model, self.device = _load_model(
            model_id,
            revision,
            tokenizer_revision,
            config["dtype"],
            config["device"],
            config.get("device_map"),
            tokenizer_use_fast=config.get("tokenizer_use_fast", True),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
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
        return len(
            _encode_row(
                self.tokenizer,
                row,
                self.max_tokens,
                self.context_policy,
            )[0]
        )

    def adjustment_info(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return deterministic boundary/truncation provenance without scoring."""
        return _encode_row(
            self.tokenizer,
            row,
            self.max_tokens,
            self.context_policy,
        )[3]

    def validate_existing_row(self, row: dict[str, Any]) -> None:
        """Refuse to append a new feature schema or model revision to an old run."""
        if row.get("scoring_feature_schema") != self.feature_schema:
            raise ValueError(
                "existing target score rows use a different feature schema; "
                "choose a new run ID"
            )
        if row.get("scoring_model") != self.model_id:
            raise ValueError(
                "existing target score rows use a different scoring model; "
                "choose a new run ID"
            )
        if row.get("scorer_key", self.scorer_key) != self.scorer_key:
            raise ValueError(
                "existing target score rows use a different scorer key; "
                "choose a new run ID"
            )
        if row.get(
            "scoring_context_policy", "prompt_conditioned_response_only"
        ) != self.context_policy:
            raise ValueError(
                "existing target score rows use a different context policy; "
                "choose a new run ID"
            )
        if row.get("scoring_max_tokens") != self.max_tokens:
            raise ValueError(
                "existing target score rows use a different token limit; "
                "choose a new run ID"
            )
        if (
            row.get("scoring_model_revision") != self.resolved_revision
            or row.get("scoring_tokenizer_revision")
            != self.resolved_tokenizer_revision
        ):
            raise ValueError(
                "existing target score rows use different resolved model/tokenizer "
                "revisions; choose a new run ID"
            )
        features = row.get("token_features", {})
        top_ids = features.get("top_k_token_ids", [])
        expected_top_k = int(self.config.get("saved_top_k", 10))
        if not top_ids or any(len(token_ids) != expected_top_k for token_ids in top_ids):
            raise ValueError(
                "existing target score rows use a different saved_top_k; "
                "choose a new run ID"
            )
        expected_pooled = bool(
            self.config.get("save_mean_pooled_final_hidden_state", False)
        )
        if ("document_features" in row) != expected_pooled:
            raise ValueError(
                "existing target score rows use a different pooled-hidden-state "
                "setting; choose a new run ID"
            )

    def score_batch(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        torch = _torch()
        saved_top_k = int(self.config.get("saved_top_k", 10))
        save_pooled_hidden = bool(
            self.config.get("save_mean_pooled_final_hidden_state", False)
        )
        encoded = [
            _encode_row(
                self.tokenizer,
                row,
                self.max_tokens,
                self.context_policy,
            )
            for row in rows
        ]
        input_ids, attention = _padded_batch(self.tokenizer, encoded, self.device)
        with torch.inference_mode():
            # This is the sole target-model forward for all single-model features.
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention,
                output_hidden_states=save_pooled_hidden,
                return_dict=True,
            )
            batch_logits = model_outputs.logits[:, :-1, :]
            batch_final_hidden = (
                model_outputs.hidden_states[-1]
                if save_pooled_hidden
                else None
            )
            # Drop references to all intermediate hidden-state tensors. Only the
            # final layer remains live when optional pooling is enabled.
            del model_outputs
        outputs = []
        for index, (row, (_, start, length, _)) in enumerate(
            zip(rows, encoded)
        ):
            logits = batch_logits[index, start : start + length]
            targets = input_ids[index, start + 1 : start + 1 + length]
            tensor_features = exact_token_features(
                logits,
                targets,
                int(self.config["vocab_chunk_size"]),
                saved_top_k,
            )
            features = {
                name: values.detach().cpu().tolist()
                for name, values in tensor_features.items()
            }
            scored_row = {
                **row,
                "scoring_model": self.model_id,
                "scorer_key": self.scorer_key,
                "scoring_model_revision": self.resolved_revision,
                "scoring_tokenizer_revision": self.resolved_tokenizer_revision,
                "scoring_feature_schema": self.feature_schema,
                "scoring_context_policy": self.context_policy,
                "scoring_max_tokens": self.max_tokens,
                "num_scored_tokens": length,
                "token_features": features,
                "doc_scores": single_model_doc_scores(features),
            }
            if batch_final_hidden is not None:
                pooled = (
                    batch_final_hidden[index, start : start + length]
                    .float()
                    .mean(dim=0)
                )
                scored_row["document_features"] = {
                    "mean_pooled_final_hidden_state": pooled.detach()
                    .cpu()
                    .tolist(),
                    "pooling_token_count": length,
                    "hidden_size": int(pooled.numel()),
                }
            outputs.append(scored_row)
        del batch_logits
        if batch_final_hidden is not None:
            del batch_final_hidden
        return outputs


class BinocularsScorer:
    """Falcon pair scorer with explicit roles on independently configured GPUs."""

    def __init__(self, model_config: dict[str, Any], scoring_config: dict[str, Any]):
        pair = model_config["binoculars"]
        self.config = scoring_config
        self.context_policy = pair.get(
            "context_policy", "prompt_conditioned_response_only"
        )
        self.max_tokens = pair.get("max_tokens", scoring_config.get("max_tokens"))
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
            tokenizer_use_fast=pair.get("tokenizer_use_fast", True),
            trust_remote_code=bool(pair.get("trust_remote_code", False)),
        )
        observer_tokenizer, self.observer, self.observer_device = _load_model(
            self.observer_id,
            self.observer_revision,
            self.observer_revision,
            scoring_config["dtype"],
            pair["observer_device"],
            None,
            tokenizer_use_fast=pair.get("tokenizer_use_fast", True),
            trust_remote_code=bool(pair.get("trust_remote_code", False)),
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

    @property
    def feature_schema(self) -> str:
        return (
            "binoculars-token-features-v3-output-only-boundary-truncation"
            if self.context_policy == "binoculars_output_only_512"
            else "binoculars-token-features-v1"
        )

    def validate_existing_row(self, row: dict[str, Any]) -> None:
        if row.get(
            "scoring_feature_schema", "binoculars-token-features-v1"
        ) != self.feature_schema:
            raise ValueError(
                "existing Binoculars rows use a different feature schema; "
                "choose a new run ID"
            )
        if row.get(
            "scoring_context_policy", "prompt_conditioned_response_only"
        ) != self.context_policy:
            raise ValueError(
                "existing Binoculars rows use a different context policy; "
                "choose a new run ID"
            )
        if row.get("scoring_max_tokens") != self.max_tokens:
            raise ValueError(
                "existing Binoculars rows use a different token limit; "
                "choose a new run ID"
            )
        expected = {
            "binoculars_performer_model": self.performer_id,
            "binoculars_performer_revision": self.performer_resolved_revision,
            "binoculars_observer_model": self.observer_id,
            "binoculars_observer_revision": self.observer_resolved_revision,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "existing Binoculars rows use different models or revisions; "
                "choose a new run ID"
            )

    def token_count(self, row: dict[str, Any]) -> int:
        return len(
            _encode_row(
                self.tokenizer,
                row,
                self.max_tokens,
                self.context_policy,
            )[0]
        )

    def adjustment_info(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return deterministic boundary/truncation provenance without scoring."""
        return _encode_row(
            self.tokenizer,
            row,
            self.max_tokens,
            self.context_policy,
        )[3]

    def score_batch(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        torch = _torch()
        encoded = [
            _encode_row(
                self.tokenizer,
                row,
                self.max_tokens,
                self.context_policy,
            )
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
        for index, (row, (_, start, length, _)) in enumerate(
            zip(rows, encoded)
        ):
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
                    "scoring_context_policy": self.context_policy,
                    "scoring_max_tokens": self.max_tokens,
                    "scoring_feature_schema": self.feature_schema,
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
    validate_existing = getattr(scorer, "validate_existing_row", None)
    if validate_existing is not None and Path(output_path).exists():
        for existing_row in iter_jsonl(output_path):
            validate_existing(existing_row)
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
