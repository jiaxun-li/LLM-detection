"""Exact, resumable scorer packs for the RAID benchmark.

The module deliberately contains no command-line or stage orchestration.  It
provides the two frozen scorers and a restart-safe JSONL driver used by the
RAID pipeline.  Both scorers consume released output text only and expose the
exact target-token IDs used by inference so realized contamination can be
computed on the scored window rather than on untruncated text.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from RAID.raid_data import raid_row_key as prepared_raid_row_key
from experiment_core.preparation.generation import length_bucketed
from experiment_core.infrastructure.io import AppendSafeJsonlWriter, completed_keys, iter_jsonl
from experiment_core.infrastructure.runtime import Throughput, peak_gpu_memory_bytes
from experiment_core.detectors.scoring import (
    _compatible_tokenizers,
    _load_model,
    _padded_batch,
    _torch,
    binoculars_score,
    exact_observer_to_performer_cross_entropy,
    exact_token_features,
    single_model_doc_scores,
)


FALCON_MODEL_ID = "tiiuae/falcon-7b"
BINOCULARS_OBSERVER_ID = FALCON_MODEL_ID
BINOCULARS_PERFORMER_ID = "tiiuae/falcon-7b-instruct"
DEFAULT_MAX_TOKENS = 512
FALCON_FEATURE_SCHEMA = "raid-falcon-token-features-v1"
BINOCULARS_FEATURE_SCHEMA = "raid-binoculars-token-features-v1"
SCORE_ROW_SCHEMA = "raid-score-row-v1"


def _first_present(row: dict[str, Any], names: Sequence[str], default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def raid_row_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Return the stable identity of one prepared RAID row.

    Preparation may supply an explicit ``row_key``.  Otherwise the key is
    derived from source, row role, selected generation, and attack condition.
    Field aliases make the scorer tolerant of the released RAID naming without
    weakening uniqueness checks in :func:`score_raid_jsonl`.
    """

    explicit = row.get("row_key")
    if explicit is not None:
        if isinstance(explicit, (list, tuple)):
            return tuple(str(value) for value in explicit)
        return (str(explicit),)
    if {"source_id", "label", "attack"}.issubset(row):
        # Keep score packs directly joinable to RAID/raid_data.py outputs and
        # to evaluation maps that use the preparation key function.
        return tuple(str(value) for value in prepared_raid_row_key(row))
    source_id = _first_present(row, ("source_id", "sample_id"), None)
    if source_id is None:
        raise ValueError("RAID row requires source_id, sample_id, or row_key")
    role = _first_present(row, ("record_type", "label", "origin"), "")
    generation_id = _first_present(
        row,
        ("selected_generation_id", "base_generation_id", "generation_id"),
        "",
    )
    condition = _first_present(
        row, ("condition", "attack", "attack_name", "raid_condition"), ""
    )
    return tuple(str(value) for value in (source_id, role, generation_id, condition))


@dataclass(frozen=True)
class EncodedWindow:
    """One output-only, right-truncated causal-scoring window."""

    input_ids: list[int]
    scored_token_ids: list[int]
    full_scored_token_ids: list[int]
    original_num_input_tokens: int
    boundary_token_added: bool
    truncated_token_count: int
    policy: str
    max_tokens: int

    @property
    def start(self) -> int:
        return 0

    @property
    def num_scored_tokens(self) -> int:
        return len(self.scored_token_ids)

    def adjustment_metadata(self) -> dict[str, Any]:
        return {
            "original_num_input_tokens": self.original_num_input_tokens,
            "boundary_token_added": self.boundary_token_added,
            "truncated_token_count": self.truncated_token_count,
            "num_scored_tokens": self.num_scored_tokens,
        }


def encode_scored_window(
    tokenizer: Any,
    row: dict[str, Any],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    policy: str = "raid_output_only_512",
) -> EncodedWindow:
    """Encode only ``row['text']`` with tokenizer-native special-token behavior.

    The official Binoculars setup right-truncates its tokenizer input to 512
    tokens.  RAID uses the same default window for Falcon so clean/attack edit
    rates and all seven detector scores refer to the same visible text.  Causal
    shifting means the first input token is context and is not a scored target.
    A BOS (or EOS) context token is inserted only for an otherwise unscorable
    one-token document, and that intervention is recorded.
    """

    if policy not in {"raid_output_only_512", "binoculars_official_output_only_512"}:
        raise ValueError(f"unknown RAID scoring context policy: {policy}")
    limit = int(max_tokens)
    if limit < 2:
        raise ValueError("RAID scoring max_tokens must be at least 2")
    if "text" not in row:
        raise ValueError("RAID scoring row is missing text")
    # Tokenize the complete document for the truncation/edit audit, but suppress
    # Transformers' misleading model-limit warning: only the right-truncated
    # window below is ever passed to the model.
    token_ids = [
        int(value) for value in tokenizer.encode(row["text"], verbose=False)
    ]
    original_count = len(token_ids)
    if original_count == 0:
        raise ValueError(f"zero output tokens for RAID row {raid_row_key(row)!r}")
    boundary_added = False
    if len(token_ids) == 1:
        boundary_id = getattr(tokenizer, "bos_token_id", None)
        if boundary_id is None:
            boundary_id = getattr(tokenizer, "eos_token_id", None)
        if boundary_id is None:
            raise ValueError(
                f"one output token and no BOS/EOS boundary for {raid_row_key(row)!r}"
            )
        token_ids = [int(boundary_id), token_ids[0]]
        boundary_added = True
    full_scored_ids = token_ids[1:]
    token_ids = token_ids[:limit]
    scored_ids = token_ids[1:]
    return EncodedWindow(
        input_ids=token_ids,
        scored_token_ids=scored_ids,
        full_scored_token_ids=full_scored_ids,
        original_num_input_tokens=original_count,
        boundary_token_added=boundary_added,
        truncated_token_count=max(len(full_scored_ids) - len(scored_ids), 0),
        policy=policy,
        max_tokens=limit,
    )


def _resolved_model_revision(model: Any, requested: str | None, name: str) -> str:
    revision = getattr(getattr(model, "config", None), "_commit_hash", None) or requested
    if not revision:
        raise ValueError(f"unable to resolve {name} model revision")
    return str(revision)


def _resolved_tokenizer_revision(
    tokenizer: Any, requested: str | None, fallback: str, name: str
) -> str:
    revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    revision = revision or requested or fallback
    if not revision:
        raise ValueError(f"unable to resolve {name} tokenizer revision")
    return str(revision)


def _actual_model_dtype(model: Any, fallback: str) -> str:
    try:
        value = str(next(model.parameters()).dtype)
        return value.removeprefix("torch.")
    except (AttributeError, StopIteration, TypeError):
        return str(fallback)


def _actual_device(model: Any, fallback: Any) -> str:
    try:
        return str(next(model.parameters()).device)
    except (AttributeError, StopIteration, TypeError):
        return str(fallback)


def _validate_common_score_row(
    row: dict[str, Any], *, feature_schema: str, context_policy: str, max_tokens: int
) -> None:
    if row.get("score_row_schema") != SCORE_ROW_SCHEMA:
        raise ValueError("existing RAID scores use a different row schema; use a new run ID")
    if row.get("scoring_feature_schema") != feature_schema:
        raise ValueError(
            "existing RAID scores use a different feature schema; use a new run ID"
        )
    if row.get("scoring_context_policy") != context_policy:
        raise ValueError(
            "existing RAID scores use a different context policy; use a new run ID"
        )
    if row.get("scoring_max_tokens") != max_tokens:
        raise ValueError(
            "existing RAID scores use a different token limit; use a new run ID"
        )
    if tuple(str(value) for value in row.get("score_row_key", [])) != raid_row_key(row):
        raise ValueError("existing RAID score row has an invalid stable key")
    token_features = row.get("token_features", {})
    scored_ids = token_features.get("scored_token_ids")
    if not isinstance(scored_ids, list) or len(scored_ids) != row.get("num_scored_tokens"):
        raise ValueError("existing RAID score row has inconsistent scored token IDs")
    full_ids = token_features.get("full_scored_token_ids")
    if full_ids is not None:
        if not isinstance(full_ids, list) or scored_ids != full_ids[: len(scored_ids)]:
            raise ValueError("existing RAID score row has inconsistent full token IDs")
        if len(full_ids) - len(scored_ids) != int(row["truncated_token_count"]):
            raise ValueError("existing RAID score row has inconsistent truncation metadata")
    if int(row["num_scored_tokens"]) < 1 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in scored_ids
    ):
        raise ValueError("existing RAID score row has invalid scored token IDs")
    if not isinstance(row.get("boundary_token_added"), bool):
        raise ValueError("existing RAID score row has invalid boundary metadata")
    if int(row.get("original_num_input_tokens", 0)) < 1 or int(
        row.get("truncated_token_count", -1)
    ) < 0:
        raise ValueError("existing RAID score row has invalid truncation metadata")


def _finite_vector(values: Any, expected_length: int) -> bool:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return False
    return array.ndim == 1 and len(array) == expected_length and bool(np.isfinite(array).all())


def _finite_doc_scores(row: dict[str, Any], required: set[str]) -> bool:
    scores = row.get("doc_scores")
    if not isinstance(scores, dict) or not required.issubset(scores):
        return False
    try:
        return all(math.isfinite(float(scores[name])) for name in required)
    except (TypeError, ValueError):
        return False


class RAIDFalconScorer:
    """Falcon-7B feature pack for all six single-model RAID detectors."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
        device: Any | None = None,
    ) -> None:
        self.config = dict(config)
        self.model_id = str(config.get("model_id", config.get("id", FALCON_MODEL_ID)))
        if self.model_id != FALCON_MODEL_ID:
            raise ValueError(f"RAID Falcon scorer must use {FALCON_MODEL_ID}")
        self.requested_revision = config.get("revision")
        self.requested_tokenizer_revision = config.get("tokenizer_revision")
        self.context_policy = str(config.get("context_policy", "raid_output_only_512"))
        self.max_tokens = int(config.get("max_tokens", DEFAULT_MAX_TOKENS))
        self.requested_dtype = str(config.get("dtype", "bfloat16"))
        if (tokenizer is None) != (model is None):
            raise ValueError("inject both tokenizer and model, or neither")
        if tokenizer is None:
            tokenizer, model, loaded_device = _load_model(
                self.model_id,
                self.requested_revision,
                self.requested_tokenizer_revision,
                self.requested_dtype,
                str(config.get("device", "cuda:0")),
                config.get("device_map"),
                tokenizer_use_fast=config.get("tokenizer_use_fast", True),
                trust_remote_code=bool(config.get("trust_remote_code", False)),
            )
            device = loaded_device
        self.tokenizer = tokenizer
        self.model = model
        self.device = device if device is not None else _actual_device(model, "cpu")
        self.resolved_revision = _resolved_model_revision(
            model, self.requested_revision, "Falcon"
        )
        self.resolved_tokenizer_revision = _resolved_tokenizer_revision(
            tokenizer,
            self.requested_tokenizer_revision,
            self.resolved_revision,
            "Falcon",
        )
        self.resolved_dtype = _actual_model_dtype(model, self.requested_dtype)
        self.resolved_device = _actual_device(model, self.device)

    @property
    def feature_schema(self) -> str:
        return FALCON_FEATURE_SCHEMA

    def encode(self, row: dict[str, Any]) -> EncodedWindow:
        return encode_scored_window(
            self.tokenizer, row, self.max_tokens, self.context_policy
        )

    def token_count(self, row: dict[str, Any]) -> int:
        return len(self.encode(row).input_ids)

    def adjustment_info(self, row: dict[str, Any]) -> dict[str, Any]:
        return self.encode(row).adjustment_metadata()

    def validate_existing_row(self, row: dict[str, Any]) -> None:
        _validate_common_score_row(
            row,
            feature_schema=self.feature_schema,
            context_policy=self.context_policy,
            max_tokens=self.max_tokens,
        )
        expected = {
            "scoring_model": self.model_id,
            "scoring_model_revision": self.resolved_revision,
            "scoring_tokenizer_revision": self.resolved_tokenizer_revision,
            "scoring_dtype": self.resolved_dtype,
            "scoring_device": self.resolved_device,
        }
        if any(row.get(name) != value for name, value in expected.items()):
            raise ValueError(
                "existing RAID Falcon scores use different model provenance; "
                "use a new run ID"
            )
        required = {"logp", "nll", "rank", "log_rank", "entropy", "entropy_gap"}
        features = row["token_features"]
        if not required.issubset(features):
            raise ValueError("existing RAID Falcon score row is missing token features")
        if "full_scored_token_ids" not in features:
            raise ValueError("existing RAID Falcon score row is missing full token IDs")
        length = int(row["num_scored_tokens"])
        if any(not _finite_vector(features[name], length) for name in required):
            raise ValueError("existing RAID Falcon score row has inconsistent feature lengths")
        if not _finite_doc_scores(
            row,
            {"log_likelihood", "rank", "log_rank", "lrr", "entropy", "entropy_gap"},
        ):
            raise ValueError("existing RAID Falcon score row has invalid document scores")

    def score_batch(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        torch = _torch()
        encoded = [self.encode(row) for row in rows]
        padded = [(item.input_ids, 0, item.num_scored_tokens, {}) for item in encoded]
        input_ids, attention = _padded_batch(self.tokenizer, padded, self.device)
        with torch.inference_mode():
            logits_batch = self.model(
                input_ids=input_ids, attention_mask=attention
            ).logits[:, :-1, :]
        outputs: list[dict[str, Any]] = []
        for index, (row, window) in enumerate(zip(rows, encoded)):
            length = window.num_scored_tokens
            logits = logits_batch[index, :length]
            targets = input_ids[index, 1 : 1 + length]
            tensors = exact_token_features(
                logits,
                targets,
                int(self.config.get("vocab_chunk_size", 4096)),
                saved_top_k=0,
            )
            features = {
                name: values.detach().cpu().tolist()
                for name, values in tensors.items()
            }
            logp = np.asarray(features["logp"], dtype=float)
            entropy = np.asarray(features["entropy"], dtype=float)
            features["nll"] = (-logp).tolist()
            features["entropy_gap"] = (-logp - entropy).tolist()
            features["scored_token_ids"] = list(window.scored_token_ids)
            features["full_scored_token_ids"] = list(window.full_scored_token_ids)
            outputs.append(
                {
                    **row,
                    "score_row_key": list(raid_row_key(row)),
                    "score_row_schema": SCORE_ROW_SCHEMA,
                    "scoring_feature_schema": self.feature_schema,
                    "scoring_model": self.model_id,
                    "scoring_model_revision": self.resolved_revision,
                    "scoring_tokenizer_revision": self.resolved_tokenizer_revision,
                    "scoring_dtype": self.resolved_dtype,
                    "scoring_device": self.resolved_device,
                    "scoring_context_policy": self.context_policy,
                    "scoring_max_tokens": self.max_tokens,
                    **window.adjustment_metadata(),
                    "token_features": features,
                    "doc_scores": single_model_doc_scores(features),
                }
            )
        del logits_batch
        return outputs


class RAIDBinocularsScorer:
    """Legacy exponential-gap scorer, retained for historical reproducibility.

    The separate RAIDBinocularsOriginScorer implements the published ratio.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        tokenizer: Any | None = None,
        performer: Any | None = None,
        observer: Any | None = None,
        performer_device: Any | None = None,
        observer_device: Any | None = None,
    ) -> None:
        self.config = dict(config)
        self.observer_id = str(config.get("observer", BINOCULARS_OBSERVER_ID))
        self.performer_id = str(config.get("performer", BINOCULARS_PERFORMER_ID))
        if self.observer_id != BINOCULARS_OBSERVER_ID:
            raise ValueError(f"RAID Binoculars observer must use {BINOCULARS_OBSERVER_ID}")
        if self.performer_id != BINOCULARS_PERFORMER_ID:
            raise ValueError(
                f"RAID Binoculars performer must use {BINOCULARS_PERFORMER_ID}"
            )
        self.observer_requested_revision = config.get("observer_revision")
        self.performer_requested_revision = config.get("performer_revision")
        self.performer_tokenizer_requested_revision = config.get(
            "performer_tokenizer_revision", self.performer_requested_revision
        )
        self.tokenizer_requested_revision = config.get(
            "tokenizer_revision", self.observer_requested_revision
        )
        self.requested_dtype = str(config.get("dtype", "bfloat16"))
        self.context_policy = str(
            config.get("context_policy", "binoculars_official_output_only_512")
        )
        self.max_tokens = int(config.get("max_tokens", DEFAULT_MAX_TOKENS))
        injected = (tokenizer, performer, observer)
        if any(value is None for value in injected) and not all(
            value is None for value in injected
        ):
            raise ValueError("inject tokenizer, performer, and observer together")
        if tokenizer is None:
            # Upstream Binoculars tokenizes with the observer/base tokenizer.
            tokenizer, observer, loaded_observer_device = _load_model(
                self.observer_id,
                self.observer_requested_revision,
                self.tokenizer_requested_revision,
                self.requested_dtype,
                str(config.get("observer_device", "cuda:0")),
                None,
                tokenizer_use_fast=config.get("tokenizer_use_fast", True),
                trust_remote_code=bool(config.get("trust_remote_code", False)),
            )
            performer_tokenizer, performer, loaded_performer_device = _load_model(
                self.performer_id,
                self.performer_requested_revision,
                self.performer_tokenizer_requested_revision,
                self.requested_dtype,
                str(config.get("performer_device", "cuda:1")),
                None,
                tokenizer_use_fast=config.get("tokenizer_use_fast", True),
                trust_remote_code=bool(config.get("trust_remote_code", False)),
            )
            _compatible_tokenizers(tokenizer, performer_tokenizer)
            observer_vocab = (
                tokenizer.get_vocab()
                if hasattr(tokenizer, "get_vocab")
                else getattr(tokenizer, "vocab", None)
            )
            performer_vocab = (
                performer_tokenizer.get_vocab()
                if hasattr(performer_tokenizer, "get_vocab")
                else getattr(performer_tokenizer, "vocab", None)
            )
            if observer_vocab is None or observer_vocab != performer_vocab:
                raise ValueError(
                    "Binoculars performer and observer vocabularies are not identical"
                )
            observer_device = loaded_observer_device
            performer_device = loaded_performer_device
        else:
            # Injected tests use one already-verified compatible tokenizer.
            performer_tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.performer = performer
        self.observer = observer
        self.observer_device = (
            observer_device if observer_device is not None else _actual_device(observer, "cpu")
        )
        self.performer_device = (
            performer_device if performer_device is not None else _actual_device(performer, "cpu")
        )
        self.observer_resolved_revision = _resolved_model_revision(
            observer, self.observer_requested_revision, "Binoculars observer"
        )
        self.performer_resolved_revision = _resolved_model_revision(
            performer, self.performer_requested_revision, "Binoculars performer"
        )
        self.resolved_tokenizer_revision = _resolved_tokenizer_revision(
            tokenizer,
            self.tokenizer_requested_revision,
            self.observer_resolved_revision,
            "Binoculars observer",
        )
        self.performer_resolved_tokenizer_revision = _resolved_tokenizer_revision(
            performer_tokenizer,
            self.performer_tokenizer_requested_revision,
            self.performer_resolved_revision,
            "Binoculars performer",
        )
        self.observer_dtype = _actual_model_dtype(observer, self.requested_dtype)
        self.performer_dtype = _actual_model_dtype(performer, self.requested_dtype)
        self.resolved_observer_device = _actual_device(observer, self.observer_device)
        self.resolved_performer_device = _actual_device(performer, self.performer_device)

    @property
    def feature_schema(self) -> str:
        return BINOCULARS_FEATURE_SCHEMA

    def encode(self, row: dict[str, Any]) -> EncodedWindow:
        return encode_scored_window(
            self.tokenizer, row, self.max_tokens, self.context_policy
        )

    def token_count(self, row: dict[str, Any]) -> int:
        return len(self.encode(row).input_ids)

    def adjustment_info(self, row: dict[str, Any]) -> dict[str, Any]:
        return self.encode(row).adjustment_metadata()

    def validate_existing_row(self, row: dict[str, Any]) -> None:
        _validate_common_score_row(
            row,
            feature_schema=self.feature_schema,
            context_policy=self.context_policy,
            max_tokens=self.max_tokens,
        )
        expected = {
            "binoculars_observer_model": self.observer_id,
            "binoculars_observer_revision": self.observer_resolved_revision,
            "binoculars_performer_model": self.performer_id,
            "binoculars_performer_revision": self.performer_resolved_revision,
            "binoculars_tokenizer_revision": self.resolved_tokenizer_revision,
            "binoculars_performer_tokenizer_revision": (
                self.performer_resolved_tokenizer_revision
            ),
            "binoculars_observer_dtype": self.observer_dtype,
            "binoculars_performer_dtype": self.performer_dtype,
            "binoculars_observer_device": self.resolved_observer_device,
            "binoculars_performer_device": self.resolved_performer_device,
        }
        if any(row.get(name) != value for name, value in expected.items()):
            raise ValueError(
                "existing RAID Binoculars scores use different model provenance; "
                "use a new run ID"
            )
        features = row["token_features"]
        required = {
            "performer_nll",
            "observer_to_performer_cross_entropy",
            "local_gap",
        }
        if not required.issubset(features):
            raise ValueError("existing RAID Binoculars row is missing token features")
        length = int(row["num_scored_tokens"])
        if any(not _finite_vector(features[name], length) for name in required):
            raise ValueError(
                "existing RAID Binoculars score row has inconsistent feature lengths"
            )
        if not _finite_doc_scores(
            row,
            {
                "binoculars",
                "binoculars_numerator_perplexity",
                "binoculars_denominator_cross_perplexity",
            },
        ):
            raise ValueError("existing RAID Binoculars row has invalid document scores")

    def score_batch(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        torch = _torch()
        encoded = [self.encode(row) for row in rows]
        padded = [(item.input_ids, 0, item.num_scored_tokens, {}) for item in encoded]
        observer_ids, observer_attention = _padded_batch(
            self.tokenizer, padded, self.observer_device
        )
        performer_ids = observer_ids.to(self.performer_device)
        performer_attention = observer_attention.to(self.performer_device)
        with torch.inference_mode():
            observer_batch = self.observer(
                input_ids=observer_ids, attention_mask=observer_attention
            ).logits[:, :-1, :]
            performer_batch = self.performer(
                input_ids=performer_ids, attention_mask=performer_attention
            ).logits[:, :-1, :]
        outputs: list[dict[str, Any]] = []
        for index, (row, window) in enumerate(zip(rows, encoded)):
            length = window.num_scored_tokens
            observer_logits = observer_batch[index, :length].to(
                self.performer_device
            ).float()
            performer_logits = performer_batch[index, :length].float()
            targets = performer_ids[index, 1 : 1 + length]
            target_logits = performer_logits.gather(-1, targets[:, None]).squeeze(-1)
            performer_nll = -(
                target_logits - torch.logsumexp(performer_logits, dim=-1)
            )
            cross_entropy = exact_observer_to_performer_cross_entropy(
                observer_logits,
                performer_logits,
                int(self.config.get("vocab_chunk_size", 4096)),
            )
            local_gap = performer_nll - cross_entropy
            mean_nll = float(performer_nll.mean().item())
            mean_xent = float(cross_entropy.mean().item())
            score = binoculars_score(mean_nll, mean_xent)
            features = {
                "performer_nll": performer_nll.detach().cpu().tolist(),
                "observer_to_performer_cross_entropy": cross_entropy.detach()
                .cpu()
                .tolist(),
                "local_gap": local_gap.detach().cpu().tolist(),
                "scored_token_ids": list(window.scored_token_ids),
            }
            outputs.append(
                {
                    **row,
                    "score_row_key": list(raid_row_key(row)),
                    "score_row_schema": SCORE_ROW_SCHEMA,
                    "scoring_feature_schema": self.feature_schema,
                    "binoculars_observer_model": self.observer_id,
                    "binoculars_observer_revision": self.observer_resolved_revision,
                    "binoculars_performer_model": self.performer_id,
                    "binoculars_performer_revision": self.performer_resolved_revision,
                    "binoculars_tokenizer_revision": self.resolved_tokenizer_revision,
                    "binoculars_performer_tokenizer_revision": (
                        self.performer_resolved_tokenizer_revision
                    ),
                    "binoculars_observer_dtype": self.observer_dtype,
                    "binoculars_performer_dtype": self.performer_dtype,
                    "binoculars_observer_device": self.resolved_observer_device,
                    "binoculars_performer_device": self.resolved_performer_device,
                    "binoculars_formula": (
                        "exp(mean_performer_nll - "
                        "mean_observer_to_performer_cross_entropy)"
                    ),
                    "scoring_context_policy": self.context_policy,
                    "scoring_max_tokens": self.max_tokens,
                    **window.adjustment_metadata(),
                    "token_features": features,
                    "doc_scores": {
                        "binoculars": score,
                        "binoculars_numerator_perplexity": math.exp(mean_nll),
                        "binoculars_denominator_cross_perplexity": math.exp(mean_xent),
                    },
                }
            )
        del observer_batch, performer_batch
        return outputs


def _input_keys(path: str | Path) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for row in iter_jsonl(path):
        key = raid_row_key(row)
        if key in keys:
            raise ValueError(f"duplicate prepared RAID row key {key!r} in {path}")
        keys.add(key)
    return keys


def score_raid_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    scorer: Any,
    scoring_config: dict[str, Any],
) -> int:
    """Stream, microbatch, and append-safe resume a RAID score pack."""

    input_keys = _input_keys(input_path)
    output = Path(output_path)
    existing = completed_keys(output, raid_row_key)
    unexpected = existing - input_keys
    if unexpected:
        raise ValueError(f"RAID score output contains keys absent from input: {sorted(unexpected)[:3]}")
    if output.exists():
        for row in iter_jsonl(output):
            scorer.validate_existing_row(row)
    pending: Iterable[dict[str, Any]] = (
        row for row in iter_jsonl(input_path) if raid_row_key(row) not in existing
    )
    progress = Throughput(len(input_keys), "RAID scoring")
    progress.examples = len(existing)
    written = len(existing)
    with AppendSafeJsonlWriter(
        output, checkpoint_interval=int(scoring_config.get("checkpoint_interval", 100))
    ) as writer:
        for batch in length_bucketed(
            pending,
            scorer.token_count,
            int(scoring_config.get("batch_size", 8)),
            int(scoring_config.get("length_bucket_width", 64)),
        ):
            microbatch_size = max(int(scoring_config.get("microbatch_size", 1)), 1)
            for start in range(0, len(batch), microbatch_size):
                part = batch[start : start + microbatch_size]
                scored = scorer.score_batch(part)
                if len(scored) != len(part):
                    raise ValueError("RAID scorer returned a different number of rows")
                for prepared, result in zip(part, scored):
                    if raid_row_key(prepared) != raid_row_key(result):
                        raise ValueError("RAID scorer changed a prepared row key")
                    scorer.validate_existing_row(result)
                    writer.write(result)
                    written += 1
                progress.update(
                    len(scored),
                    sum(int(row["num_scored_tokens"]) for row in scored),
                    peak_gpu_memory_bytes(),
                )
    if written != len(input_keys):
        raise ValueError(f"RAID scoring produced {written} rows; expected {len(input_keys)}")
    return written
