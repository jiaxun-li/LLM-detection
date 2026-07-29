"""Length-bucketed Transformers and optional vLLM generation backends."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence


def dtype_from_name(torch: Any, name: str) -> Any:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def length_bucketed(
    items: Iterable[Any],
    length_fn: Any,
    batch_size: int,
    width: int,
) -> Iterator[list[Any]]:
    buckets: dict[int, list[Any]] = defaultdict(list)
    for item in items:
        length = int(length_fn(item))
        bucket = length // max(width, 1)
        buckets[bucket].append(item)
        if len(buckets[bucket]) >= batch_size:
            yield buckets.pop(bucket)
    for bucket in sorted(buckets):
        if buckets[bucket]:
            yield buckets[bucket]


@dataclass
class GenerationRequest:
    key: str
    prompt: str
    seed: int


class TransformersBackend:
    """Dependable batched generation backend."""

    def __init__(self, model_id: str, revision: str | None, tokenizer_revision: str | None, config: dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers generation requires torch and transformers; "
                "install requirements.txt on the GPU host"
            ) from exc
        self.torch = torch
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=tokenizer_revision,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        kwargs: dict[str, Any] = {
            "revision": revision,
            "torch_dtype": dtype_from_name(torch, config["dtype"]),
        }
        if config.get("device_map") is not None:
            kwargs["device_map"] = config["device_map"]
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if config.get("device_map") is None:
            self.model.to(config["device"])
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self._rng_states: dict[int, tuple[Any, list[Any] | None]] = {}
        self.resolved_model_revision = (
            getattr(self.model.config, "_commit_hash", None) or revision
        )
        self.resolved_tokenizer_revision = (
            self.tokenizer.init_kwargs.get("_commit_hash")
            or tokenizer_revision
            or self.resolved_model_revision
        )

    def prompt_token_count(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def generate(self, requests: Sequence[GenerationRequest]) -> dict[str, list[int]]:
        results: dict[str, list[int]] = {}
        # Group by seed so each configured generation stream is explicit.
        by_seed: dict[int, list[GenerationRequest]] = defaultdict(list)
        for request in requests:
            by_seed[request.seed].append(request)
        for seed in sorted(by_seed):
            seed_requests = by_seed[seed]
            if seed in self._rng_states:
                cpu_state, cuda_states = self._rng_states[seed]
                self.torch.set_rng_state(cpu_state)
                if cuda_states is not None:
                    self.torch.cuda.set_rng_state_all(cuda_states)
            else:
                self.torch.manual_seed(seed)
                if self.torch.cuda.is_available():
                    self.torch.cuda.manual_seed_all(seed)
            batches = length_bucketed(
                seed_requests,
                lambda item: self.prompt_token_count(item.prompt),
                int(self.config["batch_size"]),
                int(self.config["length_bucket_width"]),
            )
            for batch in batches:
                encoded = self.tokenizer(
                    [item.prompt for item in batch],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(self.device)
                input_width = encoded.input_ids.shape[1]
                with self.torch.inference_mode():
                    outputs = self.model.generate(
                        **encoded,
                        max_new_tokens=int(self.config["max_new_tokens"]),
                        min_new_tokens=int(self.config.get("min_new_tokens", 0)),
                        do_sample=True,
                        temperature=float(self.config["temperature"]),
                        top_p=float(self.config["top_p"]),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                for item, output in zip(batch, outputs):
                    continuation = output[input_width:].tolist()
                    if self.tokenizer.eos_token_id in continuation:
                        continuation = continuation[: continuation.index(self.tokenizer.eos_token_id)]
                    results[item.key] = continuation[: int(self.config["continuation_tokens"])]
            self._rng_states[seed] = (
                self.torch.get_rng_state(),
                self.torch.cuda.get_rng_state_all()
                if self.torch.cuda.is_available()
                else None,
            )
        return results

    def candidate_nlls(self, prompt: str, candidate_spans: Sequence[Sequence[int]]) -> list[float]:
        """Score each candidate once, in length-bucketed batches."""
        keys = [str(index) for index in range(len(candidate_spans))]
        scored = self.candidate_nll_batch(
            [
                (key, prompt, span)
                for key, span in zip(keys, candidate_spans)
            ]
        )
        return [scored[key] for key in keys]

    def candidate_nll_batch(
        self,
        requests: Sequence[tuple[str, str, Sequence[int]]],
    ) -> dict[str, float]:
        """Length-bucket tail candidates across source examples."""
        torch = self.torch
        encoded_requests = [
            (
                key,
                self.tokenizer.encode(prompt, add_special_tokens=False),
                list(span),
            )
            for key, prompt, span in requests
        ]
        result: dict[str, float] = {}
        for batch in length_bucketed(
            encoded_requests,
            lambda item: len(item[1]) + len(item[2]),
            int(self.config["batch_size"]),
            int(self.config["length_bucket_width"]),
        ):
            sequences = [prompt_ids + span for _, prompt_ids, span in batch]
            max_length = max(map(len, sequences))
            pad = self.tokenizer.pad_token_id
            input_ids = torch.full((len(batch), max_length), pad, dtype=torch.long, device=self.device)
            attention = torch.zeros_like(input_ids)
            for row_index, sequence in enumerate(sequences):
                input_ids[row_index, : len(sequence)] = torch.tensor(sequence, device=self.device)
                attention[row_index, : len(sequence)] = 1
            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=attention).logits[:, :-1, :].float()
            for row_index, (key, prompt_ids, span) in enumerate(batch):
                start = max(len(prompt_ids) - 1, 0)
                labels = input_ids[row_index, 1 : 1 + logits.shape[1]]
                selected = logits[row_index, start : start + len(span)]
                selected_labels = labels[start : start + len(span)]
                logp = selected.gather(-1, selected_labels[:, None]).squeeze(-1)
                logp = logp - torch.logsumexp(selected, dim=-1)
                result[key] = float(-logp.mean().item())
        return result


class VLLMBackend:
    """Optional high-throughput generation backend; imported only when selected."""

    def __init__(self, model_id: str, revision: str | None, tokenizer_revision: str | None, config: dict[str, Any]):
        try:
            from transformers import AutoTokenizer
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("generation.backend=vllm requires the optional vllm package") from exc
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tokenizer_revision)
        self.llm = LLM(
            model=model_id,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            tensor_parallel_size=int(config.get("tensor_parallel_size", 1)),
            dtype=config["dtype"],
        )
        model_config = self.llm.llm_engine.model_config
        self.resolved_model_revision = (
            getattr(getattr(model_config, "hf_config", None), "_commit_hash", None)
            or getattr(model_config, "revision", None)
            or revision
        )
        self.resolved_tokenizer_revision = tokenizer_revision or self.resolved_model_revision

    def prompt_token_count(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def generate(self, requests: Sequence[GenerationRequest]) -> dict[str, list[int]]:
        from vllm import SamplingParams

        results: dict[str, list[int]] = {}
        # Per-request SamplingParams makes the corruption-independent seed explicit.
        for batch in length_bucketed(
            requests,
            lambda item: self.prompt_token_count(item.prompt),
            int(self.config["batch_size"]),
            int(self.config["length_bucket_width"]),
        ):
            params = [
                SamplingParams(
                    temperature=float(self.config["temperature"]),
                    top_p=float(self.config["top_p"]),
                    max_tokens=int(self.config["max_new_tokens"]),
                    min_tokens=int(self.config.get("min_new_tokens", 0)),
                    seed=item.seed,
                )
                for item in batch
            ]
            outputs = self.llm.generate([item.prompt for item in batch], params, use_tqdm=False)
            for item, output in zip(batch, outputs):
                results[item.key] = list(output.outputs[0].token_ids)[
                    : int(self.config["continuation_tokens"])
                ]
        return results

    def candidate_nlls(self, prompt: str, candidate_spans: Sequence[Sequence[int]]) -> list[float]:
        raise RuntimeError(
            "Tail white-box scoring uses the target Transformers scorer even when "
            "vLLM is selected for generation"
        )


def make_generation_backend(
    model_id: str,
    revision: str | None,
    tokenizer_revision: str | None,
    config: dict[str, Any],
) -> TransformersBackend | VLLMBackend:
    if config["backend"] == "vllm":
        return VLLMBackend(model_id, revision, tokenizer_revision, config)
    return TransformersBackend(model_id, revision, tokenizer_revision, config)
