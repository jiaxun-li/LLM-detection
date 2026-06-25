"""Score real-text examples with token-level detector features.

Inputs are JSONL files produced by `prepare_real_contamination.py`.
Outputs are JSONL files containing token-level arrays for later clipping and
evaluation.

Example:

  python score_real_text.py \
    --input real_data/xsum/Qwen__Qwen2.5-0.5B/clean.jsonl \
    --output real_data/xsum/Qwen__Qwen2.5-0.5B/clean_scores.jsonl \
    --observer-model Qwen/Qwen2.5-0.5B \
    --reference-model Qwen/Qwen2.5-1.5B

To match the upstream Binoculars implementation defaults:

  python score_real_text.py \
    --input real_data/xsum/Qwen__Qwen2.5-0.5B/clean.jsonl \
    --output real_data/xsum/Qwen__Qwen2.5-0.5B/clean_scores.jsonl \
    --binoculars-default-models

The observer model A provides:

  - logp_a
  - rank_a
  - logrank_a
  - entropy_a

If a reference model B is provided with the same tokenizer, the script also
computes:

  - logp_b
  - Binoculars-style cross entropy: H(p_B, p_A)

For `--binoculars-default-models`, A is tiiuae/falcon-7b-instruct
and B is tiiuae/falcon-7b, so the Binoculars score is the upstream
performer perplexity divided by observer-to-performer cross entropy.
By default this uses the native Transformers Falcon implementation; add
`--trust-remote-code` only if a model explicitly requires it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


EPS = 1e-12
BINOCULARS_OBSERVER_MODEL = "tiiuae/falcon-7b"
BINOCULARS_PERFORMER_MODEL = "tiiuae/falcon-7b-instruct"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model_and_tokenizer(model_id: str, device: torch.device, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=trust_remote_code,
    ).to(device)
    module_name = type(model).__module__
    if model_id.startswith("tiiuae/falcon") and "transformers_modules" in module_name:
        raise RuntimeError(
            "Falcon loaded from cached Hugging Face remote code instead of the "
            "native Transformers implementation. Remove the cached module with "
            "`rm -rf ~/.cache/huggingface/modules/transformers_modules/tiiuae` "
            "and upgrade Transformers with `pip install -U transformers accelerate safetensors`."
        )
    print(
        f"loaded model={model_id} class={type(model).__name__} module={module_name} "
        f"device={device} trust_remote_code={trust_remote_code}",
        flush=True,
    )
    model.eval()
    return tokenizer, model


def check_compatible_tokenizers(observer_tokenizer, reference_tokenizer) -> None:
    """Require same token ids for token-level likelihood ratios/Binoculars."""
    probes = [
        "The quick brown fox jumps over the lazy dog.",
        "A model should tokenize this sentence identically.",
        "Numbers: 10, 20, 30. Symbols: !?;:",
    ]
    for probe in probes:
        observer_ids = observer_tokenizer(probe, add_special_tokens=False).input_ids
        reference_ids = reference_tokenizer(probe, add_special_tokens=False).input_ids
        if observer_ids != reference_ids:
            raise ValueError(
                "Observer/reference tokenizers are not compatible. "
                "Use models from the same tokenizer family for likelihood ratio "
                "and Binoculars-style cross entropy."
            )


def continuation_token_slice(tokenizer, prompt: str, text: str) -> tuple[list[int], int, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    text_ids = tokenizer(text, add_special_tokens=False).input_ids
    full_ids = prompt_ids + text_ids
    if len(prompt_ids) == 0:
        start = 0
    else:
        # Logits at position j predict token j + 1. The first continuation
        # token is predicted by the last prompt token.
        start = len(prompt_ids) - 1
    end = start + len(text_ids)
    return full_ids, start, end


def model_logits(model, input_ids: list[int], device: torch.device) -> torch.Tensor:
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids).logits[:, :-1, :].squeeze(0)
    return logits


def score_row(
    row: dict[str, Any],
    observer_tokenizer,
    observer_model,
    reference_model,
    device: torch.device,
    reference_device: torch.device,
    store_tokens: bool,
    observer_model_id: str,
    reference_model_id: Optional[str],
) -> dict[str, Any]:
    prompt = row["prompt"]
    text = row["text"]
    input_ids, start, end = continuation_token_slice(observer_tokenizer, prompt, text)
    if end <= start:
        raise ValueError("empty continuation after tokenization")

    target_ids = torch.tensor(input_ids[1:], dtype=torch.long, device=device)[start:end]
    observer_logits = model_logits(observer_model, input_ids, device)[start:end]
    observer_log_probs = F.log_softmax(observer_logits.float(), dim=-1)
    observer_probs = observer_log_probs.exp()

    target_logp_a = observer_log_probs.gather(-1, target_ids[:, None]).squeeze(-1)
    target_logits = observer_logits.gather(-1, target_ids[:, None]).squeeze(-1)
    rank_a = (observer_logits > target_logits[:, None]).sum(dim=-1) + 1
    entropy_a = -(observer_probs * observer_log_probs).sum(dim=-1)

    token_features: dict[str, Any] = {
        "logp_a": target_logp_a.cpu().tolist(),
        "rank_a": rank_a.cpu().tolist(),
        "logrank_a": torch.log(rank_a.float()).cpu().tolist(),
        "entropy_a": entropy_a.cpu().tolist(),
    }

    if reference_model is not None:
        reference_logits = model_logits(reference_model, input_ids, reference_device)[start:end].to(device)
        reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
        reference_probs = reference_log_probs.exp()
        target_logp_b = reference_log_probs.gather(-1, target_ids[:, None]).squeeze(-1)
        cross_entropy_ba = -(reference_probs * observer_log_probs).sum(dim=-1)

        token_features["logp_b"] = target_logp_b.cpu().tolist()
        token_features["cross_entropy_ba"] = cross_entropy_ba.cpu().tolist()

    nll_a = -target_logp_a
    doc_scores = {
        "log_likelihood": float(target_logp_a.mean().item()),
        "rank": float(rank_a.float().mean().item()),
        "log_rank": float(torch.log(rank_a.float()).mean().item()),
        "entropy": float(entropy_a.mean().item()),
        "entropy_gap": float((nll_a - entropy_a).mean().item()),
    }
    if reference_model is not None:
        cross_entropy_ba = torch.tensor(token_features["cross_entropy_ba"], device=device)
        doc_scores["binoculars"] = float(nll_a.mean().item() / (cross_entropy_ba.mean().item() + EPS))

    out = {
        **row,
        "scoring_observer_model": observer_model_id,
        "scoring_reference_model": reference_model_id,
        "num_scored_tokens": int(len(target_ids)),
        "doc_scores": doc_scores,
        "token_features": token_features,
    }
    if store_tokens:
        out["scored_token_ids"] = target_ids.cpu().tolist()
        out["scored_tokens"] = observer_tokenizer.convert_ids_to_tokens(target_ids.cpu().tolist())
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observer-model", default=None)
    parser.add_argument("--reference-model", default=None)
    parser.add_argument(
        "--binoculars-default-models",
        action="store_true",
        help=(
            "Use upstream Binoculars defaults: performer/numerator "
            "tiiuae/falcon-7b-instruct and observer/comparison tiiuae/falcon-7b."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reference-device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--store-tokens", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"no rows found in {args.input}")

    if args.binoculars_default_models:
        observer_model_id = BINOCULARS_PERFORMER_MODEL
        reference_model_id = BINOCULARS_OBSERVER_MODEL
        trust_remote_code = args.trust_remote_code
    else:
        observer_model_id = args.observer_model or rows[0].get("evaluator_model") or rows[0].get("generator_model")
        reference_model_id = args.reference_model
        trust_remote_code = args.trust_remote_code

    if observer_model_id is None:
        raise ValueError("observer model is required")

    device = torch.device(args.device)
    reference_device = torch.device(
        args.reference_device
        or ("cuda:1" if args.binoculars_default_models and torch.cuda.device_count() > 1 else args.device)
    )
    observer_tokenizer, observer_model = load_model_and_tokenizer(observer_model_id, device, trust_remote_code)

    reference_model = None
    if reference_model_id:
        reference_tokenizer, reference_model = load_model_and_tokenizer(
            reference_model_id,
            reference_device,
            trust_remote_code,
        )
        check_compatible_tokenizers(observer_tokenizer, reference_tokenizer)

    scored = []
    for idx, row in enumerate(rows, start=1):
        scored.append(
            score_row(
                row,
                observer_tokenizer,
                observer_model,
                reference_model,
                device,
                reference_device,
                store_tokens=args.store_tokens,
                observer_model_id=observer_model_id,
                reference_model_id=reference_model_id,
            )
        )
        print(f"scored {idx}/{len(rows)}", flush=True)

    write_jsonl(Path(args.output), scored)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
