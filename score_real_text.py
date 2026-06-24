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

The observer model A provides:

  - logp_a
  - rank_a
  - logrank_a
  - entropy_a

If a reference model B is provided with the same tokenizer, the script also
computes:

  - logp_b
  - Binoculars-style cross entropy: H(p_B, p_A)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


EPS = 1e-12


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


def load_model_and_tokenizer(model_id: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
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
    store_tokens: bool,
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
        reference_logits = model_logits(reference_model, input_ids, device)[start:end]
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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

    observer_model_id = args.observer_model or rows[0].get("evaluator_model") or rows[0].get("generator_model")
    if observer_model_id is None:
        raise ValueError("observer model is required")

    device = torch.device(args.device)
    observer_tokenizer, observer_model = load_model_and_tokenizer(observer_model_id, device)

    reference_model = None
    if args.reference_model:
        reference_tokenizer, reference_model = load_model_and_tokenizer(args.reference_model, device)
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
                store_tokens=args.store_tokens,
            )
        )
        print(f"scored {idx}/{len(rows)}", flush=True)

    write_jsonl(Path(args.output), scored)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
