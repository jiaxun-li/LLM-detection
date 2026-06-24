"""Prepare real-text contamination data for clipped detector experiments.

This script builds a small Radvand-style setup:

  1. Load human texts from XSum, SQuAD, or WritingPrompts.
  2. Use the first prompt_tokens as a prompt.
  3. Keep the next continuation_tokens as the human continuation.
  4. Generate an LLM continuation from the same prompt.
  5. Create random and tail contaminated LLM continuations.

The output is JSONL so later scoring scripts can compute logp, rank, log-rank,
Binoculars, and entropy-gap features without regenerating text.

Example:

  python prepare_real_contamination.py --dataset xsum --n-samples 100

Notes:

  - This needs `datasets`, `transformers`, and `torch`.
  - Tail contamination uses evaluator-model NLL to choose the most harmful
    human sentences. It is intentionally detector-facing, unlike random mixing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASETS = {
    "xsum": {
        "hf_id": "EdinburghNLP/xsum",
        "split": "validation",
        "text_fields": ["document"],
    },
    "squad": {
        "hf_id": "rajpurkar/squad",
        "split": "validation",
        "text_fields": ["context"],
    },
    "writingprompts": {
        "hf_id": "euclaise/writingprompts",
        "split": "validation",
        "text_fields": ["story", "text", "completion"],
    },
}


DEFAULT_RATIOS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


@dataclass
class BaseExample:
    dataset: str
    example_id: str
    prompt: str
    human_continuation: str
    llm_continuation: str


def sentence_split(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def detokenize_sentences(sentences: Iterable[str]) -> str:
    return " ".join(sentence.strip() for sentence in sentences if sentence.strip())


def get_text(row: dict, candidate_fields: list[str]) -> str | None:
    for field in candidate_fields:
        value = row.get(field)
        if isinstance(value, str) and len(value.split()) >= 80:
            return value
    return None


def token_slice_text(tokenizer, text: str, start: int, end: int) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return tokenizer.decode(ids[start:end], skip_special_tokens=True).strip()


def build_prompt_and_human_continuation(
    tokenizer,
    text: str,
    prompt_tokens: int,
    continuation_tokens: int,
) -> tuple[str, str] | None:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) < prompt_tokens + max(40, continuation_tokens // 2):
        return None
    prompt = tokenizer.decode(ids[:prompt_tokens], skip_special_tokens=True).strip()
    continuation = tokenizer.decode(
        ids[prompt_tokens : prompt_tokens + continuation_tokens],
        skip_special_tokens=True,
    ).strip()
    if len(sentence_split(continuation)) < 2:
        return None
    return prompt, continuation


def load_human_texts(dataset_name: str, n_needed: int, seed: int) -> list[tuple[str, str]]:
    spec = DATASETS[dataset_name]
    dataset = load_dataset(spec["hf_id"], split=spec["split"])
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    out = []
    seen = set()
    for idx in indices:
        row = dataset[idx]
        text = get_text(row, spec["text_fields"])
        if text is None:
            continue
        key = re.sub(r"\s+", " ", text[:500]).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append((str(row.get("id", idx)), text))
        if len(out) >= n_needed:
            break
    return out


def generate_continuation(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs.input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def mean_nll(tokenizer, model, prefix: str, text: str, device: torch.device) -> float:
    """Mean NLL of `text` tokens under model after `prefix`."""
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    text_ids = tokenizer(text, add_special_tokens=False).input_ids
    if not text_ids:
        return float("-inf")

    ids = torch.tensor([prefix_ids + text_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids).logits[:, :-1, :]

    labels = ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # Positions predicting the text tokens. The first text token is predicted
    # from the last prefix token.
    start = max(len(prefix_ids) - 1, 0)
    end = start + len(text_ids)
    return float(-token_logp[0, start:end].mean().item())


def contaminate_random(
    rng: random.Random,
    llm_text: str,
    human_text: str,
    ratio: float,
) -> str:
    llm_sentences = sentence_split(llm_text)
    human_sentences = sentence_split(human_text)
    if not llm_sentences or not human_sentences or ratio <= 0:
        return llm_text

    k = min(len(llm_sentences), max(1, round(len(llm_sentences) * ratio)))
    positions = rng.sample(range(len(llm_sentences)), k)
    replacements = [rng.choice(human_sentences) for _ in positions]

    mixed = list(llm_sentences)
    for pos, replacement in zip(positions, replacements):
        mixed[pos] = replacement
    return detokenize_sentences(mixed)


def contaminate_tail(
    tokenizer,
    model,
    device: torch.device,
    prompt: str,
    llm_text: str,
    human_text: str,
    ratio: float,
) -> str:
    llm_sentences = sentence_split(llm_text)
    human_sentences = sentence_split(human_text)
    if not llm_sentences or not human_sentences or ratio <= 0:
        return llm_text

    k = min(len(llm_sentences), len(human_sentences), max(1, round(len(llm_sentences) * ratio)))
    scored = [
        (mean_nll(tokenizer, model, prompt, sentence, device), idx, sentence)
        for idx, sentence in enumerate(human_sentences)
    ]
    scored.sort(reverse=True)
    selected = scored[:k]

    # Put the most harmful human sentences into evenly spaced positions so the
    # contamination is not accidentally concentrated in one paragraph ending.
    positions = [
        min(len(llm_sentences) - 1, math.floor((j + 0.5) * len(llm_sentences) / k))
        for j in range(k)
    ]
    mixed = list(llm_sentences)
    for pos, (_, _, replacement) in zip(positions, selected):
        mixed[pos] = replacement
    return detokenize_sentences(mixed)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="xsum")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--evaluator-model", default=None)
    parser.add_argument("--prompt-tokens", type=int, default=30)
    parser.add_argument("--continuation-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--output-dir", default="real_data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    evaluator_model_id = args.evaluator_model or args.generator_model

    tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    generator = AutoModelForCausalLM.from_pretrained(
        args.generator_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    generator.eval()

    if evaluator_model_id == args.generator_model:
        evaluator_tokenizer = tokenizer
        evaluator = generator
    else:
        evaluator_tokenizer = AutoTokenizer.from_pretrained(evaluator_model_id)
        if evaluator_tokenizer.pad_token_id is None:
            evaluator_tokenizer.pad_token = evaluator_tokenizer.eos_token
        evaluator = AutoModelForCausalLM.from_pretrained(
            evaluator_model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        evaluator.eval()

    raw_texts = load_human_texts(args.dataset, args.n_samples * 4, args.seed)
    base_examples: list[BaseExample] = []

    for example_id, text in raw_texts:
        pair = build_prompt_and_human_continuation(
            tokenizer,
            text,
            prompt_tokens=args.prompt_tokens,
            continuation_tokens=args.continuation_tokens,
        )
        if pair is None:
            continue
        prompt, human_continuation = pair
        llm_continuation = generate_continuation(
            tokenizer,
            generator,
            prompt,
            max_new_tokens=args.continuation_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        if len(sentence_split(llm_continuation)) < 2:
            continue
        base_examples.append(
            BaseExample(
                dataset=args.dataset,
                example_id=example_id,
                prompt=prompt,
                human_continuation=human_continuation,
                llm_continuation=llm_continuation,
            )
        )
        print(f"prepared {len(base_examples)}/{args.n_samples}", flush=True)
        if len(base_examples) >= args.n_samples:
            break

    output_dir = Path(args.output_dir) / args.dataset / args.generator_model.replace("/", "__")

    clean_rows = []
    contaminated_rows = []
    for ex_idx, ex in enumerate(base_examples):
        clean_rows.append(
            {
                "dataset": ex.dataset,
                "example_id": ex.example_id,
                "sample_id": ex_idx,
                "prompt": ex.prompt,
                "label": "human",
                "text": ex.human_continuation,
                "source": "human_clean",
                "contamination_mode": "none",
                "contamination_ratio": 0.0,
                "generator_model": args.generator_model,
                "evaluator_model": evaluator_model_id,
            }
        )
        clean_rows.append(
            {
                "dataset": ex.dataset,
                "example_id": ex.example_id,
                "sample_id": ex_idx,
                "prompt": ex.prompt,
                "label": "llm",
                "text": ex.llm_continuation,
                "source": "llm_clean",
                "contamination_mode": "none",
                "contamination_ratio": 0.0,
                "generator_model": args.generator_model,
                "evaluator_model": evaluator_model_id,
            }
        )

        for ratio in args.ratios:
            for mode in ["random", "tail"]:
                if mode == "random":
                    mixed = contaminate_random(rng, ex.llm_continuation, ex.human_continuation, ratio)
                else:
                    mixed = contaminate_tail(
                        evaluator_tokenizer,
                        evaluator,
                        device,
                        ex.prompt,
                        ex.llm_continuation,
                        ex.human_continuation,
                        ratio,
                    )
                contaminated_rows.append(
                    {
                        "dataset": ex.dataset,
                        "example_id": ex.example_id,
                        "sample_id": ex_idx,
                        "prompt": ex.prompt,
                        "label": "llm",
                        "text": mixed,
                        "source": "llm_contaminated",
                        "contamination_mode": mode,
                        "contamination_ratio": ratio,
                        "generator_model": args.generator_model,
                        "evaluator_model": evaluator_model_id,
                    }
                )

    write_jsonl(output_dir / "clean.jsonl", clean_rows)
    write_jsonl(output_dir / "contaminated.jsonl", contaminated_rows)
    print(f"wrote {output_dir / 'clean.jsonl'}")
    print(f"wrote {output_dir / 'contaminated.jsonl'}")


if __name__ == "__main__":
    main()
