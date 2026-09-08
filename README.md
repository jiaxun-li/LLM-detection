# Robust LLM Detection Under Human-Text Contamination

This repository implements a configuration-first, restartable experiment for
comparing raw token aggregation with one-sided clipped aggregation under random
and white-box tail human-text contamination. The primary protocol targets an
ICLR/AISTATS-quality study across XSum, SQuAD, and WritingPrompts.

## Canonical guides

- Start with the [documentation index](docs/README.md),
  [configuration guide](docs/CONFIGURATION.md), and
  [local archive catalog](docs/ARCHIVE_CATALOG.md).
- [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) explains active scripts, archived
  studies, and local downloads. The current released analysis is the corrected
  nine-cell primary study plus RAID; the broader matrix below records the
  original study design, not a claim that every planned cell was completed.
- [`docs/DETECTOR_REVISION.md`](docs/DETECTOR_REVISION.md) specifies the corrected
  detector reanalysis used by the final results.
- [`docs/primary/SCIENTIFIC_WORKFLOW.md`](docs/primary/SCIENTIFIC_WORKFLOW.md) defines the frozen
  21-cell protocol, leakage rules, metrics, result classes, and interpretation.
- [`docs/CODEBASE_GUIDE.md`](docs/CODEBASE_GUIDE.md) maps that protocol to current
  modules, files, I/O contracts, resume behavior, tests, and Delta operations.

Read both before changing a configuration or launching a full-scale cell.

## Paper design

The paper configuration is [`configs/paper.json`](configs/paper.json). It
records:

- 500 clipping-tuning, 500 human calibration, and 2,000 final-test source IDs;
- three generation seeds that partition those 3,000 IDs;
- a 30-token prompt, about 220 continuation tokens, temperature 0.8, top-p 0.95;
- random and white-box tail contamination at 0%, 5%, 10%, 20%, 30%, 40%, and
  50%, with three random corruption draws;
- a twelve-token prompt/base round-trip guard and a twenty-token constructed-row
  guard; the latter was amended before Qwen-32B WritingPrompts scoring after all
  72,000 positive-ratio constructions were audited (34 exceeded twelve tokens
  and the maximum was nineteen);
- Granite 3.3 8B Base, Mistral Small 24B, and Qwen 2.5 32B primary targets;
- the GPT-NeoX Erebus replication and Qwen 7B/14B/32B/72B scaling models;
- target-model log likelihood, rank, log rank, DetectLLM LRR, entropy, and
  entropy gap, plus the Falcon performer/observer Binoculars pair.

[`configs/smoke.json`](configs/smoke.json) reduces split sizes, corruption
draws, bootstrap repetitions, and model size without changing the protocol.

Granite 3.3 8B Base replaces the originally planned Llama 3.1 8B target so the
primary study remains reproducible without provider-specific geographic access
approval. The replacement preserves the approximately 8B base-model scale and
adds a public Apache-2.0 model from a distinct model family.

## Entry point

One task handles one dataset × target model:

```bash
python run_experiment.py \
  --config configs/smoke.json \
  --dataset xsum \
  --model Qwen/Qwen2.5-0.5B \
  --run-id smoke-xsum-qwen \
  --stage all
```

The stages can also be resumed independently:

```bash
python run_experiment.py --config configs/smoke.json --dataset xsum \
  --model Qwen/Qwen2.5-0.5B --run-id smoke-xsum-qwen --stage prepare
python run_experiment.py --config configs/smoke.json --dataset xsum \
  --model Qwen/Qwen2.5-0.5B --run-id smoke-xsum-qwen --stage score
python run_experiment.py --config configs/smoke.json --dataset xsum \
  --model Qwen/Qwen2.5-0.5B --run-id smoke-xsum-qwen --stage evaluate
```

`archive/legacy_entrypoints/prepare_real_contamination.py`, `archive/legacy_entrypoints/score_real_text.py`, and
`archive/legacy_entrypoints/evaluate_real_clipping.py` remain as aliases for these three stages. They now
accept the configuration-first arguments above; the old prototype flags are no
longer supported.

Do not run `--stage all` locally unless model and dataset downloads are
intentional. The automated tests need no downloads:

```bash
python -m unittest discover -s tests -v
```

## Output layout and restart behavior

Each run writes:

```text
runs/source_manifests/<dataset>-<split>-<count>-seed<seed>-<signature>.jsonl
runs/<run-id>/manifest.json
runs/<run-id>/base_generations.jsonl
runs/<run-id>/tail_candidate_cache.jsonl
runs/<run-id>/data.jsonl
runs/<run-id>/target_scores.jsonl
runs/<run-id>/binoculars_scores.jsonl
results/<run-id>/metrics.csv
```

Source manifests are model-independent, so every target model uses identical
source/sample IDs and split labels. JSONL stages append complete fsynced records,
repair only an interrupted final line, index completed provenance keys, and
write completion markers only after row-count/uniqueness validation.
Repeated SQuAD contexts are deduplicated; short unique passages are packed
deterministically without reuse before source IDs are assigned, ensuring the
30+220 token construction has enough target-independent source text.

Each target-score row uses the versioned `target-token-features-v2` schema. It
keeps per-token log-probability, exact rank, log-rank, and entropy and also saves
the configured top-k token IDs/log-probabilities, the top-1 versus top-2
log-probability margin, and the observed target versus top-1 margin. The default
is `scoring.saved_top_k: 10`. Full-vocabulary logits are deliberately not
persisted.

`scoring.save_mean_pooled_final_hidden_state` optionally adds one mean-pooled
final-layer vector per document. It is disabled for paper runs by default
because it increases forward-pass memory and output size; the dedicated Delta
smoke configuration enables it to exercise the code path. Score rows and the
run manifest retain resolved model and tokenizer commit revisions so a
specialized feature can be reproduced later.

The tidy metrics CSV contains calibrated thresholds, actual FPR, TPR at
calibrated 1% and 5% FPR, AUROC, normalized partial AUROC over 0–5% FPR, paired
clipped-minus-raw differences, robustness AUC, clustered bootstrap intervals,
sample counts, contamination provenance, model revisions, and frozen clipping
specifications.

## NCSA Delta

See [`docs/delta/DELTA.md`](docs/delta/DELTA.md). The normal job uses `gpuA100x4`; `gpuH200x8` is an
explicit opt-in for confirmed large-model runs such as Qwen 72B.

Before any matrix work, use the isolated one-GPU Qwen 0.5B gate in
[`docs/delta/DELTA_SMOKE.md`](docs/delta/DELTA_SMOKE.md). It has an account-aware submission wrapper,
strict output validation, and a mandatory duplicate-free resume pass.

Implementation status is audited in
[`docs/primary/REQUIREMENT_CHECKLIST.md`](docs/primary/REQUIREMENT_CHECKLIST.md).
