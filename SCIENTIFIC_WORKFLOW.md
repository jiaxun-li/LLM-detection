# Scientific workflow

This is the canonical scientific protocol for the repository. The frozen
full-scale specification is [`configs/paper.json`](configs/paper.json); the
smoke configurations are engineering checks and are not substitutes for the
paper experiment. Implementation details and commands live in
[`CODEBASE_GUIDE.md`](CODEBASE_GUIDE.md).

## Research question and hypotheses

The study asks whether document-level LLM detectors remain useful when a
generated continuation contains increasing amounts of human text, and whether
one-sided clipping of adverse token contributions improves robustness relative
to the detector's ordinary, raw token aggregation.

The preregistered directional hypotheses are:

1. Human-token contamination reduces detector TPR as its ratio increases, with
   white-box tail contamination at least as challenging as random
   contamination.
2. A clipping rule selected without calibration or test data improves the
   contamination-robustness curve and/or TPR at calibrated FPR, while preserving
   useful clean-text discrimination.
3. The result is not confined to one dataset or model family; it should appear
   in the primary matrix, remain interpretable in the Erebus replication, and
   show a coherent pattern across Qwen model scale.

Raw aggregation is the scientific control because it is the detector's ordinary
document score. Clipped aggregation is the intervention: after the detector's
orientation is fixed, it limits only unusually adverse token contributions (or
the corresponding NLL/log-rank components for LRR). Comparing the two on the
same documents, calibration data and threshold procedure, and bootstrap source
samples isolates the aggregation change rather than a change in generated text.

## Study matrix: exactly 21 unique cells

Every target model is crossed with XSum, SQuAD, and WritingPrompts. There are
seven unique target models and therefore **7 models x 3 datasets = 21 unique
dataset-model cells**.

| Role | Target model | Cells |
|---|---|---:|
| Primary | `ibm-granite/granite-3.3-8b-base` | 3 |
| Primary | `mistralai/Mistral-Small-24B-Base-2501` | 3 |
| Primary and Qwen scaling | `Qwen/Qwen2.5-32B` | 3 |
| Replication | `KoboldAI/GPT-NeoX-20B-Erebus` | 3 |
| Qwen scaling | `Qwen/Qwen2.5-7B` | 3 |
| Qwen scaling | `Qwen/Qwen2.5-14B` | 3 |
| Qwen scaling | `Qwen/Qwen2.5-72B` | 3 |

The primary analysis is the nine cells from Granite 8B, Mistral 24B, and Qwen
32B. The replication is the three Erebus 20B cells. The Qwen scaling analysis
contains 7B, 14B, 32B, and 72B across all three datasets, but the Qwen 32B cells
are the same three cells already produced for the primary analysis. They must
be reused, not generated, scored, or counted a second time.

The current matrix wrapper deduplicates model IDs within
`EXPERIMENT_SET=all`. By contrast, running `primary` and then
`qwen-scaling` as separate wrapper invocations would submit Qwen 32B twice.
Operators must use one deduplicated matrix or explicitly omit the already
completed Qwen 32B cells from later scaling work.

## Datasets and shared source manifests

The paper configuration uses:

| Short name | Hugging Face dataset | Split | Source field behavior |
|---|---|---|---|
| `xsum` | `EdinburghNLP/xsum` | `validation` | `document` |
| `squad` | `rajpurkar/squad` | `train` | deduplicated `context`; short unique contexts are packed without reuse |
| `writingprompts` | `euclaise/writingprompts` | `validation` | first available of `story`, `text`, or `completion` |

Source selection is target-model independent. For each dataset,
[`select_source_manifest`](llm_detection/pipeline.py) streams the dataset,
normalizes and deduplicates text, retains a deterministic hash-selected subset,
and writes one shared manifest under `runs/source_manifests/`. The manifest
signature depends on the dataset specification, split counts, and generation
seed list, but not on the target model. All target models therefore receive the
same `source_id`, `sample_id`, and split assignments.

Each dataset has exactly 3,000 unique source IDs:

- 500 `clipping_tuning` IDs;
- 500 `calibration` IDs;
- 2,000 `test` IDs.

The selection seed is 1731. The three sets are disjoint. Generation seeds 101,
202, and 303 are assigned round-robin across the selected examples, so the
seeds **divide the 3,000 examples**. They do not create three generations per
source and do not multiply the sample count.

## Generation and contamination

For every source, the target model receives a 30-token prompt and generates an
approximately 220-token continuation. The frozen generation settings are:

- target continuation length 220 tokens;
- `max_new_tokens: 220` and `min_new_tokens: 210`;
- temperature 0.8;
- top-p 0.95;
- Transformers by default, with vLLM as an optional generation backend.

The human continuation is taken from the same target-independent source after
the prompt and is truncated to the realized LLM continuation length. Human
contamination replaces target-tokenizer tokens one for one before decoding.
The stored row records requested and realized contamination ratios, original,
human, replaced, and final token counts, and decode/re-tokenize length drift.
The paper permits at most eight tokens of round-trip drift.

The fixed experimental ratios are 0%, 5%, 10%, 20%, 30%, 40%, and 50%. They
are experimental conditions, not hyperparameters: no ratio is selected,
dropped, or weighted in response to calibration or test performance.

Two attacks are constructed:

- **Random contamination** chooses human spans and non-overlapping replacement
  windows using a deterministic corruption seed. There are three draws per
  positive ratio and source. Multiple draws measure sensitivity to the random
  placement rather than letting a single favorable or unfavorable placement
  determine the result.
- **White-box tail contamination** splits the human continuation into candidate
  spans, scores every span once by target-model NLL conditional on the prompt,
  freezes the highest-NLL ordering in `tail_candidate_cache.jsonl`, and consumes
  that ordering into an approximately length-matched suffix replacement. The
  cache is reused across all ratios for that source.

Ratio zero is represented only by one clean human row and one clean LLM row. It
is not duplicated for attack mode or random draw. For each source:

```text
2 clean rows + 6 positive ratios x (3 random draws + 1 tail draw)
= 2 + 6 x 4
= 26 rows/source
```

Thus every full-scale cell has:

```text
3,000 sources x 26 rows/source = 78,000 prepared rows
```

Both the target-model score file and, when enabled, the Binoculars score file
must also contain 78,000 matching provenance keys.

## Seven detectors

All seven configured detectors produce one document score, but their token
evidence differs:

| Detector | Concept and implemented raw aggregation |
|---|---|
| Log likelihood | Mean target-model token log probability; generated text is expected to look more probable under its generator. |
| Rank | Mean one-based competition rank of the observed token, `1 + count(logit > target_logit)`. |
| Log rank | Mean natural log of that exact token rank. |
| DetectLLM LRR | Mean token NLL divided by mean log rank, with a small numerical epsilon. |
| Entropy | Mean exact full-vocabulary predictive entropy. |
| Entropy gap | Mean token NLL minus predictive entropy. |
| Binoculars | `exp(mean performer NLL - mean H(observer, performer))`. |

Binoculars has explicit, non-interchangeable roles:

- **Performer:** `tiiuae/falcon-7b-instruct`, which supplies observed-token NLL
  and the distribution in the cross-entropy log term.
- **Observer:** `tiiuae/falcon-7b`, whose predictive distribution weights the
  observer-to-performer cross-entropy.

The configured devices are `cuda:0` for the performer and `cuda:1` for the
observer. Local tests validate the roles and formula, but not the real
two-GPU execution.

## Leakage-safe analysis

For each dataset-model-detector cell, analysis proceeds in this order:

1. Use only clean human and clean LLM rows from `clipping_tuning` to determine
   whether larger or smaller detector values indicate LLM text.
2. Use only `clipping_tuning` rows to select one primary clipping specification.
   The frozen paper tuning mixture contains random and tail rows at 10%, 20%,
   30%, 40%, and 50%; 5% is evaluated but is not in this tuning mixture.
3. Freeze that direction and clipping specification for both attack modes, all
   ratios, calibration, and final testing. The optional mode-specific oracle is
   disabled in the paper configuration and is not a primary result.
4. Only after freezing the specification, precompute every row's raw and
   clipped scalar score once for that detector and analysis.
5. Derive separate raw and clipped thresholds at target FPR 1% and 5% using
   only the 500 clean **human** calibration documents. Calibration LLM rows and
   all test rows are excluded from threshold selection.
6. Apply the frozen thresholds to the 2,000 final-test source IDs. Test outcomes
   cannot change orientation, clipping, ratios, generation parameters, or
   thresholds.

These constraints prohibit tuning on final-test AUROC/TPR, choosing a favorable
contamination draw, choosing ratios after looking at results, or recalibrating
on attacked or external test data.

## Metrics and paired source-cluster bootstrap

For both raw and clipped aggregation, each detector, attack mode, ratio, and
target FPR reports:

- the calibration threshold and final-test actual FPR;
- TPR at calibrated 1% and 5% target FPR;
- AUROC;
- normalized partial AUROC over 0%-5% FPR;
- paired clipped-minus-raw differences for TPR, AUROC, and partial AUROC;
- normalized AUC of TPR versus contamination ratio (robustness AUC);
- calibration-human, test-human, test-LLM, corruption-draw, and unique-source
  counts;
- percentile 95% confidence intervals.

The paper configuration uses 2,000 bootstrap repetitions and base seed 481516.
Resampling is paired and clustered by `sample_id`: a source ID is sampled with
replacement and all relevant rows for that source, including its random draws,
travel together. The same sampled source IDs are used for raw and clipped
scores, which makes their difference paired. Deterministic condition-specific
seeds are derived from the base seed. Calibration thresholds remain frozen
during this final-test bootstrap; the bootstrap quantifies test-sample
uncertainty, not a second calibration procedure.

## Reproducibility and provenance

Every prepared row carries dataset, source, split, generation, model, tokenizer,
contamination, length, and corruption provenance. Score rows add resolved model
and tokenizer revisions and the scoring feature schema. `manifest.json` records
the Git commit and dirty-worktree flag, software and accelerator information,
Slurm identifiers, configuration groups, input/output paths, and stage status.
Stage completion markers are written only after the caller's row-count and
uniqueness checks succeed.

The frozen JSON currently leaves dataset, model, and tokenizer revisions as
`null`. The loaders resolve current Hugging Face commits and record them in
manifests and rows, but a first run is not content-addressed in advance. For a
strict future replication, pin the recorded revisions in a new configuration
and use a new run ID. Do not mix rows from different resolved revisions.

On Delta, keep small source code and launch material under `/u`, reusable
environments and deliberately retained caches under `/projects`, and the
high-volume JSONL score packs, run directories, and results under `/work/hdd`.
The 78,000 token-feature rows plus a second Binoculars file per cell make
home-directory storage inappropriate; `/work/hdd` provides the intended
capacity and scratch I/O profile. This placement is an operational convention,
not currently enforced by the JSON configuration or Slurm wrapper: relative
paths resolve below `--workspace`, and the wrapper defaults that workspace to
`$PWD`.

## Full-scale versus engineering results

Only the paper values above define the frozen full-scale study. The current
small configurations differ deliberately:

| Setting | Paper | `configs/smoke.json` | Delta Qwen 0.5B smoke |
|---|---:|---:|---:|
| Split sizes | 500/500/2,000 | 4/4/8 | 4/4/4 |
| Total sources | 3,000 | 16 | 12 |
| Generation seeds | 101/202/303 | 101/202/303 | 101 only |
| Continuation target | 220 | 220 | 64 |
| Ratios | 0/5/10/20/30/40/50% | 0/10/50% | 0/20/50% |
| Random draws | 3 | 1 | 1 |
| Detectors | 7 | 7 | 6; Binoculars skipped |
| Bootstrap repetitions | 2,000 | 50 | 5 |
| Pooled hidden state | off | off | on for schema exercise |

Use the following result labels consistently:

- **Smoke** means a bounded execution or contract check. The dedicated Delta
  smoke is explicitly `SMOKE_TEST_DEBUG_ONLY_NOT_FOR_SCIENTIFIC_USE`.
- **Pilot/debug** means any reduced configuration, override, historical toy
  output, performance experiment, or failure investigation. It cannot be
  pooled with or reported as a paper result.
- **Primary** means the nine frozen Granite/Mistral/Qwen-32B cells.
- **Replication** means the three frozen Erebus cells, analyzed separately as a
  cross-family replication.
- **Scaling** means the four-size Qwen series, reusing the primary Qwen 32B
  cells.
- **External benchmark** means a new corpus evaluated after the internal study.
  Its result must be labeled separately and must not alter the primary analysis.

## External benchmarks and interpretation risks

An external benchmark must import the originating cell's frozen direction,
clipping specification, raw and clipped calibration thresholds, detector
formula, tokenizer/model revisions, prompt/generation policy where applicable,
and feature schema. It may report performance under those frozen choices, but
must not retune clipping or recalibrate thresholds on the external benchmark.
The current evaluator always performs the internal three-split tuning and
calibration protocol; a dedicated frozen-spec external-evaluation entry point
is still required before such benchmarks can be run safely.

Contamination can change more than authorship evidence. In particular, entropy
and entropy-gap may detect disrupted local coherence or tokenizer/statistical
artifacts at the human/LLM splice boundary. A robustness gain for these
detectors must therefore not be interpreted automatically as better authorship
detection. Report random versus suffix-tail results separately, retain length
and realized-ratio provenance, and consider boundary-matched controls in future
work.

Other current risks are recorded rather than hidden: no full model/dataset
matrix has been validated locally; real CUDA placement, throughput, and Slurm
resume behavior remain cluster-only; and source selection uses a
target-independent word-length filter that can still fail later for an unusual
target tokenizer.

## Which changes invalidate which artifacts

Always use a new run ID for a material protocol/configuration change. General
resume keys do not encode every configuration value.

| Change | Regenerate prepared text? | Rescore token features? | Re-evaluate? |
|---|---:|---:|---:|
| Dataset/revision, source IDs, split sizes, selection seed | Yes, including source manifest | Yes | Yes |
| Target model/tokenizer/revision, prompt, generation seed or sampling settings | Yes, including base generation and tail cache | Yes | Yes |
| Human-contamination ratio, draw count, corruption seed, length rule, or construction algorithm | Rebuild contaminated rows; rebuild tail cache if candidate selection changed | Yes for affected rows | Yes |
| Tail candidate scoring model/formula | Rebuild tail cache and tail rows | Yes for affected rows | Yes |
| Scoring model/revision, max-token truncation, token-feature or detector formula | No if prepared text is unchanged | Yes | Yes |
| Clipping quantiles/tuning mixture, target FPR, bootstrap seed/repetitions, metric or plot selection | No | No, if existing token features suffice | Yes |
| Plot styling only | No | No | No; regenerate plots only |
| External benchmark | Prepare and score its new texts | Yes | Apply frozen internal choices only; do not retune |
