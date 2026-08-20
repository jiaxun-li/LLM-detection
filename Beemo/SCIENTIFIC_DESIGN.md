# Scientific design: Beemo expert-edit benchmark

## Purpose and status

This protocol is a new benchmark-centered primary study. It is not another
cell in the original 21-cell synthetic contamination matrix, and its results
must not be pooled with that matrix. The original matrix remains evidence about
token-splice stress; Beemo directly tests naturally coherent, expert-edited
machine outputs.

Beemo was introduced at NAACL 2025 and contains 2,187 record groups. Each group
provides the original user prompt, a human-written reference, an output from one
of ten instruction-tuned generators, an expert edit of that machine output,
three Llama-3.1-70B edits, and three GPT-4o edits. See the
[paper](https://aclanthology.org/2025.naacl-long.357/),
[official dataset card](https://huggingface.co/datasets/toloka/beemo), and
[official repository](https://github.com/Toloka/beemo).

## Research question

When the content has machine origin but is edited by a human expert, do raw
zero-shot detector scores still identify machine involvement, and does the
repository's one-sided clipping improve detection retention at strict human
false-positive rates?

The primary positive class is `expert`: an expert-edited machine response. The
negative class is `human`: the independent human-written response to the same
prompt. An expert edit is therefore not relabeled as human; the task is to
detect machine involvement after editing.

## Text versions and roles

Every record produces exactly nine rows, all kept in the same split.

| Condition | Rows per record | Role |
|---|---:|---|
| `human` | 1 | Negative human reference |
| `original` | 1 | Unedited machine-origin control |
| `expert` | 1 | Primary expert-edited positive |
| `llama_p1`–`llama_p3` | 3 | Automated-editor controls |
| `gpt_p1`–`gpt_p3` | 3 | Automated-editor controls |

The pooled `llama_all` and `gpt_all` conditions are evaluation views, not new
texts. Their three versions travel together in every record-cluster bootstrap.

## Fixed grouping and splits

Beemo publishes one 2,187-record split. This study creates an internal,
deterministic group split by hash-ordering Beemo `id` with seed `241104`:

| Split | Records | Permitted use |
|---|---:|---|
| `clipping_tuning` | 437 | Direction and clipping choice only |
| `calibration` | 875 | Raw/clipped thresholds from human references only |
| `test` | 875 | Every released final metric |

All nine variants associated with one `id` stay together. Prompts, generator
identity, category, and edited variants therefore cannot leak across splits.

## Scoring context and models

The six single-model detectors use four primary reference scorers drawn from
the Beemo paper's zero-shot backbone family:

- `openai-community/gpt2-xl`;
- `facebook/opt-1.3b`;
- `tiiuae/falcon-7b`;
- `Qwen/Qwen2-7B`.

`ibm-granite/granite-3.3-8b-base` remains an optional fifth scorer and is not
part of the primary run. Binoculars retains the repository's frozen pair:

- performer: `tiiuae/falcon-7b-instruct`;
- observer: `tiiuae/falcon-7b`.

The six single-model detectors receive only the released response text. They
use tokenizer-default special tokens and causal next-token shifting, with no
silent length truncation. Binoculars also receives only response text but keeps
its upstream 512-token truncation rule. The original user prompt remains in the
prepared provenance but is not passed to any detector. The same policy is used
for human, original, expert-edited, and LLM-edited versions.

The seven reported detectors are log likelihood, rank, log rank, LRR, entropy,
entropy gap, and Binoculars. Log likelihood, rank, log rank, and LRR are the
core authorship detectors; entropy and entropy gap are supporting diagnostics;
Binoculars is an external zero-shot baseline.

## Leakage-safe clipping and calibration

For each scorer-detector combination:

1. Determine direction from tuning-split `human` versus `original` scores.
2. Construct clipping candidates from tuning-split human and original token
   evidence only.
3. Score each candidate by the mean of four family-level AUROCs: original,
   expert, pooled Llama edits, and pooled GPT edits. Each family receives 25%
   weight, so six automated edits cannot numerically dominate the one expert
   edit.
4. Freeze one clipping specification.
5. Compute separate raw and clipped 1% and 5% FPR thresholds using only the 875
   calibration human references.
6. Apply all frozen choices to the final 875 record groups.

No final-test score may change direction, clipping, threshold, condition list,
or model choice.

## Outcomes

The primary condition is `expert`. For raw and clipped aggregation at 1% and
5% target FPR, the code reports:

- actual held-out human FPR;
- TPR with percentile confidence interval;
- AUROC and normalized partial AUROC through 5% FPR;
- paired clipped-minus-raw differences;
- condition-minus-original TPR and AUROC, measuring detection retention lost
  after editing.

The primary scientific emphasis is estimation of the expert-edit effect for
the four core detectors at 1% FPR. The 5% results, automated editors, entropy
family, Binoculars, generator subgroups, categories, and edit-ratio strata are
secondary. Do not select a detector only because its final-test curve looks
favorable. The current code provides paired confidence intervals rather than a
multiplicity-adjusted null-hypothesis test; any later formal testing plan must
be written before inspecting its new test statistic.

## Uncertainty

The full configuration uses 2,000 bootstrap repetitions. Sampling is clustered
and paired by Beemo `id`: one sampled record brings its human reference,
original response, expert edit, and all six automated edits together. Raw and
clipped scores use the same sampled IDs. Calibration thresholds remain frozen
during the bootstrap.

## Diagnostics and confounds

`subgroup_metrics.csv` reports descriptive performance by Beemo category,
source generator, and expert word-edit-ratio band. These are not independently
calibrated analyses; they use the global frozen thresholds.

Important interpretation limits:

- human references and expert edits are different text-construction processes;
- editing changes length as well as wording;
- Beemo generators are older instruction-tuned models, while the four
  zero-shot reference scorers are external cross-generator models;
- the dataset is English-only;
- the created split is ours, not an official Beemo train/test split;
- the paper does not pin every model/tokenizer revision or software version, so
  the released protocol records resolved runtime revisions.

The dataset card lists MIT at the repository level, but the official license
section states that No Robots prompts/human texts are CC-BY-NC-4.0 and generated
or edited outputs can inherit their source-model terms. Do not redistribute a
packaged copy of the full dataset without reviewing those terms.

## Success and redesign decisions

- If clipping improves expert-edit TPR with a paired interval clearly above
  zero for core detectors, the paper can claim transfer from synthetic audit
  motivation to genuine expert edits.
- If clipping helps only original or automated-edited text, it is not evidence
  for human-edit robustness.
- If expert TPR collapses and clipping does not recover it, the negative result
  is scientifically important: clipping does not solve realistic editing.
- If raw/clipped rank or LRR behavior agrees with the prior sentence-aligned
  audit, report that convergence; if it disagrees, Beemo takes priority for the
  realistic-edit claim.
- RAID should be added only after this pipeline passes, as an external attack
  and domain robustness layer rather than as a substitute for expert editing.
