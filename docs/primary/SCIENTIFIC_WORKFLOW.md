# Primary scientific workflow: released nine-cell study

Audited against active code and downloaded final revision/configuration records
on 2026-09-08. This document describes the released analysis, not an instruction
to overwrite it. See [the codebase guide](../CODEBASE_GUIDE.md),
[detector revisions](../DETECTOR_REVISION.md) and
[RAID's separate scientific design](../raid/SCIENTIFIC_DESIGN.md).

## 1. Scope and evidence hierarchy

The released primary study crosses XSum, SQuAD and WritingPrompts with
`ibm-granite/granite-3.3-8b-base`,
`mistralai/Mistral-Small-24B-Base-2501` and `Qwen/Qwen2.5-32B`:
**nine primary cells**. Erebus 20B and Qwen 7B/14B/72B remain in the configuration
as historical replication/scaling plans. Their additional twelve cells are not
in the final bundle and must not be described as completed evidence. Qwen 32 is
shared between primary and proposed scaling analyses, not an independent repeat.

RAID is the separate realistic-attack benchmark. Beemo and splice-artifact
studies are historical material under local-only `archive/`, not primary results.

For a reported number, use this authority order: final bundle metrics and its
transformation provenance; selected source/revision manifests and frozen specs;
the corresponding versioned code; then current defaults and narrative docs.
`configs/paper.json` is **not the exact saved configuration of every cell**.
The base `run_experiment.py` retains historical evaluation. Final corrected
analysis uses
[`tools/reanalysis/reevaluate_detector_revision.py`](../../tools/reanalysis/reevaluate_detector_revision.py).
Running the base pipeline alone does not recreate corrected publication tables.

## 2. Research question and interpretation

The study tests document-detector robustness to controlled token-splice
contamination, comparing raw and clipped aggregation on identical texts.
Contaminated machine continuations remain machine-positive; clean human
continuations are negative. Replacement material comes from human text, but
the resulting token splices are **not certified grammatical human edits**.
They can disrupt syntax, token boundaries and conditional context. Gains may
reflect robustness to these disruptions, not specifically human authorship.

A realized replacement budget is not the theoretical conditional mixture
probability in a contamination theorem: conditional distributions downstream of
an edit can change beyond directly replaced positions. Decreasing detection
with contamination and improvement under clipping are hypotheses, not enforced
properties. Null/adverse results remain results.

Do not describe the entire amended analysis as preregistered without a dated
registration covering the actual version. Earlier results and pilots were
inspected before subsequent amendments. Separation of fitting/calibration/test
rows prevents direct test-row fitting but does not erase study-level adaptivity.

## 3. Sources and splits

| Dataset | Dataset ID / split | Source field |
|---|---|---|
| XSum | `EdinburghNLP/xsum`, validation | `document` |
| SQuAD | `rajpurkar/squad`, train | Deduplicated `context`; short unique contexts packed without reuse |
| WritingPrompts | `euclaise/writingprompts`, validation | First present of `story`, `text`, `completion` |

`select_source_manifest` in
[`pipeline.py`](../../experiment_core/preparation/pipeline.py) normalizes,
deduplicates, applies a 260-word filter (packing where enabled), and deterministically
hash-selects sources. Its manifest signature is independent of target model.
Before a new paired cross-model analysis, verify exact source overlap from saved
source manifests rather than assuming it from current code alone.

Each cell uses 3,000 selected sources: 500 `clipping_tuning`, 500 `calibration`,
2,000 `test`, with selection seed 1731. Generation seeds 101/202/303 divide the
sources round-robin, not multiply their count. A packed SQuAD source can contain
multiple original contexts; 3,000 sources does not mean 3,000 original QA pairs.
All variants and random draws for a source are dependent observations.

## 4. Generation, scoring context and saved overrides

All nine saved revision configurations specify Transformers generation, nominal
30-token source prefix, target/max 220 new tokens, minimum 210, temperature 0.8,
top-p 0.95, batch 8, checkpoint 96, length bucket 32 and BF16. Actual continuation
length can differ from 220. Human and machine continuations are length-matched
before corruption. No chat template is introduced.

`prepare_scoring_tokens` in
[`scoring.py`](../../experiment_core/detectors/scoring.py) separately encodes
prompt and response with no added special tokens and concatenates their IDs.
Only response predictions contribute; prompt tokens provide context. If combined
length exceeds 512 tokens, the response is right-truncated to the space left
after the prompt. The saved Falcon pair uses this same policy. 220 target-model
tokens need not be 220 Falcon tokens; generated length and scored length differ.

The saved manifests establish these differences from current defaults:

| Cells | Generation/scoring device map | Recorded generation TP | Pooled hidden state saved | Base round-trip guard | Explicit constructed guard |
|---|---|---:|---|---:|---:|
| Granite, all 3 | `auto` | 2 | true | 8 | absent |
| Mistral, all 3 | `auto` | 4 | false | 12 | absent |
| Qwen 32 XSum/SQuAD | `auto` | 4 | false | 12 | absent |
| Qwen 32 WritingPrompts | `auto` | 4 | false | 12 | 20 |

Recorded `tensor_parallel_size` is not evidence of actual tensor-parallel
execution under the Transformers/device-map backend. Scoring batch/microbatch
are 4, saved top-k 10, vocabulary chunk 8192. Hidden states are auxiliary outputs,
not inputs to the seven detectors.

Current preparation records decode/re-tokenize drift, can canonically normalize
and length-match base continuations outside their guard, and falls back to the
base guard when no separate constructed guard exists. Do not retroactively
describe all historical cells with current 12/20 defaults. Historical per-row
normalization and empirical length distributions require original prepared/base
rows on Delta; large JSONL files are absent from the downloaded final bundle.

## 5. Contamination construction and counts

Ratios are 0/5/10/20/30/40/50%. For original length \(n\) and positive ratio
\(r\), token budget is \(\min(n,\max(1,\operatorname{round}(nr)))\), using
Python rounding. Replacement preserves the token-array length before decoding.
Recorded realized ratio uses donor count/final re-tokenized length, not edit
distance and not necessarily the fraction of distinct token IDs that changed.
See [`data.py`](../../experiment_core/preparation/data.py) and `pipeline.py`.

- **Random:** sentence-like human spans are shuffled, truncated to fill the
  budget, and placed in non-overlapping random recipient token windows. Three
  draws per positive ratio/source; windows can cut recipient sentences.
- **Tail:** candidate human spans are scored once for mean target-model NLL
  conditional on the prompt, ordered highest first with candidate-ID tie breaking,
  cached across ratios, then used to replace a suffix. This is a white-box
  high-NLL donor/suffix attack, not optimization against all seven detectors
  or proof of a worst-case attack.

Spans use a regex punctuation/whitespace/newline splitter, not a linguistic
sentence parser. Corruption base seed 99173 is combined with dataset, target,
sample ID and draw ID; ratio is not part of that seed. Ratios/draws are not
independent experimental replications.

Each source has two clean rows (human/machine) plus six positive ratios times
four constructions (three random, one tail): 26 rows/source, 78,000 rows/cell.
Target and Falcon score packs must cover those same provenance keys. A test
random condition has 6,000 positive rows from 2,000 sources; a tail condition has
2,000. Clean machine TPR appears in both mode panels for display, without
duplicating the original clean sample.

## 6. Final detector definitions

Let \(\ell_i=\log p(x_i\mid\mathrm{prefix})\), \(a_i=-\ell_i\),
\(r_i=1+\#\{v:\mathrm{logit}(v)>\mathrm{logit}(x_i)\}\), and
\(h_i=-\sum_v p_i(v)\log p_i(v)\). Logs are natural, rank ties are competition
ties, and bars denote response-token arithmetic means.

| Final detector ID | Raw score | Final direction |
|---|---|---|
| `log_likelihood` | \(\bar\ell\) | Learned from clean tuning means |
| `rank` | \(\bar r\) | Learned |
| `log_rank` | \(\overline{\log r}\) | Learned |
| `lrr` | \(\bar a/(\overline{\log r}+10^{-12})\) | Fixed+1: larger = machine-like |
| `entropy` | \(\bar h\) | Learned |
| `entropy_gap` | \(\overline{a-h}\) | Learned |
| `binocular_origin` | Performer mean NLL / observer-to-performer mean cross-entropy | Fixed-1: smaller = machine-like |

For learned direction, choose+1 when the machine clean tuning mean is at least
the human mean, otherwise-1. It is not selected using test AUROC. LRR means
DetectLLM likelihood/log-rank ratio, **not a likelihood ratio between two
probability models**. Entropy gap is unnormalized NLL minus entropy; do not
label it variance-normalized Fast-DetectGPT. Literature names must refer to the
exact implemented statistic.

Falcon performer is `tiiuae/falcon-7b-instruct`, observer `tiiuae/falcon-7b`.
Observer probabilities weight performer log probabilities in the cross-entropy.
Primary origin components share the saved prompt-conditioned response window:
label it **conditional Binoculars-ratio adaptation**, not exact output-only
upstream reproduction. RAID's official-window/BF16-anchored path is separate.
The old exponential mean-gap score (`binoculars`, renamed `binocular_gap`)
remains for compatibility but is excluded from final publication views.

## 7. Clipping, candidate selection and degeneracy guards

Implementation: `_candidate_specs`, `tune_clipping_spec`, `orientation`,
`evaluate` in [`evaluation.py`](../../experiment_core/analysis/evaluation.py),
and guards in [`detector_revision.py`](../../experiment_core/detectors/detector_revision.py).

Generic oriented token evidence \(z_i=d s_i\) is replaced by
\(\max(z_i,L)\) and averaged. LRR caps both components:

$$
S_{\mathrm{LRR,clip}}=
\frac{\overline{\min(a_i,U_a)}}{\overline{\min(\log r_i,U_r)}+10^{-12}}.
$$

Primary origin caps only numerator NLL; its cross-entropy denominator is
unchanged. This is an experimental extension, not the published raw detector.

Candidate quantiles pool **clean human and clean machine tuning tokens**;
documents with more tokens contribute more to these quantiles. Grid:
0.80/0.85/0.90/0.95/0.975/0.99/0.995 plus no clipping. Generic bounds are lower
\(1-q\) quantiles of oriented evidence; origin bounds are upper NLL quantiles.
LRR uses the seven-by-seven Cartesian grid: 49 cap pairs plus raw before
rejection/duplicate values.

The attacked tuning mixture pools random and tail rows at 10/20/30/40/50%,
excluding 5%. With three random draws versus one tail draw it has **3: 1
random:tail row weighting**, not equal-weight attack-average AUROC. Objective:

$$
J=0.8\,\mathrm{AUROC}(H_{\rm tune},M_{\rm attacked,tune})
 +0.2\,\mathrm{AUROC}(H_{\rm tune},M_{\rm clean,tune}).
$$

The clean term is a soft reward, not a hard zero-clean-loss constraint. RAID
pilot selectors and clean-loss budgets must not be imported into this primary
description. No-clipping is considered first; improvement must exceed
\(10^{-15}\), leaving the earlier candidate on ties. One frozen specification
serves every mode, ratio and FPR for a dataset/model/detector. Optional
mode-specific oracle is disabled.

Revised evaluation rejects nonempty candidates with constant clean tuning
document scores or exact structural full saturation; it adds no near-constant
tolerance. A constant origin numerator alone is not sufficient for rejection
if its denominator retains information. LRR requires a finite strictly positive
log-rank cap: zero would erase rank evidence even when the score still varies.
Rejections appear in `clipping_rejections.json`. Qwen 32-SQuAD V4.2 specifically
corrects this denominator-collapse candidate using saved features, not new
inference. This fix must not be described merely as flipping a plotted curve.

## 8. Calibration, metrics and uncertainty

After fixing direction/specification, calibrate raw and clipped thresholds
separately using 500 clean human calibration documents at 1% and 5% target FPR.
For sorted scores and \(m=\lfloor\alpha n\rfloor\), use the next representable
number above zero-based order statistic \(n-m-1\); classify scores greater than
or equal to threshold. At most \(m\) calibration observations exceed it.
Held-out test FPR need not equal target FPR and must accompany TPR claims.

Report TPR, actual FPR, AUROC, normalized partial AUROC over 0–5% FPR, paired
clipped-minus-raw differences, and normalized TPR-versus-contamination robustness
AUC. Partial AUROC is trapezoidal ROC area divided by 0.05, **not** the
McClish/sklearn standardized partial-AUC convention. Robustness AUC is the
trapezoidal contamination-curve area divided by its ratio range (0.5).

Final intervals use 2,000 percentile bootstrap repetitions, 2.5th/97.5th
percentiles, base seed 481516 and deterministic condition-specific seeds.
`_metric_bootstrap` resamples `sample_id` clusters, taking all relevant
human/machine rows and random draws with each sampled source. Raw/clipped
differences share the resample. `_robustness_bootstrap` keeps each resampled
source across ratios. Bounds and calibration thresholds remain fixed: these
are test-source sampling intervals conditional on fitting/calibration, not
uncertainty of the complete tuning procedure. Intervals are pointwise, not
simultaneous or multiplicity-adjusted. Repeated clean/robustness values in tidy
rows do not create additional independent observations.

## 9. Final artifact lineage

Local publication bundle:
`downloads/current/primary-nine-plus-raid-final-20260908T150247Z/primary/`.
`bundle_summary.json` records 9 cells, 3,528 metric rows, 7 methods and 99 plots
(11/cell). Each cell contributes 392 metric records:
\(7\times 2\times 7\times 2\times 2\), for detectors/modes/ratios/FPRs/aggregations.
These are metric records, not prepared texts.

| Cell | Prepared/scored source run |
|---|---|
| Granite XSum | `granite8b-xsum-full-20260730T063720Z` |
| Granite SQuAD | `granite8b-squad-full-20260730T063720Z` |
| Granite WritingPrompts | `granite8b-writingprompts-full-20260730T063720Z` |
| Mistral XSum | `mistral24-xsum-full-20260803T172630Z` |
| Mistral SQuAD | `mistral24-squad-full-20260803T172630Z` |
| Mistral WritingPrompts | `mistral24-writingprompts-roundtripfix-20260804T133827Z` |
| Qwen 32 XSum | `qwen32-xsum-full-v2-20260803T051524Z` |
| Qwen 32 SQuAD | `qwen32-squad-promptfix-20260804T133827Z` |
| Qwen 32 WritingPrompts | `qwen32-writingprompts-guard20-20260821T152530Z` |

Eight cells use source ID plus `-revision-20260904T163234Z`. Qwen 32-SQuAD uses
`qwen32-squad-promptfix-20260804T133827Z-lrr-v42-20260907T163410Z`.
Copied original source reports may state 8 detectors/448 rows and include gap.
They remain immutable provenance. The final export filters to 7/392; its bundle
manifest/hashes establish that transformation. Do not apply an original source
report's artifact hashes/counts to the transformed publication CSV.

`primary_all_metrics.csv` is the tidy final table; clean, clipping-effect and
robustness files are derived views. `primary_cells/` preserves cell material.
Configurations/hashes are provenance, not substitutes for score packs/weights.
Resolved model/tokenizer/dataset IDs should be recovered from original source
and scoring manifests for strict replication; null config revision fields alone
do not supply pinning. This local audit did not replay Delta inference.

## 10. Reporting and change boundaries

A TPR increase at a higher held-out FPR is not automatically a like-for-like
improvement. Low/zero SQuAD performance is not grounds for a test-selected
direction reversal. Present primary conditional origin and RAID official origin
separately. RAID fits its own tuning/calibration splits; it is not frozen-threshold
transfer from these cells. Avoid claiming gains for every detector or attack.

Changes to sources, generation/tokenizer or contamination require rebuilding
affected texts/features. A formula change can reuse features only when every
required component/position is saved. Tuning/guard/bootstrap changes need a new
evaluation ID, not new GPU inference when features suffice. Styling needs only
plotting. Preserve historical outputs and record amendments. Optional vLLM,
unreleased models, actual hardware timings and empirical length distributions
require their own evidence; current code/config alone does not establish them.
