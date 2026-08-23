# RAID external-contamination benchmark

## Status and scope

This document freezes the scientific protocol for the RAID external benchmark.
It is separate from the 21-cell controlled contamination study and the Beemo
realistic-edit benchmark. RAID is used to test whether the same one-sided
token-level clipping intervention remains useful under naturally heterogeneous,
attack-generated contamination.

The study records realized contamination rate for interpretation, but it fits
only one universal clipping specification per detector. There is no
contamination-rate-adaptive clipping and no attack-specific clipping.

The detector transformations are defined in [`../clipping_method.md`](../clipping_method.md).
For additive detectors and Binoculars, one specification is a single oriented
lower bound. For DetectLLM LRR, one specification is the pair of upper caps on
token NLL and token log rank. A single numerical bound is not shared across
detectors because their local evidence and scales differ.

## Research questions

1. Does raw Binoculars on the retained RAID sample show the same broad attack
   pattern as the published RAID benchmark at 5% FPR?
2. Does one universally fitted clipping specification improve TPR at 5% FPR
   for the seven detector methods under RAID attacks?
3. How does the paired clipped-minus-raw effect vary with realized token-level
   contamination rate?
4. Does clipping preserve useful detection on the unattacked machine condition?

The primary outcome is TPR at a target FPR of 5%. AUROC is used for clipping
selection and is retained as a secondary diagnostic, not as the primary final
outcome.

## RAID release and source unit

Use the core English RAID release corresponding to the ACL 2024 paper:

- 14,971 human source documents;
- eight domains: Abstracts, Books, News, Poetry, Recipes, Reddit, Reviews, and
  Wikipedia;
- 34 available generator/decoding configurations per source after RAID's
  balancing step;
- one unattacked machine output and eleven attacked variants per selected
  machine generation.

The paper's 14,971 sources describe the full core benchmark universe. The
current public labeled `train.csv` contains 13,371 source IDs from the labeled
training portion of that universe, while the public test release withholds labels. A run on the
current labeled release therefore records its actual source count and is
reported as the labeled-train realization of this protocol. An archival full
paper snapshot may instead recover all 14,971 sources. The Binoculars table is
a sanity comparator in either case, not an assertion of exact sample identity.

The independent sampling and bootstrap unit is the human `source_id`. A source
group contains one human document, one selected unattacked machine generation,
and all eleven attacked variants of that machine generation.

## Deterministic one-to-one sample

For every human source:

1. retain the human document once;
2. uniformly select one of the source's complete machine-generation families;
3. retain the selected unattacked generation and its eleven RAID attacks;
4. preserve the source ID, selected base-generation ID, generator, chat/base
   status, decoding method, repetition-penalty setting, attack name, and native
   RAID attack rate;
5. link every attack row to the selected unattacked generation.

Selection uses a fixed seed but does not enforce generator or decoding balance.
The selected configuration is recorded so the realized sample can be audited.
If a selected family lacks any required attack, selection is repeated among the
source's complete families. A source with no complete family is excluded and
reported explicitly.

With all 14,971 sources retained, the expected sample is:

| Record type | Count |
|---|---:|
| Unique human documents | 14,971 |
| Selected unattacked machine documents | 14,971 |
| Attacked machine documents | 164,681 |
| All machine-condition rows | 179,652 |
| Human plus machine rows | 194,623 |

Human documents are stored once and reused across detector configurations and
attack comparisons. They are not physically duplicated twelve times.

For the current 13,371-source public labeled training release, the analogous
expected counts are 13,371 human rows, 160,452 machine-condition rows, and
173,823 total prepared rows. The validator accepts only this labeled-release
count or the 14,971-source archival paper count for a non-debug full run.

## Frozen source split

Assign source groups to disjoint splits using a fixed split seed and
domain-stratified random assignment:

| Split | Fraction | Approximate sources | Purpose |
|---|---:|---:|---|
| `clipping_tuning` | 40% | 5,988 | Fit orientation and one clipping specification per detector |
| `calibration` | 20% | 2,994 | Fit per-domain raw and clipped thresholds at 5% FPR |
| `test` | 40% | 5,989 | Final attack- and contamination-rate evaluation |

Every selected clean/attacked machine family stays with its human source in one
split. Split counts are resolved deterministically after the available source
set is known, and the exact per-domain counts are stored in the manifest.

## Realized contamination rate

For a selected unattacked machine document \(x\) and an attacked document
\(\widetilde{x}\), use the Falcon tokenizer and define

$$
\rho(x,\widetilde{x})
=
\frac{
d_{\mathrm{Lev}}\!\left(T(x),T(\widetilde{x})\right)
}{
\max\left(|T(x)|,|T(\widetilde{x})|\right)
}.
$$

Here \(T\) is the exact tokenizer/preprocessing path used for the Falcon
scorer and \(d_{\mathrm{Lev}}\) is token-level Levenshtein distance. The
distance is computed on the exact scored token windows after the frozen
truncation rule. The unattacked condition has \(\rho=0\); an attack that makes
no realized change also has \(\rho=0\).

For every machine row, store:

- clean and attacked scored-token counts;
- Levenshtein distance;
- minimum substitution, deletion, and insertion counts from one deterministic
  minimum edit script;
- normalized realized contamination rate \(\rho\);
- RAID's native attack parameter \(\theta\), which is the fraction of eligible
  attack operations and is not treated as document contamination rate;
- whether truncation hid any full-text edits from the scored window.

Contamination rate is descriptive. It never changes the fitted clipping
specification.

## Descriptive contamination-rate groups

Use only the `clipping_tuning` contamination-rate distribution to choose a
small set of interpretable reporting intervals. Give \(\rho=0\) its own group,
ensure every positive-rate interval has adequate tuning and anticipated test
counts, and freeze the cutpoints before calibration or test results are read.

The cutpoints are saved in the run manifest. For every final interval report
its sample count, median and interquartile range of \(\rho\), attack
composition, and selected generator/decoding composition. These intervals are
used only for tables and plots; the same universal clipped detector is applied
in every interval.

## Scoring context and models

All detectors score released RAID output text only. Generation prompts are not
prepended.

The six single-model detectors use `tiiuae/falcon-7b`:

1. log likelihood;
2. rank;
3. log rank;
4. DetectLLM LRR;
5. entropy;
6. entropy gap.

Binoculars uses its official default pair:

- observer: `tiiuae/falcon-7b`;
- performer: `tiiuae/falcon-7b-instruct`.

The raw Binoculars implementation must match the official tokenizer, special
token, truncation, model-revision, precision, and formula behavior. Falcon
single-model scoring uses the same frozen scored-text policy where the model
contracts permit it. Resolved model and tokenizer revisions are stored in the
manifest and score rows.

One Falcon forward pass supplies reusable token arrays for the six
single-model detectors. Binoculars stores the local arrays needed to reproduce
its raw score and documented clipped extension. The Falcon score pack also
retains full pre-truncation target-token IDs so the scored-window edit rate and
edits hidden beyond the 512-token boundary can be audited exactly.

## Seven raw detectors

The raw document scores are:

| Detector | Raw aggregation |
|---|---|
| Log likelihood | Mean observed-token log probability |
| Rank | Mean one-based competition rank |
| Log rank | Mean natural log of token rank |
| DetectLLM LRR | Mean NLL divided by mean log rank plus numerical epsilon |
| Entropy | Mean full-vocabulary predictive entropy |
| Entropy gap | Mean token NLL minus predictive entropy |
| Binoculars | Exponential of mean performer NLL minus mean observer-to-performer cross-entropy |

Raw aggregation is always retained as the paired scientific control.

## Orientation and universal clipping selection

For each detector, use only clean human and selected unattacked machine rows
from `clipping_tuning` to learn orientation \(d\), so larger oriented values
mean more machine-like. Freeze the orientation before calibration and test
data are used.

Construct candidate specifications from clean human and clean machine tuning
tokens using

$$
\mathcal Q
=
\{0.80,0.85,0.90,0.95,0.975,0.99,0.995\}.
$$

The no-clipping specification is the first candidate. Additive detectors and
Binoculars receive seven oriented lower-bound candidates. LRR receives all 49
pairs of NLL and log-rank upper caps.

For detector \(d\) and candidate specification \(c\), select one universal
specification with

$$
J_d(c)
=
0.8
\left[
\frac{1}{11}
\sum_{a=1}^{11}
\operatorname{AUROC}(H,M_a;c)
\right]
+
0.2\,\operatorname{AUROC}(H,M_{\mathrm{clean}};c).
$$

Every attack receives equal weight. The clean-machine component prevents a
candidate from being selected solely by sacrificing unattacked detection.
Ties within numerical tolerance retain the earlier candidate, and candidate
order begins with no clipping. The fitted orientation and one detector-specific
clipping specification are frozen for every domain, attack, contamination
rate, calibration row, and test row.

Detector-specific clipping follows `clipping_method.md` exactly:

- log likelihood, rank, log rank, entropy, and entropy gap use the generic
  oriented lower floor;
- LRR separately caps token NLL and token log rank before recomputing the ratio;
- Binoculars clips its oriented local gap before averaging and applying the
  exponential outer transform.

Official raw Binoculars is never replaced by its clipped extension.

## Calibration at 5% FPR

For every detector and domain, derive separate raw and clipped classification
thresholds from only the human documents in the `calibration` split. Use the
conservative empirical threshold whose achieved calibration FPR is closest to
but does not exceed 5%, following RAID's low-FPR evaluation principle.

Record the threshold, calibration-human count, achieved calibration FPR, and
held-out test FPR. There is no 1%-FPR analysis in this benchmark.

No orientation, clipping specification, contamination-rate cutpoint, or
classification threshold may use final-test outcomes.

## Final estimands and reports

### Attack-level analysis

For every detector and each of the twelve machine conditions, report raw and
clipped TPR at calibrated 5% FPR. The paired clipping effect is

$$
\Delta_{d,a}
=
\operatorname{TPR}^{\mathrm{clip}}_{d,a}
-
\operatorname{TPR}^{\mathrm{raw}}_{d,a}.
$$

The unattacked condition measures whether clipping sacrifices clean-machine
detection. Report all eleven RAID attacks; separately reproduce the subset and
ordering used in the published RAID attack table.

### Contamination-rate analysis

For each frozen reporting interval and detector, report:

- raw TPR at calibrated 5% FPR;
- universal-clipped TPR at calibrated 5% FPR;
- paired clipped-minus-raw TPR;
- raw and clipped AUROC as secondary diagnostics;
- sample, source, attack, and generator/decoding counts.

No rate-specific clipping bound is fitted or applied.

### Binoculars sanity check

Compare raw Binoculars with the RAID paper's published 5%-FPR attack row. The
retained experiment uses one sampled machine family per source and an
independent 20% calibration subset, so exact equality is not expected. Report
published values beside the new point estimates and confidence intervals,
along with achieved held-out FPR. The broad published pattern, especially the
large synonym and homoglyph degradation, is the preregistered qualitative
sanity check.

## Paired source-cluster bootstrap

Use 2,000 repetitions and a fixed bootstrap seed. Within each domain, resample
test `source_id` values with replacement. A sampled source carries its selected
unattacked generation and all eleven attacks together. Use the same sampled
source indices for raw and clipped scores so all differences remain paired.

Orientation, contamination-rate cutpoints, clipping specifications, and
calibration thresholds remain frozen. The bootstrap quantifies final-test
sample uncertainty, matching the repository's primary-study convention.
Report percentile 95% confidence intervals for raw TPR, clipped TPR,
clipped-minus-raw TPR, held-out FPR, AUROC, and contamination-interval
summaries where applicable.

## Artifact layout and provenance

The authoritative entry point is `RAID/run_raid.py`. A run with ID `<run-id>`
uses:

```text
runs/raid/<run-id>/
  manifest.json
  selected_sources.jsonl
  data.jsonl
  falcon_scores.jsonl
  binoculars_scores.jsonl
  frozen_specs.json
  *.complete.json

results/raid/<run-id>/
  metrics.csv
  contamination_summary.csv
  attack_summary.csv
  binoculars_sanity.csv
  contamination_records.csv
  validation_report.json
  plots/
```

Every material change to source selection, split seeds, model/tokenizer
revision, scored context, detector formula, contamination-rate definition,
clipping objective, quantile grid, calibration rule, target FPR, bootstrap
seed, or reporting cutpoints requires a new run ID.

Preparation, Falcon scoring, Binoculars scoring, evaluation, plotting, and
validation are independently restartable stages. Append-safe score files use
stable row keys and completion markers. Manifests record resolved revisions,
Git state, software, accelerator, Slurm, paths, seeds, counts, exclusions, and
stage status.

### Parallel execution invariance

Parallelization is operational and does not change any scientific sample,
score, clipping rule, threshold, or metric. After the source split is frozen,
all 13 rows belonging to a source are assigned to the same score shard by
`stable_int("raid-score-shard-v1", source_id) % num_shards`. Falcon and
Binoculars shards are scored independently. The merge stage rejects missing,
extra, or duplicate stable row keys and rejects model-provenance differences
across shards before evaluation begins. Preparation and final evaluation run
once globally; clipping is never fitted separately by shard.

The official CSV relationship index is cached by the CSV SHA-256 digest.
Reusing a completed cache changes only preparation time. An explicitly adopted
index from an interrupted run is permitted only for bounded debug validation;
a full scientific run requires a completed checksum-keyed cache marker.

## Validation contract

A scientifically complete full run must satisfy all of the following:

1. the manifest reports `completion_status: complete`;
2. completed stages are `prepare`, `score`, `evaluate`, `plot`, and `validate`;
3. source groups are disjoint across 40/20/40 splits and all variants of a
   source remain together;
4. every retained source has one human row, one selected unattacked machine
   row, and eleven linked attack rows;
5. expected row counts, stable unique keys, scorer revisions, and score schemas
   pass validation;
6. all seven detectors have raw and universal-clipped results;
7. only 5%-FPR rows are present, with separate per-domain raw and clipped
   thresholds and held-out FPR;
8. every final metric contains source counts and 2,000-repetition paired
   bootstrap intervals;
9. contamination summaries use cutpoints frozen from tuning rates and never
   fit rate-specific clipping;
10. raw Binoculars sanity output contains the RAID-published comparator values;
11. stderr contains no traceback, CUDA failure, silent CPU/offload surprise,
    missing attack family, or truncation/schema mismatch;
12. large data and score packs resolve below `/work/hdd` on Delta.

Smoke and bounded runs must be labeled `debug_only` and cannot be reported as
scientific results.

## Delta execution contract

The Delta orchestrator is `RAID/submit_raid.sh`. It submits a CPU preparation
job, Falcon and Binoculars score arrays, and a dependency-gated CPU finalize
job. Before any full launch:

```bash
cd ~/LLM-detection
readlink -f runs
readlink -f results
df -h /work/hdd/bhuc/$USER
```

Both data paths must resolve below `/work/hdd`. Use the existing CUDA 12.8
environment at `/projects/bhuc/jli101/venvs/delta-smoke` and shared caches under
`/work/hdd/bhuc/jli101/llm-detection-smoke/`. The workflow must accept an
explicit run ID, bounded source limit, bootstrap override, and shard count;
require a two-GPU-per-Binoculars-shard smoke gate before a full run; and never store full
JSONL score packs under `/u`.

Slurm `COMPLETED` is necessary but insufficient. After every launch, inspect
the Slurm exit code and stderr, then run `RAID/validate_raid.py` and inspect the
manifest, stage markers, row counts, detector/configuration counts, target FPR,
held-out FPR, and validation report.

## Interpretation boundaries

RAID attacks are heterogeneous. RAID's native \(\theta\) is the fraction of
eligible operations, not the theoretical contamination probability or realized
document edit rate. The normalized Falcon-token edit rate \(\rho\) is an
observed sequence-modification measure, not an estimate of a latent adversarial
probability.

Paraphrasing is global rewriting and high-rate character/tokenizer attacks may
not resemble sparse human editing. Attack-level and contamination-rate results
must therefore both be shown. A positive clipping effect establishes improved
robust aggregation under the measured RAID modification, not a
human-editing-specific mechanism.
