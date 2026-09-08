# RAID external-contamination benchmark

## Status and scope

This document freezes the scientific protocol for the RAID external benchmark.
It is separate from the 21-cell controlled contamination study and the Beemo
realistic-edit benchmark. RAID is used to test whether the same one-sided
token-level clipping intervention remains useful under naturally heterogeneous,
attack-generated contamination.

The primary method family fits two universal clipping specifications per
detector. The **full-universal** specification uses all eleven attacked
conditions. The **eligible-universal** specification uses only attacked tuning
rows with \(0<\rho\leq0.50\). Both produce one deployable bound per detector;
neither needs the attack name or contamination rate at test time. A secondary
rate-oracle analysis fits one additional detector-specific
specification in each of four fixed realized-contamination intervals:
\(0<\rho\leq0.05\), \(0.05<\rho\leq0.10\),
\(0.10<\rho\leq0.20\), and \(0.20<\rho\leq0.50\). Rows with
\(\rho=0\) or \(\rho>0.50\) remain in the dataset and universal attack
analysis but are excluded from rate-oracle bin fitting and attacked-bin
reporting. Unattacked `none` rows are reused only for the clean-loss constraint
and the counterfactual clean-cost report. There is no attack-specific clipping.

The detector transformations are defined in [`../docs/methods/clipping_method.md`](../methods/clipping_method.md).
For additive detectors and Binoculars, one specification is a single oriented
lower bound. For DetectLLM LRR, one specification is the pair of upper caps on
token NLL and token log rank. A single numerical bound is not shared across
detectors because their local evidence and scales differ.

## Research questions

1. Does raw Binoculars on the retained RAID sample show the same broad attack
   pattern as the published RAID benchmark at 5% FPR?
2. Do full-universal and eligible-universal clipping improve TPR at 5% FPR
   for the seven detector methods under RAID attacks?
3. How do the two universal methods and rate-oracle clipping compare within four fixed,
   practically interpretable realized-contamination intervals?
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

The completed 500-source pilot was used to choose which fitting scopes to carry
forward. Its exact source IDs are therefore development data. Before the final
source split is made, all 500 IDs must be excluded from the available source
universe. The exclusion JSON's absolute path, SHA-256 digest, byte size, and
source count are stored in the run manifest; preparation records all applied
IDs; and validation requires exact application and zero overlap. The RAID CSV,
checksum-keyed relationship index, pilot score packs, and pilot results are
preserved unchanged.

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

Before development-source exclusion, the current 13,371-source public labeled
training release contains 13,371 human rows, 160,452 machine-condition rows,
and 173,823 total prepared rows. After excluding the 500 pilot sources, the
expected final labeled-release sample is 12,871 sources, 154,452
machine-condition rows, and 167,323 total prepared rows. An archival
14,971-source snapshot would analogously retain 14,471 sources. The validator
accepts only these post-exclusion counts for an unbounded run.

## Frozen source split

Assign source groups to disjoint splits using a fixed split seed and
domain-stratified random assignment:

| Split | Fraction | Approximate sources | Purpose |
|---|---:|---:|---|
| `clipping_tuning` | 40% | 5,148 | Fit orientation, both universal bounds, and four rate-oracle specifications per detector |
| `calibration` | 20% | 2,574 | Fit per-domain thresholds for every raw or clipped configuration at 5% FPR |
| `test` | 40% | 5,149 | Final attack- and contamination-rate evaluation |

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

Contamination rate is descriptive for the universal analysis and selects a
fixed-bin clipping specification only in the explicitly labeled rate-oracle
analysis.

## Fixed contamination-rate groups

The rate-oracle intervals are fixed a priori at
\(0<\rho\leq0.05\), \(0.05<\rho\leq0.10\),
\(0.10<\rho\leq0.20\), and \(0.20<\rho\leq0.50\). They are not
estimated from tuning quantiles. Rows with \(\rho=0\) and \(\rho>0.50\) are
excluded from attacked-bin fitting and evaluation. For every interval report its tuning and
test counts, median and interquartile range of \(\rho\), attack composition,
and selected generator/decoding composition.

Fit one clipping specification per detector and fixed interval. Let
\(A_b\) be the attacks represented in interval \(b\). For a candidate \(c\),
maximize the equally weighted mean attacked AUROC gain over raw:

$$
G_b(c)
=
\frac{1}{|A_b|}
\sum_{a\in A_b}
\left[
\operatorname{AUROC}(H,M_{a,b};c)
-
\operatorname{AUROC}(H,M_{a,b};\mathrm{raw})
\right],
$$

subject to the clean-machine constraint

$$
\operatorname{AUROC}(H,M_{\mathrm{none}};c)
\geq
\operatorname{AUROC}(H,M_{\mathrm{none}};\mathrm{raw})-0.01.
$$

No clipping is the first feasible candidate, so ties retain no clipping. This
allows a small, prespecified clean loss rather than rewarding clean performance
inside the rate-specific objective. If an interval has fewer than 250 tuning
machine rows, use the already-frozen universal specification and record the
fallback; never move the fixed boundaries.

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

For detector \(d\) and candidate specification \(c\), define

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

For **full-universal** fitting, the mean contains all eleven attacks. For
**eligible-universal** fitting, it contains every attack represented among
tuning rows with \(0<\rho\leq0.50\), still with equal weight per represented
attack rather than per row. In both scopes, the clean-machine component prevents
a candidate from being selected solely by sacrificing unattacked detection.
Ties within numerical tolerance retain the earlier candidate, and candidate
order begins with no clipping. Both detector-specific universal specifications
are frozen for every domain and attack. The four rate-oracle specifications are
separately frozen and used only for test rows in their corresponding fixed
intervals.

Detector-specific clipping follows `docs/methods/clipping_method.md` exactly:

- log likelihood, rank, log rank, entropy, and entropy gap use the generic
  oriented lower floor;
- LRR separately caps token NLL and token log rank before recomputing the ratio;
- Binoculars clips its oriented local gap before averaging and applying the
  exponential outer transform.

Official raw Binoculars is never replaced by its clipped extension.

## Calibration at 5% FPR

For every detector, domain, and clipping specification, derive separate raw,
full-universal, eligible-universal, and rate-oracle classification thresholds
from only the human documents in the `calibration` split. Use the
conservative empirical threshold whose achieved calibration FPR is closest to
but does not exceed 5%, following RAID's low-FPR evaluation principle.

Record the threshold, calibration-human count, achieved calibration FPR, and
held-out test FPR. There is no 1%-FPR analysis in this benchmark.

No orientation, clipping specification, fixed contamination-rate boundary, or
classification threshold may use final-test outcomes.

## Final estimands and reports

### Attack-level analysis

For every detector and each of the twelve machine conditions, report raw,
full-universal, and eligible-universal TPR at calibrated 5% FPR. For either
universal method \(u\), the paired clipping effect is

$$
\Delta_{d,a,u}
=
\operatorname{TPR}^{u}_{d,a}
-
\operatorname{TPR}^{\mathrm{raw}}_{d,a}.
$$

The unattacked condition measures whether clipping sacrifices clean-machine
detection. Report all eleven RAID attacks; separately reproduce the subset and
ordering used in the published RAID attack table.

### Contamination-rate analysis

For each frozen reporting interval and detector, report:

- raw TPR at calibrated 5% FPR;
- full-universal-clipped TPR at calibrated 5% FPR;
- eligible-universal-clipped TPR at calibrated 5% FPR;
- rate-oracle-clipped TPR at its separately calibrated 5% FPR;
- paired full-universal-minus-raw, eligible-universal-minus-raw, and
  rate-oracle-minus-raw TPR;
- raw, both universal-clipped, and rate-oracle-clipped AUROC as secondary diagnostics;
- sample, source, attack, and generator/decoding counts.

For every fitted interval bound, also report a trade-off table that applies the
same frozen bound and its corresponding calibration threshold to: (i) all
unattacked `none` test rows as a counterfactual clean-cost check; (ii) all
attacked test rows in that interval; and (iii) every attack represented in that
interval. This table reports paired bound-minus-raw TPR and AUROC with source
bootstrap intervals. It does not fit attack-specific bounds.

Because the correct rate interval is supplied from the clean/attacked pair,
the rate-oracle is a diagnostic upper-bound analysis, not a directly deployable
detector. The universal methods remain the primary deployable methods.

### Binoculars sanity check

Compare raw Binoculars with the RAID paper's published 5%-FPR attack row. The
retained experiment uses one sampled machine family per source and an
independent 20% calibration subset, so exact equality is not expected. Report
published values beside the new point estimates and confidence intervals,
along with achieved held-out FPR. The broad published pattern, especially the
large synonym and homoglyph degradation, is the preregistered qualitative
sanity check.

## Paired source-cluster bootstrap

The final confirmatory result uses 2,000 repetitions and a fixed bootstrap
seed. The first unbounded diagnostic run may use 500 repetitions, but must be
labeled `debug_only` and provisional; it is used to inspect full-data behavior
before paying the additional evaluation cost. Within each domain, resample
test `source_id` values with replacement. A sampled source carries its selected
unattacked generation and all eleven attacks together. Use the same sampled
source indices for raw and clipped scores so all differences remain paired.

Orientation, fixed contamination-rate boundaries, clipping specifications, and
calibration thresholds remain frozen. The bootstrap quantifies final-test
sample uncertainty, matching the repository's primary-study convention.
Report percentile 95% confidence intervals for raw TPR, clipped TPR,
clipped-minus-raw TPR, held-out FPR, AUROC, and contamination-interval
summaries where applicable.

The 2,000-repetition result is an evaluation-only promotion of the accepted
500-repetition run. It receives a new result ID while reusing exactly the same
prepared rows and Falcon/Binoculars token-feature packs. Its promotion manifest
records the source run, source and promotion Git commits, frozen configuration,
development exclusions, and the paths, sizes, modification times, and marker
hashes of the reused artifacts. It may rerun only evaluation, plotting, and
validation; it must not prepare data, load a scorer, or perform model inference.

Inspecting the 500-repetition intervals before promotion makes the final pass a
Monte Carlo refinement of the already accepted estimands, not a new independent
confirmation or another opportunity to change the method. Point estimates,
orientation, clipping bounds, calibration thresholds, source splits, and the
bootstrap seed remain fixed. Only the number of deterministic bootstrap draws
increases from 500 to 2,000. The promotion must compare the new frozen
specifications, thresholds, and every non-interval CSV value with the source
result and fail if any value changes.

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
  rate_bound_tradeoff_summary.csv
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
Completed bounded prepared artifacts may be adopted by a corrected scoring run
only after exact validation of dataset provenance, selection/split settings,
source count, row keys, shard count, and shard assignments. The new manifest
records the source run and Git commit; no scores are adopted.

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
6. all seven detectors have raw, full-universal-clipped, and
   eligible-universal-clipped results, plus four fixed-bin rate-oracle
   specifications;
7. only 5%-FPR rows are present, with separate per-domain raw and clipped
   thresholds and held-out FPR;
8. every confirmatory final metric contains source counts and 2,000-repetition paired
   bootstrap intervals;
9. contamination summaries contain only the four fixed intervals through
   \(\rho=0.50\), fit no attack-specific rule, and record every sparse-bin
   fallback to universal clipping;
10. every fixed bound has counterfactual `none`, attacked-bin aggregate, and
    represented-attack trade-off rows with paired intervals;
11. raw Binoculars sanity output contains the RAID-published comparator values;
12. stderr contains no traceback, CUDA failure, silent CPU/offload surprise,
    missing attack family, or truncation/schema mismatch;
13. all 500 recorded development sources are excluded before splitting, and
    validation confirms zero overlap;
14. large data and score packs resolve below `/work/hdd` on Delta.

Smoke and bounded runs must be labeled `debug_only` and cannot be reported as
scientific results.

## Exploratory selector comparison before the full run

The completed 500-source bounded pilot may be reused for CPU-only selector
development. This analysis is archived in `archive/raid_pilots/compare_tuning_methods.py`
(relative to the repository root); it does
not amend the frozen RAID evaluator or overwrite its artifacts. It compares
three fitting scopes:

1. full universal, using every realized attacked rate;
2. eligible-range universal, using only \(0<\rho\leq0.50\);
3. a four-bin rate-specific oracle using the fixed intervals above.

The universal scopes compare mean attack AUROC gain, worst-attack AUROC gain,
rate-balanced AUROC gain, and cross-fitted TPR gain at 5% FPR. The rate oracle
compares the same objectives except rate balancing, which is redundant inside
one fixed interval. This gives 11 unique tuning methods and 12 reported
approaches after adding raw.

Candidate bounds use the exploratory grid

$$
\{0.50,0.60,0.70,0.80,0.85,0.90,0.925,0.95,
0.975,0.99,0.995,0.999\}.
$$

Every selector is examined under clean-machine AUROC-loss budgets of \(0\),
\(0.01\), \(0.02\), and \(0.05\). Five-fold cross-fitting is source-grouped and
domain-stratified within the clipping-tuning split. Final point estimates still
use the untouched calibration humans to derive separate per-domain 5%-FPR
thresholds and use only the pilot test split for comparison.

Rows with \(\rho=0\) and \(\rho>0.50\) participate in full-universal tuning.
They do not participate in eligible-universal or rate-oracle tuning, but every
universal bound is evaluated on them. The rate oracle is evaluated only inside
its four eligible intervals and remains non-deployable because it requires the
counterfactual clean/attacked pair to identify the interval.

These comparisons contain no bootstrap confidence intervals and are explicitly
labeled `PILOT_DEBUG_NOT_FOR_FINAL_REPORTING`. The post-pilot decision freezes
the original seven-quantile grid and the original objective in Equation
\(J_d(c)\) for both full-universal and eligible-universal fitting. Exploratory
cross-fitted TPR selectors, expanded quantile grids, clean-loss-budget searches,
trimmed aggregation, and component-wise Binoculars clipping are not promoted to
the final evaluator. Test outcomes from the unbounded run may not alter this
choice.

Because selector and budget choice will inspect pilot test outcomes, every
source used by the 500-source pilot becomes development data. The comparison
writes their IDs to `development_source_ids.json`. The full benchmark must
exclude every one of those IDs before making its tuning/calibration/test split
and must validate zero overlap. This preserves an independent final test while
making the pilot a legitimate model-selection experiment.

## Exploratory one-sided trimmed aggregation

The bounded pilot also tests a single alternative to clipping: remove a fixed
fraction of the most adverse token evidence. For each additive detector, orient
the local evidence so larger values are more machine-like, sort it as
\(z_{(1)}\leq\cdots\leq z_{(n)}\), set \(k=\lfloor\alpha n\rfloor\), and use

$$
S_{\mathrm{trim},\alpha}
=
\frac{1}{n-k}\sum_{i=k+1}^{n}z_{(i)}.
$$

The candidate grid is

$$
\alpha\in\{0,0.005,0.01,0.025,0.05,0.10,0.20\}.
$$

Binoculars applies the same trim to its oriented token-level evidence and then
retains its official outer exponential transformation. LRR is not additive.
Its exploratory extension therefore removes paired NLL/log-rank tokens ranked
by the exact oriented improvement from leaving each token out, then recomputes
the ratio on retained pairs. This is labeled
`paired_leave_one_out_influence_trim` and is not described as an additive
trimmed mean.

Two universal fractions are learned per detector: full universal uses all 11
attacked families at every realized rate; eligible universal uses only attacked
rows with \(0<\rho\leq0.50\). Both use the original selection objective

$$
0.8\,\operatorname{mean}_{a}\operatorname{AUROC}_{a}
+0.2\,\operatorname{AUROC}_{\mathrm{clean}},
$$

with represented attacks equally weighted. Full-universal tuning requires all
11 attacks; eligible-universal tuning omits attack families with no realized
rows in its eligible interval. Exact ties retain the smaller fraction, so
\(\alpha=0\) wins a tie. The untouched calibration humans separately calibrate
raw and trimmed per-domain thresholds to 5% FPR, and the pilot test split is
reported by attack and fixed contamination-rate interval. AUROC is primary;
TPR and held-out FPR are diagnostics because the 500-source calibration split
is small. There is no bootstrap. Outputs are labeled
`PILOT_DEBUG_NOT_FOR_FINAL_REPORTING`, do not replace frozen clipping results,
and record the same development source exclusions.

## Exploratory Binoculars component clipping

The frozen Binoculars extension clips the combined oriented local gap
\(d(a_i-b_i)\), where \(a_i\) is performer NLL and \(b_i\) is
observer-to-performer cross-entropy. A separate pilot ablation compares this
with component-wise clipping:

$$
\widetilde z_i
=
\max\{d a_i,\ell_a\}
+
\max\{-d b_i,\ell_b\}.
$$

The oriented document score remains

$$
d\exp\!\left(d\,\frac{1}{n}\sum_i\widetilde z_i\right).
$$

Raw Binoculars is the first candidate. Gap clipping and component clipping each
use the frozen seven-element quantile grid. For component clipping, the same
quantile index determines \(\ell_a\) and \(\ell_b\) from their respective clean
tuning distributions. The experiment deliberately does not search all 49
quantile pairs. Each family is fitted under full-universal and
eligible-universal scopes with the original \(0.8\) mean-attack AUROC plus
\(0.2\) clean AUROC objective. Exact ties retain raw; ties between two nonraw
candidates retain the less aggressive bound.

This analysis reuses the bounded pilot score packs, has no bootstrap, and is
labeled `PILOT_DEBUG_NOT_FOR_FINAL_REPORTING`. Component clipping can disrupt
the common-mode cancellation built into the raw difference \(a_i-b_i\), so it
is an ablation rather than a presumed improvement. It does not replace official
raw Binoculars or the frozen gap-clipping protocol.

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
# Versioned detector amendment

The revised eight-detector analysis is specified in
[../docs/DETECTOR_REVISION.md](../DETECTOR_REVISION.md). It preserves the legacy
exponential-gap result and adds an independently scored official-style ratio;
old outputs must not be silently relabeled as the published Binoculars ratio.
