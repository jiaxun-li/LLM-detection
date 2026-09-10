# RAID external-contamination benchmark: released analysis

## 1. Scope, evidence, and version precedence

This document describes the corrected RAID analysis in the downloaded final
nine-primary-cell-plus-RAID bundle. It is a methods and interpretation reference,
not evidence of an external preregistration. Repository protocol freezing and
later documented corrections must not be described as prospective registration
of every final analysis. The broader 21-cell primary design and the archived
Beemo/splice studies are separate historical work.

The final RAID revision is
`raid-origin-anchored-v41-20260907T044156Z`, with implementation label
`official-score-anchored-clipping-v4.1`. It reuses the prepared sample and
single-model features from
`raid-full-excluded500-bootstrap500-provisional-20260824T132810Z`, adds a
separately scored official-style Binoculars ratio, and evaluates 2,000 bootstrap
replicates. The old exponential-gap detector remains in the exact eight-detector
provenance result but is excluded from the seven-detector paper view.

The local release root, relative to the repository, is:

```text
downloads/current/primary-nine-plus-raid-final-20260908T150247Z/raid/
  raid_export_summary.json
  source_eight_detector_result/   # exact validated result, including gap
  paper_seven_detector_result/    # filtered CSVs and regenerated plots
```

Evidence is the exported revision manifest, validation report, evaluation counts,
frozen specifications, and export summary, checked against active code. Historical
module names in result hashes/paths are provenance and must not be rewritten after
a repository reorganization.

The base [configs/raid.json](../../configs/raid.json) still contains the legacy
`binoculars` identifier. **The base pipeline alone does not reproduce the
corrected final detector set.** [RAID/revise_detectors.py](../../RAID/revise_detectors.py)
opts into `binocular-origin-lrr-constant-v1`. See the
[detector amendment](../DETECTOR_REVISION.md) and [Delta guide](DELTA_GUIDE.md).

Active code is now v4.2 because of the positive LRR denominator-cap safeguard.
The RAID result remains v4.1: its recorded audit found no zero-cap LRR candidate,
and all six selected LRR specifications have positive denominator caps. Do not
relabel the saved result as a v4.2 execution.

## 2. Question and outcomes

The experiment asks whether clipping token evidence improves document-level
machine-origin detection under heterogeneous RAID attacks. It does not assume
these attacks are human edits, sparse edits, or exact draws from the theoretical
contamination model.

The primary comparison is paired raw versus clipped TPR at **calibrated target
FPR 5%**, with achieved held-out human FPR reported beside it. AUROC is the
clipping-selection criterion and a secondary final diagnostic. There is no
1%-FPR RAID analysis in this release.

Two primary scopes are reported for every detector:

- **Full universal:** one specification fitted using all eleven attacked tuning
  conditions, including realized rates zero and above 0.5.
- **Eligible universal:** one specification fitted using attacked tuning rows
  with \(0<\rho\leq0.5\).

Both can be applied without the attack name or clean counterpart, although the
classifier uses a domain-specific threshold. Four **rate-specific oracle**
bounds are secondary diagnostics. Oracle means the paired rate interval is
supplied, not a mathematical upper bound or guaranteed superior performance.

## 3. Data, sampling, and independent unit

The configuration identifies the public labeled English RAID `train.csv`
release, with domains abstracts, books, news, poetry, recipes, reddit, reviews,
and wiki. The repository recognizes 13,371 labeled-release sources and 14,971
for the historical full-paper universe. **The released experiment uses the
former, not all 14,971 paper sources.** Common dataset naming does not establish
exact sample identity with the paper.

The independent sampling, splitting, and bootstrap unit is `source_id`.
Each retained source contributes 13 rows: one original human document, one
selected clean machine generation (`none`), and these eleven attacked variants:

```text
alternative_spelling  article_deletion  homoglyph  insert_paragraphs
number  paraphrase  perplexity_misspelling  synonym  upper_lower
whitespace  zero_width_space
```

`RAID/raid_data.py::select_complete_families` first identifies complete valid
families: exactly one human row, one row per required attack, correct
`adv_source_id` linkage, and matching model/decoding/repetition-penalty/domain
metadata. Sources without a valid complete family are recorded as excluded.
Clean candidates are sorted by RAID ID, then one complete family is selected
uniformly using
`random.Random(stable_int(selection_seed, source_id)).randrange(...)`.
Selection seed is **260823**. Generator/decoding configurations are not balanced;
the implementation does not require exactly 34 complete candidates per source.

The 500 pilot sources are excluded before splitting because pilot test outcomes
were inspected for selector development. The final revision manifest records
500 exclusions, byte size 22,172, and SHA-256
`02e46d83e47a1decfe6a268190042bc0e9c2ff5e530bf6633bb1f9337299dc08`.
The raw release and complete prepared records remain on Delta. The downloaded
bundle does not independently re-establish every linkage or exclusion ID.

| Released quantity | Count |
|---|---:|
| Retained source groups / human rows | 12,871 |
| Machine rows including `none` | 154,452 |
| Attacked machine rows only | 141,581 |
| Total prepared / scorer rows | 167,323 |
| Test machine documents per attack or `none` | 5,149 |

Humans are stored once, not duplicated for each attack. Reusing the same human
controls in different metrics does not create independent observations.

### Source-grouped split

`_split_assignments` uses split seed **260824**, hash ordering within domain,
and domain-stratified integer allocation. Global tuning/calibration counts are
floored at 40%/20%; test receives the remainder. All 13 rows stay together.

| Split | Sources | Prepared rows | Purpose |
|---|---:|---:|---|
| `clipping_tuning` | 5,148 | 66,924 | Direction, candidate caps, fitting |
| `calibration` | 2,574 | 33,462 | Only human rows calibrate thresholds |
| `test` | 5,149 | 66,937 | Frozen-rule evaluation and intervals |

The revision validation report's `splits` values are **row counts**, not source
counts. Machine calibration rows retain complete families but do not fit caps
or thresholds.

## 4. Models, context, and precision

Released RAID output text is scored without prepending generation prompts or
regenerating text. The six single-model methods share Falcon-7B token features.
Binoculars uses Falcon-7B as observer and Falcon-7B-Instruct as performer.

| Final ratio component | Resolved model/tokenizer revision |
|---|---|
| Observer | `ec89142b67d748a1865ea4451372db8313ada0d8` |
| Performer | `8782b5c5d8c9290412416618f36a133653e85285` |

The ratio scorer uses native Transformers Falcon (`trust_remote_code=false`),
BF16, tokenizer-native special tokens, right truncation to **512 input tokens**,
batch size one, and separate observer/performer devices. These resolved manifest
values, not null revisions in reusable config, describe the final ratio run.

`RAID/raid_scoring.py::encode_output_only_window` retains at most 512 input
IDs and scores causal targets from position 2 onward: at most **511 target
tokens**. The first token is context. Full pre-truncation target IDs are saved.
The shared Falcon path inserts and records BOS/EOS context only for an otherwise
one-token document. The official-origin path rejects an unscorable one-token
window instead. Full-input tokenizer preflight and an integrated
upstream-component probe gate this separate scorer.

## 5. Raw definitions and orientation

Write \(\ell_i=\log p_i(x_i)\), \(a_i=-\ell_i\),
\(r_i=1+\#\{v:\operatorname{logit}_i(v)>\operatorname{logit}_i(x_i)\}\),
\(q_i=\log r_i\), and \(h_i=-\sum_vp_i(v)\log p_i(v)\).
Logarithms are natural. Competition ranks assign equal rank to tied logits;
this is not exact equivalence to every upstream sorted-position tie policy.

| Reported ID | Raw aggregation | Direction |
|---|---|---|
| `log_likelihood` | \(\operatorname{mean}\ell_i\) | Learned |
| `rank` | \(\operatorname{mean}r_i\) | Learned |
| `log_rank` | \(\operatorname{mean}q_i\) | Learned |
| `lrr` | \(\operatorname{mean}a_i/(\operatorname{mean}q_i+10^{-12})\) | Fixed +1 |
| `entropy` | \(\operatorname{mean}h_i\) | Learned |
| `entropy_gap` | \(\operatorname{mean}(a_i-h_i)\) | Learned |
| `binocular_origin` | Official mean-NLL / mean-cross-entropy | Fixed -1 |

For learned directions, `learn_direction` chooses +1 when the mean clean-machine
document score is at least the mean human document score, otherwise -1, using
only clean tuning documents. It does not optimize direction on attack/test
AUROC. Stored directions are +1 for log likelihood and LRR, -1 for all other
reported methods. Higher oriented scores predict machine origin.

LRR is DetectLLM's likelihood/log-rank ratio, **not** a log-likelihood ratio of
explicit human versus machine distributions. This distinction matters when
relating its empirical behavior to likelihood-ratio clipping theory.

### Official raw Binoculars and historical gap

`RAID/binocular_origin.py::official_components` calculates performer NLL on
causally shifted targets with the shifted attention mask. Its denominator is
observer-to-performer cross-entropy over **all input positions**, masked by
`input_id != pad_token_id`. When pad and EOS coincide, those EOS positions are
excluded from the denominator. The numerator does not use this ID-based mask.

Official-style BF16 tensor reductions yield saved numerator \(A_0\) and
denominator \(B_0\), followed by float32 division. Raw scores are preserved,
not reconstructed from serialized token-array means. The upstream gate tests
components on the **same logits**, not independent end-to-end reproduction of
the complete upstream model-loading, sample-selection, and threshold pipeline.

The retained `binocular_gap` diagnostic is
\(\exp(\operatorname{mean}(a_i-b_i))\) on historical aligned features.
It was called `binoculars` in the base pipeline. The exponential is **not**
the official Binoculars formula. It is omitted from the paper-facing view.

## 6. Clipping and candidate generation

Candidates come from pooled human and clean-machine **tuning tokens**.
Concatenation gives equal token weight: longer documents contribute more.
Attacked documents select candidates but do not set quantile values. The grid is

$$
\mathcal Q=\{0.80,0.85,0.90,0.95,0.975,0.99,0.995\}.
$$

Every family starts with raw/no clipping (`{}`).

- Five additive detectors orient their token evidence \(z_i\) and use
  \(\operatorname{mean}\max(z_i,L)\). A grid value \(q\) gives the pooled
  oriented-token quantile at \(1-q\) as floor \(L\): seven floors plus raw.
- LRR caps NLL at \(U_a\), log rank at \(U_q\), then recomputes
  \(\operatorname{mean}\min(a_i,U_a)/
  (\operatorname{mean}\min(q_i,U_q)+10^{-12})\).
  Its grid is all 49 independent NLL/log-rank quantile pairs plus raw.
- Official-origin Binoculars uses seven performer-NLL upper quantiles plus raw.
  The denominator is not clipped.
- The legacy gap floors its oriented local gap and keeps its own exponential.
  It is not substituted for the official ratio numerator.

### Official-score-anchored numerator clipping

Serialized BF16 losses do not encode CUDA reduction order. The released RAID
extension anchors a float64 clipping delta to the saved official numerator:

$$
\delta_U=\operatorname{mean}_i\min(a_i,U)-\operatorname{mean}_i a_i,\qquad
A_U=\min\{A_0,\max(0,A_0+\delta_U)\},\qquad
B_U=A_U/B_0.
$$

Final division uses float32 conversion. If no loss exceeds \(U\), the stored raw
score is returned exactly. This is a documented numerator-clipping extension,
not an upstream published method and not simply BF16 replay of the capped mean.
It differs from the primary study's prompt-conditioned saved-feature ratio.
See `experiment_core/detectors/detector_revision.py::origin_score` and
`anchored_nll_mean`.

### Candidate safeguards

The revised evaluator rejects a nonempty candidate whose pooled clean tuning
document scores are exactly identical, or whose relevant contributions are all
structurally saturated. No tolerance or calibration/test observations determine
that guard. An origin numerator that is constant can remain informative through
a varying denominator and is not rejected by the structural numerator check.
Current v4.2 code also rejects nonpositive LRR log-rank caps; the saved RAID
candidates required no rerun under that later guard.

## 7. Full and eligible universal tuning

For either fitting scope \(u\), maximize

$$
J_u(c)=0.8\frac{1}{|\mathcal A_u|}
\sum_{a\in\mathcal A_u}\operatorname{AUROC}(H,M_{a,u};c)
+0.2\operatorname{AUROC}(H,M_{\mathrm{none}};c).
$$

All samples here are tuning data. The same human and clean-machine documents
are used in both scopes. Full universal requires eleven nonempty attacks.
Eligible universal filters attacked rows to \(0<\rho\leq0.5\) and omits empty
attack groups. The final eligible support contains ten attacks: all configured
attacks except `zero_width_space`. It includes the eligible subsets of
homoglyph and paraphrase; attack medians alone do not determine inclusion.

Attacks receive equal objective weight regardless of row count. AUROC pools
document scores across domains. The 0.2 clean term is a **soft preference, not
a clean-loss constraint**. Universal clipping can decrease clean AUROC or TPR.
The final selector is not the pilot's zero-loss-budget or cross-fitted-TPR
selector.

Candidate improvements must exceed \(10^{-12}\); ties retain earlier enumeration
order. Raw wins an optimal tie when it shares that optimum, not whenever any
two clipped candidates happen to tie.

### Secondary fixed-rate oracle

Intervals are \((0,0.05]\), \((0.05,0.10]\), \((0.10,0.20]\), and
\((0.20,0.50]\), chosen during development and fixed before the final split.
For each bin, maximize equal-represented-attack mean AUROC gain over raw subject
to clean tuning AUROC loss at most **0.01**. That is an AUROC constraint, not a
TPR constraint or a guarantee on held-out losses.

If a bin contains fewer than **250 tuning machine rows**, fall back to the
**full-universal** specification. This is not a unique-source or per-attack
minimum; represented attacks can have few rows. The final report records zero
fallbacks. No attack-specific specification is fit.

Zero-rate and above-0.5 rows remain in universal attack-level reporting but not
the four bin fits/summaries. All clean `none` test rows are separately tested
under each oracle bound for the counterfactual clean-cost report.

## 8. Realized contamination rate and limitations

Let \(T_s\) denote the Falcon **scored-target-token** window, excluding the first
input context token. The recorded rate is

$$
\rho(x,\widetilde x)=
\frac{d_{\mathrm{Lev}}(T_s(x),T_s(\widetilde x))}
{\max(|T_s(x)|,|T_s(\widetilde x)|)}.
$$

`measure_realized_contamination` uses unit-cost Levenshtein alignment, so
insertions/deletions do not count every shifted later position as a replacement.
Shared ends are trimmed before RapidFuzz `editops`. S/D/I are counts from
**one minimum-cost edit script**, not three separately minimized or necessarily
unique counts. One token replaced by two unmatched tokens costs two operations.
The bounded Wagner-Fischer fallback may resolve tied scripts differently; total
minimum distance defines the rate.

The detector revision reuses the cached contamination CSV, verifying its
source/split/attack key set. It does not recalculate a different rate for the
official ratio's all-position denominator. The rate is a common Falcon-token
modification covariate, not the exact fraction of every detector's evidence
terms altered by the attack.

Full pre-truncation target IDs also omit the first context token, so
`full_text_edit_audited` is not literal character-level alignment of whole
strings. `truncation_hidden_edit_count` is
\(\max(0,d_{\mathrm{full}}-d_{\mathrm{window}})\): a difference between separately
optimal distances, not localization of each hidden edit. Context changes can
also alter model probabilities at unchanged token IDs.

Native attack rate is separate. `normalize_raid_record` reads a release rate
column when provided; otherwise it assigns a configured attack default and
marks `native_attack_rate_source=raid_paper_attack_default`. An imputed
default is not a measured count of operations actually applied. Neither it nor
\(\rho\) is identified as the theoretical contamination probability. Unicode
changes can be visually small while creating large token edit rates.

## 9. Calibration and estimands

Every detector/domain/fitted specification gets a separate threshold using only
calibration humans. For \(n\) sorted oriented scores, the implementation uses
zero-based index \(n-\lfloor0.05n\rfloor-1\), then `nextafter` toward positive
infinity. Scores at least this threshold predict machine origin. At most
\(\lfloor0.05n\rfloor\) calibration false positives are permitted; ties may make
this conservative. Held-out FPR need not equal 5%.

Attack-level TPR averages machine indicators against each document's domain
threshold. Overall FPR pools test humans: neither is an equal-domain macro
average. AUROC uses pooled oriented scores, not domain-normalized percentiles.
Binned metrics use **all test humans** as negatives and only bin machine rows
as positives; humans are not restricted to sources represented in that bin.

`attack_summary.csv` reports raw/full/eligible on twelve machine conditions.
`contamination_summary.csv` adds the oracle in four eligible bins, with
row/source counts, rate quartiles, and attack/generator/decoding compositions.
Rate aggregates pool machine rows; they need not equal the equal-attack mean
used for tuning.

Trade-off rows apply each fixed oracle cap and threshold to all clean tests,
its attacked-bin aggregate, and represented individual attacks in the bin.
Every fixed configuration is calibrated; this does not establish the FPR of an
arbitrary rate-routing classifier unless human routing is specified.

## 10. Bootstrap uncertainty

The final 2,000 replicates use base seed **260825**.
`RAID/raid_evaluation.py::_bootstrap` resamples test source IDs within domain
using multinomial counts, preserving domain sample sizes. A source's human and
included machine rows share its weight. Raw and clipped values within a contrast
are paired. The many attack variants are not treated as independent subjects.

Directions, candidate selection, caps, bins, and classification thresholds stay
**fixed**. Percentile 95% intervals measure test-source uncertainty conditional
on those fitted choices, not tuning/calibration uncertainty, family-selection
randomness, model-training uncertainty, or attack-generation uncertainty.

Each detector/condition/scope gets a derived seed. Full and eligible comparisons
do not share one joint replicate table. Stored difference intervals are each
method **minus raw**, not eligible-minus-full. Never subtract interval endpoints
or use marginal CI overlap as the direct contrast test. A direct full/eligible
contrast requires jointly paired resampling, unless full is exactly raw for
every row, in which case the eligible-minus-raw contrast already applies.

There is no multiplicity adjustment. Many attack/bin/detector intervals are
exploratory heterogeneity evidence, not simultaneous family-wise guarantees.
Increasing 500 to 2,000 bootstrap draws refines Monte Carlo precision and does
not create an independent confirmation dataset.

## 11. Paper comparator

The seven published-comparator values transcribed into configuration match
[Dugan et al., RAID, Table 6](https://arxiv.org/html/2405.07940v2), whose
caption specifies FPR 5%. This documentation audit checked that table directly.
The peer-reviewed publication is available from
[ACL Anthology](https://aclanthology.org/2024.acl-long.674/).

The values are:

| Condition | Comparator TPR |
|---|---:|
| None | 0.796 |
| Paraphrase | 0.803 |
| Synonym | 0.435 |
| Perplexity misspelling | 0.780 |
| Homoglyph | 0.377 |
| Whitespace | 0.701 |
| Article deletion | 0.743 |

Use **the corrected revision's** `binoculars_sanity.csv`, which looks up
`binocular_origin`, not a same-named historical gap file. It records the
comparator, reproduced point/interval, difference, and achieved held-out FPR.
These constants do not measure the original paper's actual empirical FPR.

This is a **protocol-aligned baseline sanity comparison**, not exact replication
of every RAID choice. Labeled subset, one sampled family per source, pilot
exclusion, independent calibration, domain mixture, and software stack differ
or need explicit matching evidence. Same-logit component parity alone does not
guarantee the same table row. Cite the original RAID paper/table and the
Binoculars implementation revision recorded by the gate, not merely constants
transcribed into this repository. The current upstream source is
[Binoculars metrics.py](https://github.com/ahans30/Binoculars/blob/main/binoculars/metrics.py);
its moving main branch is not a substitute for the saved gate reference hash.

## 12. Artifact lifecycle and validated counts

Prepared rows and original features remain under `runs/raid/<source-run-id>/`.
Separate ratio shards/gate markers are under `runs/raid_origin/<revision-id>/`.
Final corrected artifacts are under `results/raid/<revision-id>/`, including
revision manifest/marker, validation report, frozen specs, evaluation counts,
metrics, attack/rate/calibration/trade-off summaries, sanity comparison,
contamination records, and plots.

The base 500-to-2,000-bootstrap promotion remains a historical route. The final
v4.1 result is a **subsequent detector revision**, not merely that promotion.
New formula, configuration, selection, or scorer identity requires a new result
identity; never rewrite old completion markers to match current source code.

Four score shards assign whole source families using
`stable_int("raid-score-shard-v1", source_id) % num_shards`.
Shards are computational partitions, not four independently fitted experiments.
Global row-key/provenance validation precedes tuning and calibration.

| Artifact | Exact eight-detector source | Seven-detector paper view |
|---|---:|---:|
| `metrics.csv` | 416 | 364 |
| `attack_summary.csv` | 96 | 84 |
| `contamination_summary.csv` | 32 | 28 |
| `calibration_summary.csv` | 384 | 336 |
| `rate_bound_tradeoff_summary.csv` | 336 | 294 |
| `contamination_records.csv` | 154,452 | 154,452 |
| `binoculars_sanity.csv` | 7 | 7 |
| Universal specifications | 16 | 14 reported |
| Rate-specific specifications | 32 | 28 reported |
| Plots | 3 | 3 regenerated |

The seven-detector view filters out gap without refitting. The exact source
validation intentionally describes eight detectors; the paper-view inventory is
in `raid_export_summary.json`. Large JSONL packs are not downloaded, so
preserve Delta packs for future score-level reanalysis.

## 13. Source map and interpretation boundaries

| Item | Implementation / evidence |
|---|---|
| Family selection and split | `RAID/raid_data.py::select_complete_families`, `_split_assignments` |
| Window and alignment | `RAID/raid_scoring.py::encode_output_only_window`; `RAID/raid_data.py::measure_realized_contamination` |
| Scores and anchored ratio | `RAID/raid_evaluation.py::raw_document_score`, `oriented_document_score`; `experiment_core/detectors/detector_revision.py::origin_score` |
| Official ratio components | `RAID/binocular_origin.py::official_components`; revision scorer config |
| Caps and fitting | `RAID/raid_evaluation.py::candidate_specifications`, `select_universal_specification`, `select_rate_adaptive_specification`; frozen specs |
| Threshold and metrics | `experiment_core/analysis/evaluation.py::calibration_threshold`, `auroc`; `RAID/raid_evaluation.py::_point` |
| Intervals | `RAID/raid_evaluation.py::_bootstrap`, `_seed` |
| Publication filtering | `tools/exports/export_final_publication_bundle.py`; export summary |

Historical `archive/raid_pilots/` tools investigated alternate selectors,
trimming, and legacy-gap component clipping. They are development analyses,
not additional final estimands; their old naming does not establish an
original-ratio experiment. Their inspected sample is excluded from the final fit.

The local audit verifies code, config, saved choices, counts, and downloaded
artifact consistency. It cannot reconstruct the original release, all GPU
logits, or every source exclusion from a bundle lacking those inputs. A passed
validator or favorable curve does not alone establish human-specific editing,
the theoretical sparse-contamination model, exact upstream replication, or
uniform improvement at exactly identical held-out FPR.
