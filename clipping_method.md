# Token-level clipping method

Versioned amendment: [DETECTOR_REVISION.md](DETECTOR_REVISION.md) adds
`binocular-origin` with numerator-only clipping, retains the former method as
`binocular-gap`, fixes LRR's direction, and rejects exactly constant clean-tuning
candidates. The historical definitions below describe the archived gap study;
its exponential-gap formula must not be called the original Binoculars ratio.

## Purpose

This document defines the token-level clipping intervention used in the primary
nine-cell contamination study and the Beemo realistic-edit benchmark. The raw
detector is always retained as the scientific control. Clipping changes only
how saved token-level evidence is aggregated into a document score; it does not
change the generated text, contamination process, detector model, calibration
documents, or final test documents.

The implementation is in `llm_detection/evaluation.py`. Beemo reuses the same
detector transformations but has its own balanced clipping-selection objective
in `Beemo/beemo_evaluation.py`.

## Orientation

Some detectors assign larger values to machine text, while others assign
smaller values. Each detector is therefore oriented using only clean human and
clean machine documents from the clipping-tuning split.

Let the raw document score be \(S\). Define

$$
d =
\begin{cases}
+1, & \text{if the mean clean-machine score is at least the mean human score},\\
-1, & \text{otherwise}.
\end{cases}
$$

The oriented raw score is \(dS\), so larger oriented values always mean “more
machine-like.” Orientation is frozen before calibration or test data are used.

## Generic clipping rule

For an additive detector, let \(x_i\) be the local contribution of token \(i\)
and let \(n\) be the number of scored tokens. Its oriented raw score is

$$
S_{\mathrm{raw}}
=
\frac{1}{n}\sum_{i=1}^{n} d x_i.
$$

For a candidate lower bound \(L\), the clipped score is

$$
S_{\mathrm{clip}}
=
\frac{1}{n}\sum_{i=1}^{n}\max(d x_i,L).
$$

Clipping therefore limits how strongly an unusually adverse, human-like token
can reduce the document's machine-evidence score. It does not increase
favorable contributions and is not a symmetric winsorization rule.

## Detector-specific definitions

| Detector | Local token evidence | Implemented clipping |
|---|---|---|
| Log likelihood | \(x_i=\log p(y_i\mid y_{<i})\) | Apply the generic oriented lower floor. When \(d=+1\), extremely low token log probabilities are floored. |
| Rank | \(x_i=\operatorname{rank}(y_i)\), using one-based competition rank | Apply the generic oriented lower floor. When \(d=-1\), this is equivalent to capping extremely high token ranks. |
| Log rank | \(x_i=\log(\operatorname{rank}(y_i))\) | Apply the generic oriented lower floor. When \(d=-1\), unusually high log ranks are capped. |
| Entropy | \(x_i=H(p_i)\) | Apply the generic oriented lower floor after learning whether high or low entropy is machine-like in that cell. |
| Entropy gap | \(x_i=-\log p(y_i\mid y_{<i})-H(p_i)\) | Apply the generic oriented lower floor to the NLL-minus-entropy contribution. |
| Binoculars | Performer NLL minus observer-to-performer cross-entropy | Clip the oriented local gap before averaging and applying the exponential outer transform. |
| DetectLLM LRR | Mean NLL divided by mean log rank | Use separate upper caps for NLL and log rank before recomputing the ratio. |

The direction is data-dependent and is not hard-coded by detector name. The
descriptions above using “when \(d=+1\)” or “when \(d=-1\)” describe the usual
interpretation rather than an imposed direction.

## Binoculars

For token \(i\), define the local Binoculars gap

$$
g_i
=
\operatorname{NLL}_{\mathrm{performer},i}
-
H(p_{\mathrm{observer},i},p_{\mathrm{performer},i}).
$$

The ordinary un-oriented Binoculars score is

$$
B_{\mathrm{raw}}
=
\exp\left(\frac{1}{n}\sum_{i=1}^{n}g_i\right).
$$

The oriented clipped score is implemented as

$$
B_{\mathrm{clip}}
=
d\exp\left(
d\frac{1}{n}\sum_{i=1}^{n}\max(dg_i,L)
\right).
$$

This retains the documented exponential Binoculars form while limiting
adverse local gaps. Because the exponential is monotone, calibration remains
well defined in the oriented score space.

## DetectLLM LRR

LRR is a ratio rather than the mean of one local statistic. Define

$$
n_i=-\log p(y_i\mid y_{<i}),
\qquad
r_i=\log(\operatorname{rank}(y_i)).
$$

The oriented raw score is

$$
S_{\mathrm{LRR,raw}}
=
d\frac{\overline{n}}{\overline{r}+\epsilon}.
$$

For candidate upper bounds \(U_n\) and \(U_r\), the clipped score is

$$
S_{\mathrm{LRR,clip}}
=
d\frac{
\overline{\min(n_i,U_n)}
}{
\overline{\min(r_i,U_r)}+\epsilon
}.
$$

This separate capping rule preserves LRR's ratio structure. Applying the
generic additive floor directly to LRR would define a different detector.

## Candidate clipping specifications

The configured quantile grid is

$$
\mathcal{Q}
=
\{0.80,0.85,0.90,0.95,0.975,0.99,0.995\}.
$$

For a generic detector, candidate lower bounds are derived from token
contributions in clean human and clean machine documents from the
clipping-tuning split:

$$
L_q
=
Q_{1-q}\left(\{d x_i\}\right),
\qquad q\in\mathcal{Q}.
$$

These candidates floor approximately the most adverse 20%, 15%, 10%, 5%,
2.5%, 1%, or 0.5% of clean oriented token contributions.

For LRR, candidates are every pair

$$
U_n=Q_q(\{n_i\}),
\qquad
U_r=Q_{q'}(\{r_i\}),
\qquad q,q'\in\mathcal{Q},
$$

giving 49 capped candidates. The empty specification, meaning no clipping, is
also a candidate for every detector. If it has the best tuning objective, the
reported clipped and raw scores are identical.

## Selection in the primary contamination study

Each dataset-model-detector cell receives its own clipping specification. It is
selected using only the 500-source clipping-tuning split. Candidate bounds are
constructed from clean human and clean target-model tokens in that split.

The predefined robustness mixture contains both random and white-box tail
contamination at 10%, 20%, 30%, 40%, and 50%. The 5% condition is evaluated but
is not used to select the clipping specification.

For candidate specification \(c\), the selection objective is

$$
J(c)
=
0.8\,\operatorname{AUROC}
(\text{human},\text{contaminated machine};c)
+
0.2\,\operatorname{AUROC}
(\text{human},\text{clean machine};c).
$$

The candidate with the largest objective is frozen. Ties within numerical
tolerance retain the first candidate, and the candidate order begins with no
clipping. The frozen specification is then reused for both attack modes, every
contamination ratio, the calibration split, and the final test split. The
mode-specific oracle analysis is disabled in the paper configuration.

## Selection in Beemo

Beemo uses the same detector-specific transformations and the same quantile
grid. Orientation compares human responses with original machine responses in
the 437-record clipping-tuning split. Candidate bounds are derived from tokens
in those two clean families.

The Beemo objective gives equal weight to four machine-origin families:

$$
J_{\mathrm{Beemo}}(c)
=
\frac{1}{4}
\left(
\operatorname{AUROC}_{\mathrm{original}}(c)
+\operatorname{AUROC}_{\mathrm{expert}}(c)
+\operatorname{AUROC}_{\mathrm{Llama}}(c)
+\operatorname{AUROC}_{\mathrm{GPT}}(c)
\right),
$$

where every AUROC contrasts that machine-origin family with independent human
responses. This balanced objective prevents the clipping bound from being
selected primarily for the largest edit family.

## Selection in RAID

RAID retains one universal specification per detector as the primary method.
Its tuning objective assigns 80% weight to the mean AUROC across the eleven
attacks (equally weighted) and 20% to unattacked-machine AUROC.

A secondary rate-oracle uses four fixed Falcon-token edit-rate intervals:
\(0<\rho\leq0.05\), \(0.05<\rho\leq0.10\),
\(0.10<\rho\leq0.20\), and \(0.20<\rho\leq0.50\). Within each
interval and detector, it selects the feasible candidate with the largest mean
attacked AUROC gain over raw, equally weighting the attacks represented in that
interval. Feasibility requires clean-machine AUROC loss relative to raw to be
at most 0.01. Thus, clean performance is a constraint rather than a positively
weighted objective term.
Rows with \(\rho=0\) or \(\rho>0.50\) are excluded from rate-bin fitting and
attacked-bin evaluation; they remain in universal and attack-level analyses.
Unattacked rows are additionally reused for the clean-loss constraint and
counterfactual clean-cost report. A fixed interval with
fewer than 250 tuning machine rows falls back to the universal specification.
The boundaries are never moved to equalize sample sizes.

The rate-oracle is not a deployable detector because assigning a test attack to
an interval requires its unattacked counterpart. It is reported as a mechanism
diagnostic and possible upper bound; universal clipping remains primary.
For each rate-specific bound, RAID separately reports its counterfactual effect
on unattacked `none` rows and its effect on the aggregate and attack-specific
test rows in the corresponding rate interval.

Before freezing the full RAID analysis, the bounded 500-source pilot also has a
separate exploratory selector comparison. It crosses full-universal and
eligible-range-universal fitting with four selectors, and four-bin rate-oracle
fitting with three nonredundant selectors. It examines four clean-AUROC-loss
budgets and a denser candidate quantile grid. These 11 methods are implemented
in `RAID/compare_tuning_methods.py`, write separate point-estimate artifacts,
and do not change the frozen primary evaluator. Their sole purpose is to choose
one universal protocol before the full benchmark is run. Since that choice
uses pilot test outcomes, the comparison records all pilot source IDs and the
future full benchmark must exclude them before creating its final split.

## Calibration and final evaluation

Raw and clipped aggregation receive separate calibration thresholds. In the
primary study, thresholds are selected from 500 clean human calibration
documents; in Beemo, they are selected from 875 human calibration records; and
in RAID, each universal or fixed-rate specification receives a separate
per-domain threshold from clean calibration humans. The primary and Beemo
target false-positive rates are 1% and 5%; RAID reports only 5%.

No clipping direction, bound, or calibration threshold is selected using final
test outcomes. The final comparisons are paired because raw and clipped scores
are calculated for the same documents, and confidence intervals resample the
same source or Beemo record clusters.

## Saved representation

Primary-study `metrics.csv` files store the frozen specification in the
`clipping_specification` column:

- `{"lower": ...}` for additive detectors and Binoculars;
- `{"nll_upper": ..., "log_rank_upper": ...}` for LRR;
- `{}` when no clipping is selected.

Beemo and RAID store complete fitted specifications and tuning diagnostics in
`frozen_specs.json`. RAID records the universal specification, all four
rate-oracle specifications, their calibration thresholds, tuning counts, and
any sparse-bin fallback. The compact export bundle also preserves these files,
along with all point estimates and confidence intervals needed for tables and
plots.

## Interpretation

Clipping should be interpreted as a robust aggregation intervention, not as a
new token-level detector. A positive clipped-minus-raw effect means that
limiting extreme adverse token contributions improved the chosen evaluation
metric under the frozen protocol. It does not by itself establish that those
extreme contributions were caused by genuine human editing; synthetic splice
boundaries, disrupted coherence, and tokenizer effects can also produce
extreme local evidence. The primary contamination study and Beemo benchmark
must therefore be reported as complementary experiments.
