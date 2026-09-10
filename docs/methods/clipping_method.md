# Detector scores and clipping: final reported methods

Audited 2026-09-08 against active code and the downloaded final nine-cell plus
RAID bundle. This replaces the historical gap-only method note. See
[Detector revision](../DETECTOR_REVISION.md) for release-specific amendments and
[Final results guide](../FINAL_RESULTS_GUIDE.md) for the evidence used in reporting.

## 1. Scope and notation

Clipping changes aggregation of token evidence, not generation, contamination,
tokenization, or model weights. These are document detectors built from token
features, not token-labeling or sentence-labeling systems. A clipping bound and
a human-calibrated classification threshold are different parameters.

Let \(p_i(v)\) be the conditional scorer distribution, \(x_i\) the observed
token, and \(n\) the number of scored positions. Natural logarithms are used:

$$
a_i=-\log p_i(x_i),\qquad
r_i=1+\#\{v:p_i(v)>p_i(x_i)\},\qquad
\ell_i=\log r_i,\qquad
h_i=-\sum_v p_i(v)\log p_i(v).
$$

A bar denotes an arithmetic mean over scored positions. Rank is competition
rank: tied probabilities receive the same rank, rather than arbitrary positions
in a sorted vocabulary. Production single-model token features use float32
logits after the configured model forward pass; this does not mean all inference
is float32.

## 2. Raw scores and direction

| Reported key | Raw document score | Revised direction |
|---|---|---|
| log_likelihood | \(-\bar a\) | Learned on clean tuning documents |
| rank | \(\bar r\) | Learned on clean tuning documents |
| log_rank | \(\bar\ell\) | Learned on clean tuning documents |
| lrr | \(\bar a/(\bar\ell+10^{-12})\) | Fixed: larger is machine-like |
| entropy | \(\bar h\) | Learned on clean tuning documents |
| entropy_gap | \(\overline{a-h}\) | Learned on clean tuning documents |
| binocular_origin | Performer mean NLL / observer-to-performer mean cross-entropy | Fixed: smaller is machine-like |

For learned directions, \(d=+1\) if the clean machine tuning mean is at least
the clean human tuning mean, otherwise \(d=-1\). The oriented score is \(dS\);
larger always supports machine origin. Direction is frozen before calibration,
not selected separately by attack or test contamination rate.

LRR means the [DetectLLM](https://arxiv.org/abs/2306.05540)
**log-likelihood/log-rank ratio**, not a log-likelihood
ratio between two probability models. It is a ratio of document means, not a
mean of tokenwise ratios. Binoculars is also not an ordinary likelihood-ratio
test. Robustness results for a likelihood ratio cannot automatically be
asserted for these heuristic scores.

The [authors' reference repository](https://github.com/mbzuai-nlp/DetectLLM)
describes the larger-machine LRR direction. Our competition-rank tie handling,
prompt conditioning in primary cells, and empirical calibration are explicit
implementation conventions, not a claim of identical upstream execution.

The signed entropy-gap statistic is connected to Section 5.2 of
[Radvand et al., A Training-free Method for LLM Text Attribution, v5](https://arxiv.org/html/2501.02406v5#S5.SS2).
That section gives an absolute-gap test and also discusses a one-sided signed
test. Our implementation uses the signed gap, an empirical tuning direction,
and human calibration; it is not an exact reproduction of the absolute-gap
test or its theoretical threshold. It has no Fast-DetectGPT variance
normalization. Cite the connection without claiming that the paper's
assumptions and guarantees have been verified for this experiment.

## 3. Additive one-sided clipping

For log likelihood, rank, log rank, entropy, and entropy gap, let \(f_i\) be the
raw token feature and \(z_i=df_i\). The clipped score is already oriented:

$$
S_{\mathrm{clip}}=\frac1n\sum_i\max(z_i,L).
$$

This caps evidence adverse to the machine label. It is not clipping the final
document score. With \(d=-1\), it upper-caps the original feature.
The saved field named lower is a bound on oriented token evidence.

## 4. Ratio-specific clipping

### LRR

The revised direction is \(+1\), with two capped components:

$$
S_{\mathrm{LRR,clip}}=
\frac{\operatorname{mean}_i\min(a_i,U_a)}
{\operatorname{mean}_i\min(\ell_i,U_\ell)+10^{-12}}.
$$

The fields are nll_upper and log_rank_upper. Thus one LRR specification contains
two component bounds, not literally one scalar. A nonempty specification must
have finite \(U_\ell>0\). A zero log-rank cap collapses the denominator;
the small stabilizer does not make that intervention valid LRR.

### Binoculars-origin: two protocols

Performer \(q\) is Falcon-7B-Instruct; observer \(p\) is Falcon-7B.
Observer-to-performer cross-entropy is \(b_i=-\sum_v p_i(v)\log q_i(v)\).
The direction is not reversed. The raw score is \(A/B\), smaller = machine-like.
The baseline attribution is
[Spotting LLMs With Binoculars](https://arxiv.org/abs/2401.12070).

**Primary nine cells:** both terms use the existing prompt-conditioned
continuation window and saved arrays. Raw is \(\bar a/\bar b\), clipped is
\(\operatorname{mean}\min(a_i,U)/\bar b\), aggregated in float64. The prompt
remains context, not scored continuation. Call this a **prompt-conditioned
Binoculars-ratio adaptation**, not an output-only upstream reproduction.

**RAID:** output-only input, tokenizer-native special tokens and a 512-input-token
truncation limit. Performer NLL uses shifted predictions; denominator CE uses
all input positions, including the final position, with the upstream pad-token
mask. The terms therefore have different position sets. The gate compares
components on identical logits with the
[upstream Binoculars metrics implementation](https://github.com/ahans30/Binoculars/blob/main/binoculars/metrics.py).
Saved official numerator \(A_{\rm off}\), denominator \(B_{\rm off}\), and
raw ratio preserve the actual upstream numerical outputs.

The custom RAID numerator-only intervention is anchored:

$$
\Delta_U=\operatorname{mean}_{64}\min(a_i,U)-\operatorname{mean}_{64}a_i,\qquad
A_U=\min\{A_{\rm off},\max(0,A_{\rm off}+\Delta_U)\}.
$$

Active clipping divides float32 \(A_U\) by float32 \(B_{\rm off}\). No clipping,
or a bound changing no token, returns the saved official raw scalar exactly.
This avoids claiming that serialized BF16 losses reconstruct CUDA parallel
reductions bit-for-bit. In exact arithmetic it reduces to the ordinary clipped
numerator mean. The denominator is unchanged. This is **our clipped extension**,
not an upstream method.

### Historical gap

The old key binoculars means \(\exp(\overline{a-b})\), renamed binocular_gap
in revised compatibility outputs. Its local-gap clipping remains readable for
historical comparison. Neither expression is the official ratio. Final
seven-detector paper views omit gap; immutable eight-detector source artifacts
retain it. Never relabel old gap scores as origin.

## 5. Candidates and safeguards

The common grid is
\(\mathcal Q=\{0.80,0.85,0.90,0.95,0.975,0.99,0.995\}\).
Quantiles use concatenated tokens from **clean human and clean machine tuning
documents**. Longer documents contribute more token observations; these are
not quantiles of document means.

- Additive: \(L=Q_{1-q}(z)\), up to seven active candidates.
- Origin: \(U=Q_q(a)\), up to seven active candidates.
- LRR: Cartesian pairs \((Q_q(a),Q_{q'}(\ell))\), up to 49 active candidates.
- Empty specification means no clipping and is evaluated first.

Duplicate quantiles need not yield distinct transformations. Revised selection
rejects a nonempty candidate if every pooled clean human/machine tuning document
score is exactly identical, or structural full saturation would produce a
constant apart from rounding. LRR structural saturation requires both
components; a constant origin numerator alone does not imply a constant ratio.
V4.2 additionally rejects nonpositive LRR denominator caps. These checks use
tuning data only. Zero test TPR is not a rejection criterion.

No clipping stays available. Primary replacement requires objective improvement
greater than \(10^{-15}\); RAID uses \(10^{-12}\). Thus no clipping wins a
numerical tie against the initial candidate. Inspect frozen specifications,
not just the method label, to establish whether clipping was selected.

## 6. Tuning objectives actually used

All AUROCs compare the applicable machine documents to the same clean human
tuning documents under the candidate's oriented scores.

| Study / method | Objective and population |
|---|---|
| Primary primary_frozen_mixture | \(0.8\,\mathrm{AUROC}_{\rm pooled\ mixture}+0.2\,\mathrm{AUROC}_{\rm clean}\) |
| RAID full universal | \(0.8\,\operatorname{mean}_a\mathrm{AUROC}_a+0.2\,\mathrm{AUROC}_{\rm clean}\); all eleven attacks |
| RAID eligible universal | Same objective; attacked tuning rows restricted to \(0<\rho\leq0.5\), averaging represented attacks |
| RAID rate-specific oracle | Mean attack-wise AUROC gain within a bin, with clean AUROC loss at most 0.01 |

Primary tuning uses requested ratios 10/20/30/40/50%, not 5%, and both modes.
Three random draws and one tail construction imply random:tail **3:1 row
weight** in the pooled objective. It is not an equal-weight attack/mode average.
One frozen specification applies to all reported ratios and both modes.

For RAID universal methods the 20% clean term is a **soft reward**, not a hard
zero-clean-loss budget. Full tuning retains zero-realized-change and dense
attacked rows; none is the separate clean term. Eligible filters the attacked
term only, not clean references or quantile construction. No attack-specific
caps are fitted.

RAID oracle bins are \((0,0.05],(0.05,0.10],(0.10,0.20],(0.20,0.50]\).
Fewer than 250 tuning machine rows or no represented attack triggers the
full-universal fallback. Routing requires a clean/attacked pair to determine
the bin. These are secondary oracle diagnostics, not deployable routing or a
guaranteed upper bound on test performance. See the
[RAID design](../raid/SCIENTIFIC_DESIGN.md) for rate construction.

## 7. Calibration and uncertainty

Every raw/clipped configuration gets its own threshold from held-out clean
human calibration scores. Primary targets are 1% and 5%; RAID uses 5% and
separate thresholds by domain. No threshold is learned from attacked test rows.

For \(m\) sorted human calibration scores and target \(\alpha\), allow
\(k=\lfloor\alpha m\rfloor\) exceedances. The threshold is the next floating-point
value above zero-based index \(\max(0,m-k-1)\). Prediction uses score at least
the threshold. Ties can make calibration conservative. This does not guarantee
held-out FPR equals the target: report actual FPR and its interval with TPR.

AUROC gives half credit for ties. Primary normalized partial AUROC integrates
the ROC curve over FPR 0–0.05 and divides by 0.05; it is not chance-corrected
standardized partial AUC. Robustness AUC instead integrates TPR against requested
contamination rate, normalized by the rate span.

Final intervals use 2,000 source-cluster bootstrap draws and 2.5%/97.5%
percentile endpoints. Primary resamples sources within a cell; RAID resamples
within domain. Human documents, machine variants, and raw/clipped measurements
stay paired. Directions, selected bounds, and calibration thresholds stay fixed.
Intervals reflect test-source variability conditional on fitting, not the
uncertainty of the full tuning/calibration pipeline. They are not
multiplicity-adjusted.

Use stored paired clipped-minus-raw intervals. A full-minus-eligible interval
cannot be obtained by subtracting separately computed interval endpoints.

## 8. Interpretation and code anchors

Clipping can improve clean discrimination, contamination robustness, both, or
neither. Report clean and attacked changes together; improvement is not a
validation requirement. Synthetic splicing is contamination, not evidence of
fluent human editing. RAID's normalized tokenizer edit rate is not the exact
conditional mixture probability in a robustness theorem.

Implementation paths relative to the repository root:

- experiment_core/detectors/scoring.py: features and legacy scores.
- experiment_core/detectors/detector_revision.py: ratios, anchoring, safeguards.
- experiment_core/analysis/evaluation.py: primary fitting/calibration/bootstrap.
- RAID/binocular_origin.py: official raw components.
- RAID/raid_evaluation.py: RAID fitting/calibration/bootstrap.

Base configurations still name legacy binoculars. Producing the final ratio
report requires the revision entry points; a fresh base run is not automatically
the final seven-detector protocol.
