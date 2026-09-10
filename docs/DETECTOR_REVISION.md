# Detector amendments and accepted release ledger

Scientific amendment ID: binocular-origin-lrr-constant-v1.
Current implementation: official-score-anchored-clipping-v4.2.
**Current code version and accepted result version are not interchangeable.**

This is a disclosed methodological amendment following inspection of earlier
results. Historical data and results were preserved, not silently overwritten.
The authoritative formulas and fitting rules are in
[Clipping method](methods/clipping_method.md); exact exported inventories are in
[Final results guide](FINAL_RESULTS_GUIDE.md).

## 1. What changed

| Item | Earlier behavior | Revised behavior |
|---|---|---|
| Binoculars naming | binoculars denoted exponential mean gap | Preserve as binocular_gap; add genuine ratio as binocular_origin |
| Primary origin | Not reported as a separate ratio | Derive prompt-conditioned ratio from saved components |
| RAID origin | Legacy gap arrays lacked official denominator positions | Separately score official output-only components |
| LRR direction | Learned from clean tuning means | Fixed +1, larger = machine-like |
| Constant clipping | Saturated candidates could win | Reject exactly constant pooled clean-tuning scores and structural saturation |
| LRR denominator cap | Zero could erase denominator information | V4.2 rejects nonfinite/nonpositive log-rank caps |
| Paper reporting | Compatibility results contain eight detectors | Export seven methods, omitting gap without deleting provenance |

Other learned directions still use clean tuning means; origin is fixed -1.
No sign is flipped merely to improve a held-out curve. A zero TPR is not an
implementation error by itself and is not a candidate rejection rule.

## 2. Two different origin protocols

Primary nine-cell origin keeps the **original prompt conditioning** and saved
continuation windows. Both numerator and denominator are float64 means of
their saved continuation-token components. Clipping caps numerator NLL only.
Call it a conditional Binoculars-ratio adaptation. It needs no new generation
or GPU scoring because the required components are already saved.

RAID origin follows output-only tokenizer-native input, right truncation at
512 input tokens, causal shifted performer NLL and all-position
observer-to-performer cross-entropy. The numerator uses shifted attention;
the denominator masks by token ID unequal to pad ID. The final input position
is included only in the denominator. The old gap arrays cannot reconstruct
this denominator, so a distinct origin score pack was necessary.

The gate compares raw components on identical logits with a separately supplied
[upstream metrics file](https://github.com/ahans30/Binoculars/blob/main/binoculars/metrics.py).
Its SHA256 and resolved model/tokenizer revisions are saved. Component equality
is not proof of identical dataset sampling, model-loading code, calibration,
or reproduction of every published paper result.

## 3. Numerical amendment history

BF16 token arrays do not retain CUDA parallel reduction order. CPU reconstruction
can therefore disagree with the saved official document numerator despite
mathematical equality. V4 uses the actual official numerator as an anchor and
adds the deterministic float64 clipping delta. V4.1 preserves that numerator
through the merge with Falcon and gap packs; omission in V4 prevented evaluation
of active origin-clipping specifications.

For numerator losses \(a_i\) and cap \(U\):

$$
A_U=\min\{A_{\rm off},\max(0,A_{\rm off}
+\operatorname{mean}_{64}\min(a_i,U)-\operatorname{mean}_{64}a_i)\}.
$$

The saved official denominator stays fixed. Active final division is float32;
raw/no-token-change returns the saved raw scalar exactly. This intervention is
our numerator-only clipping extension, not an upstream clipped detector.
Primary conditional aggregation remains float64 and does not use this anchor.

V4.2 separately forbids a nonempty LRR specification with a nonpositive
log-rank cap. Since log rank is nonnegative, a zero cap destroys its denominator,
turning LRR into a scaled numerator-only detector. The denominator stabilizer
does not justify that transformation. This check is additional to constant-score
rejection: a zero-denominator-cap score can still vary across documents.

## 4. What actually completed

| Accepted evidence | Version/identity and scope |
|---|---|
| Eight primary cells | September 4 revisions ending -revision-20260904T163234Z; retained without a blanket v4.2 rerun |
| Qwen32–SQuAD | qwen32-squad-promptfix-20260804T133827Z-lrr-v42-20260907T163410Z; evaluation-only positive-cap correction |
| RAID | raid-origin-anchored-v41-20260907T044156Z; full origin scoring/merge/evaluation and 2,000 bootstraps |
| Publication bundle | primary-nine-plus-raid-final-20260908T150247Z; filters source compatibility outputs to seven detectors |

All nine final primary selected LRR specifications have positive log-rank caps.
Qwen32–SQuAD's selected cap is approximately 0.69314718, with NLL cap 8.14993670.
The six final RAID LRR specifications (two universal and four bin-specific)
also have positive caps. The recorded RAID audit found no unselected zero-cap
candidate. Therefore RAID remained v4.1 and did not require a v4.2 rerun.
Do not alter its implementation label or hashes to match current code.

Eight older primary source reports intentionally still contain eight-method
provenance. Their publication tables omit gap. The Qwen32–SQuAD replacement
already reports seven. Source and publication row counts differ for this
documented reason, not because a detector or sample went missing.

This documentation audit reruns no scientific experiment. It does not ask the
user to submit all nine cells again.

## 5. Reproduction programs and validation contract

Paths below are relative to the repository root. Use them only for a deliberately
new or identically resumable analysis with recorded input identities.

- Primary revision: tools/reanalysis/reevaluate_detector_revision.py.
  Reuses the source target and Falcon packs; writes a new result ID.
- RAID revision: RAID/revise_detectors.py, stages gate, score, evaluate.
  Reuses prepared rows and single-model features; new origin components require
  the pair scorer unless validated components can be adopted.
- Wrapper: tools/delta/delta_detector_revision.sbatch.
  See [Delta guide](raid/DELTA_GUIDE.md) for environment, arguments and stages.
- Publication export: tools/exports/export_final_publication_bundle.py.
  Filters and redraws without changing fitted scientific results.

The integrated RAID gate first validates source identity and scans the complete
prepared input with the pinned tokenizer. It rejects invalid/unscorable windows
rather than dropping or editing texts. One whole source family per domain/split
is selected before inspecting scores: normally 24 sources and 312 rows.
The gate tests real scoring, saved-pack serialization, exact resume, merge,
evaluation and plotting, plus reference-component checks and special probes.
Small gate thresholds never supply the final full-run fits.

The scoring array partitions whole source families into four shards. These are
not four independent RAID experiments. All shard keys and completion markers
must match before final evaluation. Adoption validates and records origin
provenance, then scores missing keys; it does not license mixing arbitrary
model/tokenizer versions.

Final source RAID validation reports 167,323 score rows, eight compatibility
detectors, 416 metric rows and 2,000 bootstraps. The paper view has seven
detectors and 364 metric rows. Full marker, counts, field/provenance checks
and result hashes matter; a Slurm COMPLETED state alone is insufficient.

## 6. Maintenance and evidence limits

Changed formulas, fitting rules or configurations require a new analysis ID.
A styling-only export does not require GPU scoring. Reorganized code paths
change implementation fingerprints, so an old gate cannot certify a newly
modified executable. This does not invalidate an immutable completed result
under its historical code identity.

Local tests can verify arithmetic, guards, merge/resume and export contracts;
they cannot establish live GPU/model parity without the integrated gate.
The recorded Delta gate and completed result evidence are distinct from this
local documentation audit. See the final-results and audit guides for what
was checked from downloaded artifacts and what still requires original Delta
files.

Future amendments should document their motivation and inspected results,
preserve previous outputs, and never make favorable detector performance a
completion requirement.
