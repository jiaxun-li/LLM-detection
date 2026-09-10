# Documentation-to-code-and-results audit

Date: 2026-09-08. Scope: the active experiment documentation, primary nine-cell
release, corrected RAID release, shared detector/evaluator implementation,
configuration and Delta/export instructions. Three parallel reviewers covered
primary science, RAID science, and operations; the lead reviewed detector
mathematics, literature attribution, result tables/integrity, and their combined
documentation changes.

This is a documentation audit, not a new scientific experiment or a claim that
all original inference was independently reproduced. No experiment code,
configuration, downloaded result, source dataset, or user-owned paper material
was changed. No Delta job, Git push or Git pull was performed.

## 1. Evidence and authority

The accepted bundle is
`downloads/current/primary-nine-plus-raid-final-20260908T150247Z/`.

For report writing, use final metrics together with their export transformation
and original source/revision records. Match the corresponding recorded code
identity when reconstructing a run. Current code and JSON defaults explain
current behavior but do not override historical resolved settings or turn an
older result into a new version. This is especially important after folder
reorganization and detector amendments.

Evidence inspected:

- Active primary preparation, scoring, evaluation, detector-revision and export
  code; RAID data selection, scoring, evaluation and revision paths.
- Baseline/smoke configurations and operational wrappers.
- Final bundle manifest, completion marker, all recorded artifact sizes/hashes.
- Per-cell revision/configuration records, final CSVs, rejection records and
  RAID frozen specs, counts, validation and sanity table.
- Original-paper sources for RAID's selected comparator and entropy-gap
  attribution; upstream Binoculars component definitions.
- User-reported completed Delta verification, clearly distinguished from tests
  executed locally during this audit.

## 2. Material discrepancies corrected

| Topic | Earlier misleading description | Corrected documented contract |
|---|---|---|
| Completed scope | 21-cell planned matrix read as completed | Nine released primary cells plus RAID; extra models are historical plans |
| Human editing | Token splicing described as human-edit robustness | Controlled contamination; grammatical fluency/human-specific effect not established |
| Binoculars | Exponential gap conflated with published ratio | Gap preserved historically; origin is separate ratio, with two scoring protocols |
| Primary context | Risk of implying output-only rerun | Original prompt conditioning retained for primary origin |
| RAID precision | Saved arrays assumed to replay CUDA numerator exactly | Official scalar anchoring, precise clipping delta, position/mask distinction |
| LRR direction | Every detector learns direction | Revised LRR fixed +1, origin fixed -1; other directions learned on clean tuning |
| LRR cap | Zero denominator cap treated as harmless clipping | V4.2 rejects it; only affected Qwen32–SQuAD final cell replaced |
| Degenerate candidates | Constant/fully saturated candidates not clearly distinguished | Exact tuning-only constant/structural checks, no near-constant test-driven tolerance |
| Primary tuning | Attack-average or generic clean-loss-budget description | 0.8 pooled attacked AUROC + 0.2 clean AUROC; random:tail row weight 3:1 |
| RAID universal tuning | Zero-clean-loss or crossfit pilot selector implied final | Equal-attack AUROC with 0.8/0.2 soft clean reward, full versus eligible support |
| Oracle fitting | Deployable or guaranteed upper-bound interpretation | Four fixed paired-rate bins; 0.01 clean-AUROC loss constraint and 250-row full fallback |
| Contamination rate | Positional mismatch/theoretical epsilon/native operations conflated | Minimum token edit distance in scored window; native defaults and full-window caveats explicit |
| Calibration | Target 5% interpreted as exact test FPR | Human-only calibration, per-domain for RAID; actual test FPR separately reported |
| Bootstrap | Generic whole-procedure uncertainty implied | Source-cluster test uncertainty with frozen direction/bounds/thresholds |
| AUC definitions | Partial AUC and robustness AUC ambiguous | ROC area/0.05 versus normalized TPR–contamination curve area |
| Saved settings | Current defaults assumed for all cells | Per-cell saved overrides, including Granite guard8/hidden states, documented |
| Result counts | Eight-method source reports seen as failed seven-method exports | Immutable provenance plus explicit seven-method projection and enclosing hashes |
| Plot counts | RAID total assumed to be three | Three source plus three paper-view plots; 99 primary plots |
| Versions | Current v4.2 assumed to require all runs again | Eight earlier primary revisions, one Qwen SQuAD v4.2, accepted RAID v4.1 |
| Operations | Old paths/account/CLI/transfer recipes | Current routing, physical venv versus friendly name, GPU-only account caveat, safer shell scope |
| Literature | Entropy gap dismissed as wholly unattributed | Signed one-sided connection to Radvand et al.; not exact absolute-gap/variance-normalized detector |

The audit did not choose a new selector, change any bound, improve a result by
relabeling it, or discard an unfavorable condition.

## 3. Checks executed

| Check | Result |
|---|---|
| All final bundle recorded artifact sizes and SHA256 | 182/182 match |
| Primary combined and derived CSV counts | 3,528 metrics; 252 clean; 1,638 effects; 504 robustness; nine inventory rows |
| Primary plots | 99 PNGs |
| RAID paper/source metric counts | 364 / 416 |
| RAID attack, rate, calibration, tradeoff and sanity table counts | Match the inventories in the final-results guide |
| RAID plots | Three paper-view plus three original source PNGs |
| Large JSONL in publication bundle | Zero, intentionally |
| Final raw origin sanity values | Re-read from CSV; comparator constants checked against RAID Table 6 |
| Final selected LRR caps | Positive in the nine primary selected specifications and six RAID specifications |
| Local active regression suite | 105 discovered: 96 passed, nine Torch/CUDA-dependent skips; zero failures |
| Operations-focused regression subset | Nine tests passed |
| Active documentation link/math check | 19 documents; no missing local link targets or unbalanced math delimiters |
| Change-scope and whitespace check | Documentation only; Git diff whitespace check passed |

The local test command was `python -B -m unittest discover -s tests -v` using
the bundled Windows Python runtime. It did not download models. GPU skips are
not presented as passes. The separate user-reported Delta repository smoke
21883773 passed all 105 tests without skips in 37 seconds; that historical
engineering check is not a new full-model experiment performed by this audit.

## 4. Limits and reporting consequences

### Original score-level evidence remains on Delta

The download omits large prepared/base/scoring JSONL packs. This audit cannot
independently reconstruct all source selections, exclusions, cross-model source
overlap, empirical token-length distributions, alignment operations, or GPU
logits from summaries alone. Preserve those packs, the 500-source exclusion
list, resolved dataset/model/tokenizer records, reference metrics file/hash,
and original logs for strict replication. Null configuration revision fields
alone are not a reproducible model pin.

### Baseline agreement is scoped

Corrected RAID origin closely matches the selected published comparator values;
its held-out raw FPR is 4.6223% at target 5%. This supports a protocol-aligned
sanity check, not exact replication of every sampling/calibration/software
choice in the RAID paper. Primary origin must be labeled conditional adaptation.
LRR and Binoculars are not standard probability likelihood ratios merely because
their names include likelihood or ratio.

### Inference is conditional and the history is adaptive

Bootstrap intervals hold fitting and calibration fixed. They are pointwise,
not simultaneous, and do not account for method-selection history. The pilot
was excluded from the final RAID sample, but later methodological corrections
also followed inspected results. Describe that history honestly; do not claim
external preregistration or a wholly untouched confirmatory analysis without
separate evidence. No improvement is required for a run to be valid.

### Documentation is not the manuscript

These files support a Methods section, provenance appendix and result tables.
They are not a finished paper and do not certify the manuscript's theoretical
assumptions. User-owned `paper/` and allocation-writing materials were left
alone. Before asserting a theorem applies to a detector or an attack mechanism,
check that claim separately against the exact model and score definitions.

## 5. Starting points for writing

1. [Primary workflow](primary/SCIENTIFIC_WORKFLOW.md): sources, generation,
   contamination, primary fitting and per-cell provenance.
2. [RAID design](raid/SCIENTIFIC_DESIGN.md): realistic attacks, sample selection,
   rate definition, full/eligible/oracle estimands and comparator limitations.
3. [Clipping definitions](methods/clipping_method.md): shared formulas, direction,
   quantiles, numerical conventions and statistical interpretation.
4. [Final results guide](FINAL_RESULTS_GUIDE.md): accepted files, counts,
   deduplication and table/figure use.
5. [Detector revision ledger](DETECTOR_REVISION.md): what changed and which
   accepted result identities implement each amendment.

For future changes, update the relevant protocol and result lineage together.
Do not edit immutable result manifests merely to make them agree with newer
documentation or code.
