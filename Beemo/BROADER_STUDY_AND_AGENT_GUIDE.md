# Broader study redesign and agent handoff

## Message to Jiaxun

The Granite-XSum splice audit showed that most degradation in arbitrary-token
contamination is explained by construction boundaries. That finding does not
invalidate the existing 21 cells, but it changes their scientific role. They
should be presented as a mechanistic token-splice stress test, not as the main
evidence about normal human editing.

The recommended study now has four layers:

| Layer | Evidence | Scientific role |
|---|---|---|
| 1 | Beemo expert edits | New realistic-edit primary study |
| 2 | RAID | External domain, generator, decoding, and attack robustness |
| 3 | Sentence-aligned human/LLM recomposition | Controlled secondary mechanism study |
| 4 | Existing 21-cell token contamination | Extreme white-box/synthetic stress test |

Finish the Beemo smoke and full run before spending more GPU hours on Qwen
7B/14B/72B synthetic cells. The result determines what the paper's central
claim can be. Already completed synthetic cells remain useful and should not be
discarded.

## Decision table after Beemo

| Beemo result | Interpretation | Next action |
|---|---|---|
| Expert TPR drops; clipping recovers it | Strong support for clipping under real human editing | Add RAID frozen-transfer evaluation, then limited scorer sensitivity |
| Expert TPR drops; clipping does not recover it | Honest negative result; clipping is construction-specific or insufficient | Center paper on failure analysis and detector limits; do not scale synthetic matrix blindly |
| Expert edits remain easy even raw | Beemo may not stress these reference detectors enough | Examine categories/generators/edit ratios, then add RAID and output-only sensitivity |
| Only LLM edits benefit | Clipping helps automated rewriting, not human editing | Report editor-specific effect and avoid a human-edit claim |
| Rank/LRR transfer but log likelihood does not | Detector-dependent mechanism | Focus mechanistic analysis on the transferring detectors; preserve all seven outputs |

Do not decide by one attractive plot. Read `metrics.csv`, paired confidence
intervals, actual held-out FPR, and subgroup sample sizes. The expert condition
and the four core detectors were fixed before the full Beemo test.

## Recommended complete paper structure

1. **Motivation:** realistic outputs often contain machine text followed by
   human or automated revision.
2. **Artifact diagnosis:** paired Granite audit demonstrates why arbitrary
   token splicing is not a sufficient realism model.
3. **Primary experiment:** Beemo expert-edited machine text with leakage-safe
   tuning, human-only calibration, and record-clustered intervals.
4. **External robustness:** RAID, using frozen choices where the protocol is a
   transfer test and clearly separated calibration where a RAID-native baseline
   is desired.
5. **Controlled mechanisms:** sentence-aligned recomposition and automated
   editors explain which gains are editing-related versus boundary-related.
6. **Extreme stress:** selected completed cells from the 21-cell matrix show
   scale, dataset, and white-box behavior; unfinished cells are optional rather
   than automatically required.

## Immediate sequence

1. Publish these local changes yourself and pull them on Delta.
2. Run the 12-record all-detector Beemo gate.
3. If it passes, run the full 2,187-record Beemo study.
4. Review the expert condition first, then automated controls and subgroups.
5. Freeze a written interpretation before implementing RAID.
6. Compare the four preregistered output-only scorer backbones before deciding
   whether the optional Granite sensitivity is scientifically useful.

## Instructions for a new AI agent

Before giving commands or changing code, read, in order:

1. repository `AGENTS.md`;
2. `SCIENTIFIC_WORKFLOW.md`;
3. `CODEBASE_GUIDE.md`;
4. `Beemo/SCIENTIFIC_DESIGN.md`;
5. `Beemo/DELTA_GUIDE.md`;
6. this file.

Then establish machine and state. On Delta, request or inspect:

```bash
cd ~/LLM-detection
pwd
git branch --show-current
git log -1 --oneline
git status --short --branch
readlink -f runs
readlink -f results
df -h /work/hdd/bhuc/$USER
squeue -u jli101
```

Operational rules for the agent:

- Never push or pull; Jiaxun performs Git synchronization.
- Never assume an old Slurm state is current.
- Never place Beemo score packs under `/u`; verify both links resolve below
  `/work/hdd` before submission.
- Do not relabel `human_edits` as human. It is an expert-edited machine-origin
  positive.
- Keep all nine variants of one Beemo ID in one split and bootstrap cluster.
- Do not tune clipping or thresholds on the final test split.
- Do not let six automated edits outweigh the single expert edit; retain equal
  family weighting in clipping tuning.
- Preserve the frozen output-only policy: native tokenizer special tokens and
  no silent truncation for the four single-model scorers; 512 tokens for
  Binoculars.
- A `COMPLETED` Slurm state is insufficient. Validate manifest, stage markers,
  row keys/counts, seven detectors, both FPRs, stderr, and the validation report.
- Preserve the user's existing changes in `plot_tpr_contamination.py` and
  `scripts/export_completed_plots.py`; they predated this package.

## What remains intentionally unimplemented

- RAID loading and frozen-transfer evaluation;
- additional scorer backbones beyond the four primary models and optional
  Granite;
- formal multiplicity-adjusted hypothesis tests;
- a second reference-scoring model sensitivity analysis;
- full-scale Delta timing evidence for Beemo;
- automated comparison against the exact Beemo paper detector configurations.

These are future decisions, not silent omissions. Add them with new protocol
versions, focused tests, and new run IDs.
