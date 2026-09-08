# Experiment configurations

A configuration is the experiment's settings file. The Python code implements
the procedure; a JSON configuration supplies its inputs and numerical choices.
For example, `bootstrap_repetitions: 2000` asks the evaluator for 2,000 bootstrap
resamples. It is not a model checkpoint, dataset, or result file.

All active configuration files live in the repository's `configs/` directory:

| File | Purpose |
|---|---|
| `configs/paper.json` | Original primary protocol: datasets/model matrix, 3,000 sources per cell, generation, contamination, calibration, and 2,000 bootstraps |
| `configs/raid.json` | RAID data selection, split fractions, scorers, rate groups, tuning, and 2,000 bootstraps; formerly `RAID/config.json` |
| `configs/smoke.json` | Small generic primary pipeline check |
| `configs/delta_smoke_qwen_0_5b.json` | Dedicated bounded Delta Qwen 0.5B engineering check |

The JSON values were preserved byte-for-byte during the repository reorganization.
Primary and RAID detector corrections are applied by the dedicated revision
entry points described in [DETECTOR_REVISION.md](DETECTOR_REVISION.md); do not
interpret the original base configuration alone as the complete final amendment.

Keep these files tracked in Git: their settings are needed for reproducibility.
Historical Beemo and splice-audit configurations remain with those studies in
the ignored local `archive/`. Do not mix smoke results with final-study results.

To design a different experiment, copy the relevant settings into a deliberately
named new configuration and use a new run ID. Do not silently edit the frozen
settings of a completed run. Existing manifests, frozen specifications, and
downloaded protocol snapshots record what actually produced a released result.

Config paths are passed by `--config` where supported. Run commands from the
repository root. Relative output paths in the settings still point to `runs/`
and `results/`; the reorganization did not relocate Delta datasets or score packs.
