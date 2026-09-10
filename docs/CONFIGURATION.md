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

## Which settings produced the final results?

The checked-in JSON is a baseline, not a substitute for the resolved run record.
For the primary study, read each source `manifest.json` and revision
`revision_manifest.json.configuration`; the revision runner copies generation,
contamination, scoring and evaluation groups from that source and applies the
detector amendment. It does not load the current `configs/paper.json` anew.
For RAID, read `revision_manifest.json.protocol_config` and the gated
`scorer_config`, not only `configs/raid.json`. The final exporter accepts a paper
config for export metadata but does not regenerate the historical experiments.

Use this authority order when reporting settings:

1. Validated final metric tables and their exact source/revision identifiers.
2. Resolved source and revision manifests, frozen specifications, validation
   reports and content hashes for those identifiers.
3. Code identified by the recorded implementation/Git identity.
4. Checked-in baseline JSON and general documentation for interpretation.

Do not change a resolved value in a report merely because a newer default is
different. Conversely, a historical metric field named `binoculars` is not
proof that its formula is the final original-paper ratio.

## Primary baseline controls

`configs/paper.json` uses source splits 500/500/2,000, source-selection seed
1731, generation seeds 101/202/303, a 30-token prompt, and a nominal 220-token
continuation with generation minimum 210. Actual final lengths are recorded;
the settings do not imply every visible continuation has exactly 220 tokens.
Temperature/top-p are 0.8/0.95. Three random contamination draws and one tail
condition at each of six positive requested rates produce 26 rows per source
including the two clean rows, hence 78,000 for 3,000 sources.

The evaluation mixture uses requested rates 0.10–0.50 and both modes; the 0.05
condition is evaluated but is not in the configured primary tuning mixture.
Calibration targets are 1% and 5%; bootstrap seed is 481516 with 2,000 resamples.
The original JSON still names the legacy `binoculars`; revised final outputs
use `binocular_origin` and fixed LRR direction through the amendment.

The matrix arrays include planned replication/scaling models. Only the nine
Granite8B/Mistral24B/Qwen32B × dataset cells belong to the current final bundle.
`submit_delta_matrix.sh` also contains its own hardcoded lists: editing JSON
alone does not update that launcher's matrix.

## RAID baseline controls

`configs/raid.json` requires 11 attacks per selected machine generation, one
clean machine document and one human document per source (13 rows). It defines
source selection/split seeds 260823/260824, fractions 0.40/0.20/0.40, exclusion
of 500 development sources, a 512-token right-truncated output-only window,
5% FPR, 2,000 bootstraps with seed 260825, and fixed rate boundaries
0.05/0.10/0.20/0.50. The 14,971 paper source count is reference metadata;
the accepted local release had 13,371 eligible pre-exclusion sources and
12,871 final sources, not 14,971 observed final sources.

The JSON's `required_development_source_exclusions: 500` specifies a required
**count**, not the identities or the location of the pilot exclusion file.
For a new unbounded base run the operator must separately provide
`--exclude-source-ids-path` (or `EXCLUDE_SOURCE_IDS_PATH` through the launcher).
The loader checks uniqueness/count, preparation applies those exact IDs, and
the manifest records exclusion-file provenance. The revision path inherits the
completed source's exclusions; it does not invent a fresh list from the config.
Two different 500-ID lists are not scientifically interchangeable.

Full-universal and eligible-universal are separate fits; the latter uses
positive rate up to 0.50. The code's revision-enabled selector, frozen bounds,
and selected/no-clipping status must be read with the scientific RAID guide.
Fields retained for historical compatibility are not automatically operative
hyperparameters in every revision path. Never infer the final objective solely
from older `attack_weight`/`clean_weight` values in this baseline.

## Editing and launch safety

The general primary CLI supports repeated `--set dotted.key=value` overrides,
but base preparation cache keys do not encode every algorithm setting.
Changed scientific settings require a new run ID. The primary revision CLI
instead accepts source ID, new revision ID, workspace and optional bootstrap
count. RAID revision has stage/shard/reference/adoption options; it is not the
same CLI as `RAID/run_raid.py`. Check the relevant `--help` before constructing
a new launch. Never replace saved manifests or frozen specs to make an existing
run appear compatible with a changed configuration.
