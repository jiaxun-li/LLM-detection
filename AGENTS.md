# Repository context for agents

The current active/archive file classification is in `docs/REPOSITORY_MAP.md`.
Final study code lives in `experiment_core/`, `RAID/`, and `tools/`; historical
studies and pilot tools live in the Git-ignored `archive/`. Archived programs run
from the repository root and their tests live in `archive/tests/`. The active
`tests/` suite must work without the local archive. Never move or edit local
`paper/` materials as part of experiment maintenance. `downloads/current/`
contains the validated final bundle; older downloads are in `downloads/archive/`.

Read this file before changing code or giving Delta commands. The canonical
scientific description is `docs/primary/SCIENTIFIC_WORKFLOW.md`; the canonical code and
operations reference is `docs/CODEBASE_GUIDE.md`.

## User and repository locations

- User: Jiaxun Li; NCSA Delta username: `jli101`.
- Windows checkout: `E:\Research\LLM detection`.
- Delta login-node checkout: `/u/jli101/LLM-detection` (normally
  `~/LLM-detection`).
- GitHub repository: `https://github.com/jiaxun-li/LLM-detection.git`.
- Stable branch: `main`; use short-lived `codex/` branches for new changes.
- Delta Slurm charge account: `bhuc-delta-gpu`.
- Normal GPU partition: `gpuA100x4`; use `gpuH200x8` only for a deliberately
  reviewed large-model launch such as Qwen 72B.

The user performs every Git push and pull. Agents must never push or pull and
must not rewrite, reset, or discard the user's Git state. Agents may inspect
status, diffs, logs, and remote-tracking refs. Leave changes local for the user
to review and publish.

## Delta environment and storage

- Working CUDA 12.8 virtual environment:
  `/projects/bhuc/jli101/venvs/delta-smoke`.
- Friendly interactive activation: `source tools/delta/activate_environment.sh`.
  Displays `llm-detection` without moving the installed environment. Existing
  batch launchers deliberately retain the working `delta-smoke` path.
- Shared Hugging Face cache:
  `/work/hdd/bhuc/jli101/llm-detection-smoke/huggingface`.
- Torch cache:
  `/work/hdd/bhuc/jli101/llm-detection-smoke/torch`.
- Large run JSONL files, score packs, metrics, and plots belong on
  `/work/hdd/bhuc/jli101`, not under `/u/jli101` and not in Git.
- In the Delta checkout, `runs` and `results` are expected to be links into the
  work filesystem. Never assume their target. Verify before a large launch:

  ```bash
  cd ~/LLM-detection
  readlink -f runs
  readlink -f results
  df -h /work/hdd/bhuc/$USER
  ```

  Both resolved data paths must be under `/work/hdd`. If they are not, stop and
  fix or clarify storage placement before submitting a full job.

Do not download another copy of a model when the shared cache already contains
it. Never store Hugging Face tokens, Duo codes, passwords, or other credentials
in this repository.

## Current scientific study

The accepted release is the nine primary cells plus RAID, not the historical
21-cell matrix in configs/paper.json. Read docs/FINAL_RESULTS_GUIDE.md and
docs/DOCUMENTATION_AUDIT.md alongside the primary/RAID scientific guides before
writing claims or planning reanalysis.

- Primary: XSum, SQuAD, WritingPrompts crossed with Granite 3.3 8B Base,
  Mistral Small 24B Base, Qwen 2.5 32B. Each cell uses 3,000 source groups,
  78,000 prepared/scored rows and 2,000 source-cluster bootstraps. Final
  publication tables have seven detectors and 392 metric rows per cell.
- The broader Erebus/Qwen-scale matrix is proposed/historical, not accepted
  completion evidence. Beemo and splice studies are local ignored archives.
- The seven reported methods include binocular_origin and exclude binocular_gap.
  The base binoculars key still denotes the legacy gap. Use revision entry
  points for final methods; do not relabel saved gap columns as origin.
- Primary origin is prompt-conditioned from saved continuation features.
  RAID origin is separately scored output-only with official component windows.
- Revised LRR direction is fixed +1, origin -1. Other directions are learned
  only from clean tuning. Current v4.2 rejects nonpositive LRR denominator caps.
- Eight accepted primary revisions are September 4 outputs. Qwen32–SQuAD alone
  uses the later lrr-v42-20260907T163410Z correction. Do not rerun all nine merely
  because that cell was amended.
- Accepted RAID revision: raid-origin-anchored-v41-20260907T044156Z,
  12,871 sources after 500 pilot exclusions, 167,323 score rows, 2,000
  bootstraps. Source eight-detector metrics have 416 rows; paper seven-detector
  projection has 364. All six selected LRR specs passed the positive-cap audit;
  do not relabel this v4.1 result as v4.2.
- Final local bundle:
  downloads/current/primary-nine-plus-raid-final-20260908T150247Z.
  It records 182 hashed artifacts, 3,528 primary metrics and 99 primary plots.
  RAID has three source and three paper-view plots. Large JSONL packs remain
  on Delta and must be preserved for reanalysis.
- Current configs are defaults, not every cell's exact settings. Granite saved
  guard 8/hidden states true; other base guards 12; separate constructed guard
  20 is explicitly saved only for Qwen WritingPrompts. Use resolved manifests.
- Clipping selection/calibration and fixed-fit bootstrap details differ between
  primary and RAID. Do not import archived pilot selectors into the final method.
- Preserve source reports even when they describe eight detectors beside a
  filtered seven-detector publication CSV; use enclosing export hashes to
  validate the transformed artifact.
- Completed results do not establish current Slurm jobs, account balances,
  available storage, or compatibility with a new source/dependency version.

## Evaluator status

`experiment_core/analysis/evaluation.py` now precomputes raw and frozen-clipped scalar
scores once per detector/analysis and reuses indexed NumPy arrays for
calibration, point metrics, clustered bootstrap intervals, and robustness AUC.
The user-reported post-reorganization Delta repository smoke (21883773) passed
all 105 active tests without skips. This is engineering evidence, not a new
replay of final inference. No controlled full-scale before/after timing benchmark
establishes an evaluator speedup, so do not promise a speedup number.

For a meaningful evaluator validation, rerun only `--stage evaluate` on an
existing Granite cell, preserve the old `metrics.csv`, verify identical output,
and record elapsed time. Do not regenerate or rescore Granite merely to test
the evaluator.

## Launch and resume rules

- `run_experiment.py` is the authoritative entry point.
- Use a unique, recorded `RUN_ID` for every changed configuration.
- Resume an interrupted job with exactly the same `RUN_ID` and configuration.
- Full jobs must use `configs/paper.json`; bounded checks use a smoke config.
- Binoculars is enabled unless `--skip-binoculars` is explicitly passed.
- The repository Slurm script defaults to four A100s, 240 GB RAM, and 24 hours.
  Command-line `sbatch` resource options override those defaults.
- The Slurm script runs with the repository as its workspace. Therefore verify
  the `runs` and `results` links before submission, or invoke
  `run_experiment.py` with an explicit work-filesystem `--workspace`.
- Delta charges reserved resources rather than observed utilization. Avoid
  excessive GPU, memory, CPU, and wall-time requests.

Useful status checks:

```bash
squeue -u jli101
sacct -j <job-id> --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
tail -F logs/<job-log>.out
tail -n 100 logs/<job-log>.err
```

Completion requires all of the following, not merely a `COMPLETED` Slurm state:

- `manifest.json` reports `completion_status: complete`;
- completed stages are `prepare`, `score`, and `evaluate`;
- row counts and completion markers are present;
- `metrics.csv` includes all seven detectors and both 1% and 5% FPR results;
- stderr contains no hidden traceback, CUDA failure, or offload surprise.

Revised scientific outputs additionally require their revision markers and artifact
hashes. Publication projections have different counts from source compatibility
results; use docs/FINAL_RESULTS_GUIDE.md. Archived Beemo and splice diagnostics
have separate historical contracts and are not part of the final bundle.

## How to start a new agent session

First establish which machine and state the user is discussing. Do not confuse
the Windows checkout with the Delta checkout. On Delta, ask for or inspect:

```bash
cd ~/LLM-detection
pwd
git branch --show-current
git log -1 --oneline
git status --short --branch
readlink -f runs
readlink -f results
accounts
squeue -u jli101
```

Job IDs, balances, queue state, available storage, and active runs are dynamic;
always check them rather than copying an old value from chat or documentation.
