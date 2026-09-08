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

The study contains 21 unique cells: three datasets times seven unique target
models.

- Datasets: XSum, SQuAD, and WritingPrompts.
- Models: Granite 3.3 8B Base, Mistral Small 24B Base, Qwen 2.5 32B,
  GPT-NeoX 20B Erebus, and Qwen 2.5 7B/14B/72B.
- Qwen 32B appears in both the primary and scaling descriptions but is one
  reusable model/dataset cell, not a duplicate run.
- Every released full cell includes the six target-model detectors and
  Binoculars. The dedicated 0.5B smoke test intentionally skips Binoculars.
- Full paper settings are in `configs/paper.json`: 3,000 unique sources,
  78,000 prepared/scored rows per cell, random and tail contamination at
  0/5/10/20/30/40/50 percent, three random draws, and 2,000 clustered-bootstrap
  repetitions. The prompt/base decode/re-tokenize integrity guard is 12 tokens,
  and the separately audited constructed-row guard is 20 tokens. Small prompt
  drift is recorded; a base continuation exceeding its guard is
  canonically re-tokenized and length-matched before contamination. Initial and
  final counts, signed drift, normalization status, and realized contamination
  ratio remain recorded.
- The three Granite 8B dataset cells have already produced full prepared data,
  target scores, and Binoculars scores. Their evaluation can be rerun without
  repeating GPU inference.
- Qwen 2.5 32B on XSum has produced a completed full cell. Other cells remain
  dynamic; do not claim that they are running or complete without checking
  Slurm and their manifests.

The Granite-XSum paired splice-artifact audit is a separate diagnostic, not a
22nd paper cell. Its frozen protocol is in
`archive/studies/splice_audit/splice_artifact_audit_granite_xsum.json`, its entry point is
`archive/studies/splice_audit/run_splice_artifact_audit.py`, and its Delta wrapper is
`archive/studies/splice_audit/delta_splice_artifact_audit.sbatch`. It compares human versus
same-model donor text under paired token windows and paired sentence positions,
while reusing the completed Granite-XSum cell's frozen thresholds, directions,
and clipping specifications. Audit outputs belong below
`runs/splice_artifact_audits/` and `results/splice_artifact_audits/`. The local
implementation is not evidence that the Delta gate or full audit has passed;
check its own manifest, row counts, stderr, summary, and completion marker.

The benchmark-centered realistic-edit study lives under `archive/studies/beemo/`. Its frozen
local protocol is `archive/studies/beemo/config.json`; the authoritative entry point is
`archive/studies/beemo/run_beemo.py`; and the scientific and Delta instructions are
`archive/studies/beemo/SCIENTIFIC_DESIGN.md` and `archive/studies/beemo/DELTA_GUIDE.md`. It treats expert edits
as machine-origin positives, keeps all nine variants of a record in one split
and bootstrap cluster, and uses 437/875/875 records for clipping tuning,
human-only calibration, and testing. Local synthetic tests are not evidence of
a real Delta completion; check its own manifest, validation report, score
counts, and stderr before describing it as passed.

## Evaluator status

`experiment_core/analysis/evaluation.py` now precomputes raw and frozen-clipped scalar
scores once per detector/analysis and reuses indexed NumPy arrays for
calibration, point metrics, clustered bootstrap intervals, and robustness AUC.
The local no-download suite passed 38 tests with two Torch-dependent skips when
this work was reviewed. The implementation has not yet received a full-scale
before/after Delta timing benchmark, so do not promise a speedup number.

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

For Beemo, use its separate validation contract: 19,683 prepared rows; 19,683
rows in each of the four primary scorer packs and the shared Binoculars pack;
1,000 metric rows; seven detector methods across 25 scorer-detector
configurations; both FPR targets; and completed `prepare`, `score`, `evaluate`,
`plot`, and `validate` stages. Optional Granite adds one scorer pack, six
configurations, and 240 metric rows.

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
