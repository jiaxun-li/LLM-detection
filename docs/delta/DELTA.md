# NCSA Delta operations

The current report uses the completed nine primary cells plus corrected RAID.
Do not launch the broader configured matrix or repeat inference just to read,
validate, consolidate or plot those results. Available launchers are not
evidence that all configurable experiments were performed.

## Machine, branch, environment and storage

The commands below are **Delta Bash**, not Windows PowerShell. Git pushes and
pulls are performed by the user. Stable branch: `main`; feature branches use
`codex/`. The friendly environment name is `llm-detection`, but its physical
installed path remains `/projects/bhuc/jli101/venvs/delta-smoke`.

```bash
cd ~/LLM-detection
git branch --show-current
git log -1 --oneline
git status --short --branch
source tools/delta/activate_environment.sh
readlink -f runs
readlink -f results
df -h /work/hdd/bhuc/$USER
accounts
squeue -u "$USER"
```

The activation helper loads modules, activates the existing venv and changes
only its displayed prompt. Do not rename its directory or reinstall packages
for ordinary result inspection. Batch launchers retain the working path;
`deactivate` still works.

Before large work, both resolved result/run paths must be below `/work/hdd`.
Use the shared HF cache `/work/hdd/bhuc/jli101/llm-detection-smoke/huggingface`
and Torch cache `/work/hdd/bhuc/jli101/llm-detection-smoke/torch`. Storage links
belong to each checkout: `LLM-detection` and `LLM-detection-fast-smoke` may
resolve to different data. Small downloaded bundles do not replace the score
packs needed for reanalysis; preserve original packs and manifests.

The recorded allocation is `bhuc-delta-gpu`; check `accounts` for current
availability. CPU-only work may require one reserved GPU on `gpuA100x4`
because of the account type. `Priority`/`Resources` means queued, `Dependency`
means waiting, and `DependencyNeverSatisfied` requires reviewing a failed
prerequisite. Available allocation hours do not guarantee immediate scheduling.

## Choose the relevant program

| Task | Entry point | Model inference? |
|---|---|---|
| Read final tables/figures | downloaded final bundle or preserved results | No |
| Check layout/environment | `tools/delta/delta_repository_smoke.sbatch` | No; tiny tensor tests only |
| Test real primary preparation/resume | `tools/delta/submit_delta_smoke.sh` | Qwen0.5B, no Binoculars |
| Revised primary evaluation | `tools/reanalysis/reevaluate_detector_revision.py` | No |
| Official RAID origin gate/scoring | `RAID/revise_detectors.py --stage gate/score` | Yes, except validated adopted rows |
| Corrected RAID evaluation | `RAID/revise_detectors.py --stage evaluate` | No |
| Final bundle/plots | `tools/exports/export_final_publication_bundle.py` | No |
| New primary base experiment | `run_experiment.py` | Yes; historical base protocol |
| New RAID base experiment | `RAID/submit_raid.sh` | Yes; historical base protocol |

See [repository smoke](REPOSITORY_SMOKE.md), [real-model smoke](DELTA_SMOKE.md),
[detector amendment](../DETECTOR_REVISION.md),
[RAID operations](../raid/DELTA_GUIDE.md), and
[final-result guide](../FINAL_RESULTS_GUIDE.md). The base workflow does not
automatically substitute the final corrected detector definitions.

## Wrapper resources and defaults

| Wrapper under `tools/delta/` | GPUs | CPUs | RAM | Wall limit |
|---|---:|---:|---:|---:|
| `delta_repository_smoke.sbatch` | 1 | 8 | 32G | 15 min |
| `delta_smoke_qwen_0_5b.sbatch` | 1 | 8 | 32G | 45 min |
| `delta_experiment.sbatch` | 4 | 32 | 240G | 24 h |
| `delta_detector_revision.sbatch` | 2 | 16 | 120G | 4 h |
| `delta_export_final_publication.sbatch` | 1 | 8 | 32G | 1 h |

Command-line `sbatch` options override defaults. Evaluation does not need two
GPUs intrinsically; request the account-required GPU and adequate RAM/time.
A wall limit is not a runtime estimate.

The general primary wrapper defaults to `$PWD/.venv` and
`$PWD/.cache/huggingface`, unlike the revision wrapper. Before an approved new
general primary launch export:

```bash
export VENV_PATH=/projects/bhuc/$USER/venvs/delta-smoke
export HF_HOME=/work/hdd/bhuc/$USER/llm-detection-smoke/huggingface
export TORCH_HOME=/work/hdd/bhuc/$USER/llm-detection-smoke/torch
```

`submit_delta_matrix.sh` supports `primary`, `replication`, `qwen-scaling`, and
`all` using its own hardcoded lists. It accepts A100 and explicit `gpuH200x8`;
Qwen72B is skipped by that wrapper on A100. Separate calls can duplicate Qwen32B
even though `all` deduplicates it within one invocation. Do not use it to
reproduce an existing final result merely because the code is reorganized.

## Safe submission and monitoring

Create `logs/` before `sbatch`: Slurm opens log paths before the script starts.
Record IDs only after successful submission. Do not enable shell-exit-on-error
in an interactive terminal; contain strict commands in a subshell or batch
script. This example is only the no-download repository check:

```bash
(
    set -euo pipefail
    cd ~/LLM-detection
    mkdir -p logs
    JOB_ID=$(sbatch --parsable tools/delta/delta_repository_smoke.sbatch)
    JOB_ID=${JOB_ID%%;*}
    printf '%s\n' "$JOB_ID" > logs/last_repository_smoke_job.txt
    echo "Submitted: $JOB_ID"
)
```

```bash
cd ~/LLM-detection
JOB_ID=$(cat logs/last_repository_smoke_job.txt)
squeue -j "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
tail -n 30 "logs/repository-smoke-$JOB_ID.out"
tail -n 80 "logs/repository-smoke-$JOB_ID.err"
```

For other programs use their actual `#SBATCH --output/--error`, not an inferred
job-name pattern. `scontrol show job JOB_ID` can show `StdOut`/`StdErr` for
available job records. A missing log is not a clean log. Repository smoke
requires `COMPLETED`, `0:0`, no skips/failures and its final PASS message; it is
not a real-model parity gate. Scientific jobs additionally require their own
manifest, markers, exact keys/counts and artifact hashes.

## Reuse and historical work

Changed settings require a new run/revision ID. An exact resume must match
configuration and code identity. Relocated source files changed implementation
fingerprints: an old gate is not a gate for the reorganized tree. This does not
invalidate completed artifacts under their recorded identities or require
regenerating them.

`archive/` and `downloads/` are ignored and machine-local. Git does not transfer
them. Historical Beemo/pilot/splice commands require independently preserved
scripts and exact sources. Do not delete artifacts merely because Git ignores
them. Only a deliberately new installation should run `setup_delta_env.sh`;
optional vLLM needs a separately reviewed compatible environment. Never store
credentials in the repository.
