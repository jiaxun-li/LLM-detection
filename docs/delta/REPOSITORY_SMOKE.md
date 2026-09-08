# Reorganized repository: Delta verification

Run `tools/delta/delta_repository_smoke.sbatch` from the repository root after
publishing and pulling the layout changes. Create `logs/` before submission.

This requests one A100, eight CPUs, 32 GB RAM, and a 15-minute limit. It checks
the active unit suite (including small CUDA arithmetic tests), renamed command
entrypoints, and shell syntax. Network/model access is offline. Synthetic tests
use temporary directories; completed scientific runs and results are not edited.
No dataset preparation, model download, generation, or model scoring is performed.

Success requires Slurm `COMPLETED` with `0:0`, no skipped or failed tests, and
`DELTA REPOSITORY SMOKE: PASS` at the end of stdout. This validates the layout
and installed environment, not a new full scientific experiment or real-model
inference gate. Source fingerprints changed with the moves; do not reuse an old
gate as if it identified the reorganized code.

```bash
cd ~/LLM-detection
mkdir -p logs
sbatch tools/delta/delta_repository_smoke.sbatch
```

Use the returned job number to inspect `sacct` and
`logs/repository-smoke-JOB_ID.out` / `.err`. A queue delay is separate from the
test runtime. Do not enable shell-exit-on-error in an interactive terminal.
