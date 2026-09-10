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

Recorded verification: the user supplied Delta job `21883773`, completed with
exit code `0:0` in 37 seconds, all 105 then-active tests passing without skips,
nine command-path checks and the final PASS message. This is a historical
environment/layout check for the recorded checkout, not a permanent guarantee
for later dependency/code changes and not a throughput estimate for inference.
Tests use synthetic/fake model inputs where appropriate. They cannot establish
real-model parity on every possible sequence.
