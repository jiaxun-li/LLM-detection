# Experiment execution status

The conference pipeline implementation and local no-download validation are
complete. Use [`REQUIREMENT_CHECKLIST.md`](REQUIREMENT_CHECKLIST.md) for the
audited requirement map and [`DELTA.md`](DELTA.md) for the remaining
cluster-only validation.

The historical `real_data/`, `real_results/`, and `greatlake/` outputs are
smoke-scale artifacts from the former two-way-split prototype. Preserve them,
but do not combine them with metrics from `run_experiment.py`.
