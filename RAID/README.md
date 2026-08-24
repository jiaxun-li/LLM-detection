# RAID benchmark implementation

The frozen scientific protocol is [`SCIENTIFIC_DESIGN.md`](SCIENTIFIC_DESIGN.md).
Delta setup, smoke-gate, submission, resume, and validation commands are in
[`DELTA_GUIDE.md`](DELTA_GUIDE.md).

The scientific entry point is `run_raid.py`. Delta launches must use the
dependency-safe wrapper so CPU preparation, independent GPU shards, and final
evaluation receive appropriate resources:

```bash
bash RAID/submit_raid.sh
```

A completed bounded preparation can be adopted under a corrected run ID by
setting `ADOPT_PREPARED_RUN_DIR`; adoption verifies exact prepared and shard
keys before scoring begins.

`run_raid_shard.py` is called only by the Slurm array wrappers. Do not launch
multiple copies of `run_raid.py --stage score` against one run directory.

Bounded checks must use `--limit-sources` and remain `debug_only`. Full results
must pass `RAID/validate_raid.py`; a Slurm `COMPLETED` state alone is not enough.

`compare_tuning_methods.py` is a separate, CPU-only pilot analysis. It reuses a
bounded run's completed Falcon and Binoculars score packs, never changes the
frozen evaluator, and writes only below
`results/raid/<run-id>/tuning_comparison_v1/`. Its 11 tuning methods and four
clean-loss budgets are exploratory model selection, not final RAID results.
Its recorded development source IDs must be excluded from the future full-run
split because pilot test outcomes are used to choose the final selector.

`compare_trimmed_mean.py` is another isolated, CPU-only pilot analysis. It
tests one-sided token trimming at fractions 0, 0.5%, 1%, 2.5%, 5%, 10%, and
20% under full-universal and eligible-universal fitting. It reuses the same
score packs and writes only below
`results/raid/<run-id>/trimmed_mean_comparison_v1/`.

`compare_binoculars_components.py` is a Binoculars-only pilot ablation. It
compares the existing local-gap clipping rule with clipping the oriented
performer-NLL and cross-entropy components separately. The two component bounds
share one quantile index, avoiding a post-pilot 49-pair search. It writes below
`results/raid/<run-id>/binoculars_component_comparison_v1/`.
