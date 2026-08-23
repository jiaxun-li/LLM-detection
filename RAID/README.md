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
