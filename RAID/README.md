# RAID benchmark implementation

The frozen scientific protocol is [`SCIENTIFIC_DESIGN.md`](SCIENTIFIC_DESIGN.md).
Delta setup, smoke-gate, submission, resume, and validation commands are in
[`DELTA_GUIDE.md`](DELTA_GUIDE.md).

The authoritative entry point is:

```bash
python RAID/run_raid.py --run-id <unique-id> --data-path <train.csv> --stage all
```

Bounded checks must use `--limit-sources` and remain `debug_only`. Full results
must pass `RAID/validate_raid.py`; a Slurm `COMPLETED` state alone is not enough.

