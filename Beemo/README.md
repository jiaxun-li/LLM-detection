# Beemo experiment package

This folder contains the benchmark-centered replacement study for realistic
human editing. It evaluates the repository's raw and clipped detector
aggregations on the expert-edited machine outputs released in
[Beemo](https://huggingface.co/datasets/toloka/beemo).

The primary protocol scores response text only with GPT2-XL, OPT-1.3B,
Falcon-7B, and Qwen2-7B. Granite is optional, and Binoculars remains one shared
Falcon-pair baseline.

Start with:

1. [`SCIENTIFIC_DESIGN.md`](SCIENTIFIC_DESIGN.md) for the estimands and split
   policy.
2. [`DELTA_GUIDE.md`](DELTA_GUIDE.md) for the smoke gate, full submission, and
   validation commands.
3. [`BROADER_STUDY_AND_AGENT_GUIDE.md`](BROADER_STUDY_AND_AGENT_GUIDE.md) for
   how this changes the whole research program and what a future AI agent must
   check before acting.

Main executable files:

- `run_beemo.py`: authoritative all-stage entry point;
- `prepare_beemo.py`, `score_beemo.py`, `evaluate_beemo.py`, and
  `plot_beemo.py`: single-stage convenience entry points;
- `validate_beemo.py`: independent artifact validation;
- `delta_beemo.sbatch`: two-GPU Delta job;
- `submit_beemo.sh`: guarded full-run submission helper.

Large data are written below `runs/beemo/<run-id>` and results below
`results/beemo/<run-id>`. On Delta, the repository-level `runs` and `results`
links must resolve under `/work/hdd`.
