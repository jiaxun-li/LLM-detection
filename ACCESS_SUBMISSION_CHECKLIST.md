# ACCESS Submission Checklist

This folder is close to a useful ACCESS allocation support package, but it
still needs a few submission-facing pieces before upload.

## Ready

- End-to-end code exists for data preparation, scoring, evaluation, and plots.
- Slurm scripts exist for NCSA Delta GPU execution.
- A bounded one-GPU Delta smoke workflow and a no-network synthetic validator
  are present. No real GPU smoke result is bundled in this repository.
- `requirements.txt` lists the main Python dependencies.

## Fix Before Sharing

- Run the bounded Delta smoke gate, then at least one larger benchmark. Treat
  the synthetic smoke as contract validation, not evidence of scientific
  stability.
- Decide whether to include generated JSONL score files. They are useful for
  reproducibility, but generated data can distract from the code package.
- Remove or avoid sharing local-only files such as `.DS_Store`, `__pycache__/`,
  `.cache/`, and Slurm `logs/`.
- Replace personal paths and usernames in public-facing docs if this folder is
  shared outside the lab.
- Add the PI/advisor CV or resume PDF separately. ACCESS requires PDF CVs for
  the PI and any co-PIs.
- If the requester is a graduate student, add a signed advisor-support letter
  on institutional letterhead as a PDF.

## Proposal Text To Prepare

- Project title.
- Short abstract covering the scientific question, ACCESS resource use, and
  required software.
- Keywords and field of science.
- Resource justification: number of datasets, models, samples, GPU count,
  expected wall time per job, total GPU-hours, and storage estimate.
- Computational plan: explain the three-stage workflow:
  data construction, token-feature scoring, and clipping evaluation.
- Resource appropriateness: justify GPUs, model memory, Hugging Face model
  downloads, and storage for JSONL scores and plots.
- Other compute resources: describe local or campus resources and why ACCESS is
  needed for larger model and dataset sweeps.

## Suggested ACCESS Project Type

- Explore ACCESS is enough for initial benchmarking, code development, and
  graduate-student-scale validation.
- Discover ACCESS is a better fit if you already know you need a modest GPU
  allocation for several datasets and model families.
- Accelerate ACCESS requires a stronger quantitative resource estimate and a
  three-page project description.

## Evidence To Generate

- A smoke-test log showing the workflow completes.
- A timing table from one representative run:
  dataset, model, samples, GPU type, GPUs, wall time, peak memory if available,
  output size.
- Final plots for random and tail contamination at 1% and 5% FPR.
- A short note explaining why the smoke-test curves are coarse and why larger
  ACCESS runs are needed.
