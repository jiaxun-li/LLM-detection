# Codebase guide

For preservation-safe primary/RAID detector reanalysis, use the new entry points
and Delta wrapper in [docs/DETECTOR_REVISION.md](DETECTOR_REVISION.md). Legacy entry
points intentionally retain their old scientific configuration.

This is the canonical implementation and operations guide. For the research
design and interpretation rules, see
[`docs/primary/SCIENTIFIC_WORKFLOW.md`](primary/SCIENTIFIC_WORKFLOW.md). Commands that load datasets,
models, CUDA, modules, or Slurm are marked **cluster-only**; the unit and
synthetic tests require no downloads.

## Evidence and protocol layers

The released study is **nine primary cells plus RAID**, not the entire 21-cell
model matrix still listed in `configs/paper.json`. That larger matrix is a
historical design/launcher capability, not evidence of completed experiments.
Use the final bundle's per-cell source/revision identities when writing a
report; do not select the newest directory by its timestamp.

Three layers must be distinguished:

1. `run_experiment.py` and `RAID/run_raid.py` retain the original preparation,
   scoring and evaluation workflow. Their `binoculars` name denotes the legacy
   exponential mean-gap calculation, not the corrected original-paper ratio.
2. `tools/reanalysis/reevaluate_detector_revision.py` and
   `RAID/revise_detectors.py` apply the detector amendment to preserved sources.
   Primary uses saved prompt-conditioned features. RAID uses separately gated
   output-only official-origin components. LRR direction and candidate guards
   are enabled by the revision configuration, not by renaming a CSV column.
3. `tools/exports/export_final_publication_bundle.py` validates and packages
   accepted results. It filters the legacy gap from paper-facing tables and
   generates plots; it does not refit bounds, thresholds, or bootstrap intervals.

The primary final bundle combines eight September 4 corrected revisions with
the later Qwen32–SQuAD LRR-v4.2 correction. Retaining source eight-detector
reports beside filtered seven-detector tables is intentional provenance, not
a contradiction in the reported detector set. See the final bundle manifest
and [DETECTOR_REVISION.md](DETECTOR_REVISION.md).

## Active repository map

See [docs/REPOSITORY_MAP.md](REPOSITORY_MAP.md) for script purposes and the complete
active/archive classification.

```text
.
|-- run_experiment.py             # Primary preparation/scoring/evaluation
|-- plot_tpr_contamination.py     # Primary scientific figures
|-- experiment_core/             # Preparation, detectors, analysis and infrastructure
|-- RAID/                        # Active external benchmark and origin revision
|-- tools/                       # Delta launchers, validation, reanalysis and export
|-- configs/                     # Primary, RAID and bounded smoke settings
|-- docs/                        # Experiment methodology, operations and file map
|-- tests/                       # Active regression coverage (no archive dependency)
|-- archive/                     # Local-only historical code and tests (Git-ignored)
|-- downloads/current/           # Verified final primary + RAID export (local)
|-- downloads/archive/           # Older downloads (local)
`-- paper/                       # User-owned writing materials (local)
```

`run_experiment.py` remains the primary stage entry point. Revised detector
evaluation uses `tools/reanalysis/reevaluate_detector_revision.py` and
`RAID/revise_detectors.py`, as specified in
[docs/DETECTOR_REVISION.md](DETECTOR_REVISION.md). The general pipeline retains its
historical configuration; relocating files does not change detector formulas.

Deprecated stage aliases, toy experiments, Beemo, the paired splice diagnostic,
pilot selector comparisons, historical exporters and the old monolithic RAID
launcher now live under `archive/`. Retained archived programs reference shared
implementation and are intended to run from the repository root. This is not
a guarantee of runtime compatibility or historical numerical reproduction.
Their tests are in `archive/tests/` and can be run separately when present.
The active suite under `tests/`
does not import the local archive. See [ARCHIVE_CATALOG.md](ARCHIVE_CATALOG.md).

The final exporter uses active `tools/exports/export_helpers.py`; it does not
depend on the archived Beemo exporter. Python caches and `.DS_Store` are ignored
maintenance artifacts, not evidence of experiment completion. The allocation
checklist and `paper/` are user-owned writing/planning material and are outside
the scope of experiment/documentation maintenance.

## Configuration and command dispatch

[`load_config`](../experiment_core/infrastructure/config.py) reads JSON, injects `_config_path`,
applies repeated dotted `--set key=value` overrides, and calls
`validate_config`. Override values are parsed as JSON when possible, so
numbers, booleans, `null`, arrays, and quoted strings retain their types; an
unparseable value remains a string. Missing intermediate dictionaries are
created. [`resolved_run_config`](../experiment_core/infrastructure/config.py) deep-copies the
configuration and adds the selected dataset, target model, run ID, and revision
fields.

The CLI fixes the dataset names to `xsum`, `squad`, and `writingprompts`, but
accepts any `--model` string. A model is not automatically checked against the
configured primary/replication/scaling lists. `--skip-binoculars` removes that
detector from the resolved scoring list and must also be passed on a later
standalone evaluation stage for a run without a Binoculars file.

Compact no-download parsing example:

```bash
python -c "from experiment_core.infrastructure.config import load_config; load_config('configs/paper.json')"
```

Example override syntax:

```bash
python run_experiment.py --config configs/smoke.json \
  --dataset xsum --model Qwen/Qwen2.5-0.5B --run-id local-contract \
  --stage select-sources --set generation.batch_size=4 \
  --set 'generation.device="cuda:0"'
```

The example's source-selection stage is **cluster/network-only** if the shared
manifest does not already exist. Avoid reusing a run ID after changing an
override: the general manifest compatibility check covers run ID, dataset, and
target model, not the complete resolved configuration.

## Source manifests, splits, and seeds

[`source_manifest_path`](../experiment_core/preparation/pipeline.py) builds:

```text
<workspace>/<paths.source_manifests>/
  <dataset>-<hf-split>-<count>-seed<selection-seed>-<signature>.jsonl
```

The signature covers the dataset specification, split map, and generation seed
list. It deliberately excludes target model. If a manifest exists,
[`select_source_manifest`](../experiment_core/preparation/pipeline.py) validates its disjoint
split counts and reuses it. Otherwise it streams the configured Hugging Face
dataset, obtains its resolved revision, normalizes text, deduplicates by text
hash, and retains the smallest deterministic hash priorities in bounded memory.
SQuAD contexts are deduplicated; short unique contexts are packed sequentially
without reuse until `minimum_source_words` is reached.

[`assign_splits_and_generation_seeds`](../experiment_core/preparation/data.py) selects the
configured number of unique sources, labels the three splits in the fixed order
`clipping_tuning`, `calibration`, `test`, and assigns one generation seed
round-robin to each selected source. `sample_id` equals the model-independent
`source_id`. [`validate_disjoint_splits`](../experiment_core/preparation/data.py) rejects
duplicates or wrong counts.

**Cluster/network-only source selection:**

```bash
python run_experiment.py --config "$PWD/configs/paper.json" --dataset xsum \
  --model Qwen/Qwen2.5-32B --stage select-sources \
  --workspace "/work/hdd/bhuc/$USER/llm-detection-new-experiment"
```

The target model argument is required by the common CLI but does not influence
the manifest.

## Prepare stage: generation, cache, and contamination

The prepare path in [`prepare_run_data`](../experiment_core/preparation/pipeline.py) has three
restartable products:

1. `generate_base_examples` tokenizes the model-independent source, uses the
   configured prompt prefix (30 tokens in the paper baseline), takes the next source tokens as the
   human continuation, and generates the LLM continuation. The realized human
   and LLM sides are length matched. `TransformersBackend` groups requests by
   assigned seed and length bucket; `VLLMBackend` is an optional generation-only
   accelerator. It records the decoded prompt's actual re-encoded input IDs and
   signed length drift. Small drift is accepted within the configured integrity
   guard. A continuation exceeding that guard is converted to a stable visible
   text/token representation and re-matched to its human pair; initial counts,
   drift, and the normalization flag are retained in `base_generations.jsonl`.
2. `build_tail_cache` sentence-splits the human continuation, batches
   prompt-plus-span candidates, scores each candidate exactly once by
   target-model NLL, and writes the frozen descending order to
   `tail_candidate_cache.jsonl`. If generation used vLLM, this stage loads a
   Transformers target model because vLLM candidate scoring is intentionally
   unsupported.
3. `construct_contaminated_data` writes the two clean rows, random
   non-overlapping span replacements, and deterministic suffix-tail
   replacements. It recomputes final token length after decode/re-tokenize and
   stores requested/realized ratios and token-count provenance.

The `contamination.tail_model_revision` field currently has no call-site;
tail selection uses the resolved target-model/tokenizer revisions. Changing
that field alone does not change the cache. A separate tail model would require
an implementation change.

Random corruption seeds are deterministic hashes of the base corruption seed,
dataset, target model, sample ID, and draw ID. Thus source identities are shared
across models, while random placement streams are target-model specific.
[`data_row_key`](../experiment_core/preparation/data.py) contains dataset, target model, sample,
split, label, mode, requested ratio, and draw ID.

## Score stage: exact reusable features

[`score_run`](../experiment_core/preparation/pipeline.py) first creates a
[`TargetModelScorer`](../experiment_core/detectors/scoring.py). Rows are length bucketed,
microbatched, and streamed by [`score_jsonl`](../experiment_core/detectors/scoring.py).
A single target-model forward supplies all six single-model detectors.

For every continuation token, [`exact_token_features`](../experiment_core/detectors/scoring.py)
computes:

- `logp = target_logit - logsumexp(all_logits)`;
- exact one-based competition `rank = 1 + count(logit > target_logit)`, so ties
  share rank;
- `log_rank = log(rank)`;
- exact full-vocabulary entropy as `log Z - E_p[logit]`.

Vocabulary chunks bound temporary rank, normalization, and entropy work without
approximating those values. The model's full output logits still exist, so this
is not a claim that scoring memory is independent of vocabulary size.
`single_model_doc_scores` then saves mean log likelihood, mean rank, mean log
rank, `mean(NLL)/(mean(log_rank)+epsilon)`, mean entropy, and mean
`NLL-entropy`.

Each `target-token-features-v2` row also stores the configured top-k token IDs
and log probabilities, top-1/top-2 log-probability margin, and observed-target
minus top-1 margin. The paper baseline and accepted source manifests record
`saved_top_k: 10`. The optional
`document_features.mean_pooled_final_hidden_state` flag defaults to false in
`configs/paper.json` and `configs/smoke.json`, and true in the dedicated Delta
smoke. The accepted Granite source manifests also record it as true; the other
six final primary cells record false. These saved vectors are not consumed by
the seven reported detector formulas or the compact evaluator. Report this as
a source-specific feature-storage choice, not a universal paper-run default. Existing target
score rows are rejected on resume if their schema, resolved model/tokenizer
revision, top-k width, or pooled-hidden setting differs.

In the general primary score stage, if Binoculars is enabled, target-model
objects are released first and [`BinocularsScorer`](../experiment_core/detectors/scoring.py)
uses the configured Falcon pair. `configs/paper.json` defaults to:

- performer `tiiuae/falcon-7b-instruct` on `cuda:0`;
- observer `tiiuae/falcon-7b` on `cuda:1`.

Those device numbers describe this primary baseline, not all historical
allocations or the separate RAID scorer; RAID's config assigns observer to
`cuda:0` and performer to `cuda:1`. Model roles determine the cross-entropy
direction; a GPU index is placement metadata, not a detector definition.

It verifies tokenizer compatibility, transfers observer logits to the performer
device, computes exact `H(observer, performer)` in vocabulary chunks, and saves
the token arrays needed to reconstruct both the archived exponential mean-gap
diagnostic and the revised primary conditional ratio
`mean performer NLL / mean cross-entropy`. New primary revision outputs report
only the ratio as `binocular_origin`; its clipping caps numerator NLL and keeps
the denominator fixed. Binoculars output keys must match target-score keys
before evaluation.

## Evaluate stage: ordering, scalar cache, and timing

[`evaluate`](../experiment_core/analysis/evaluation.py) reads score JSONL into compact rows.
[`compact_evaluation_row`](../experiment_core/analysis/evaluation.py) drops text, prompt,
document-level hidden vectors, top-k arrays, and margins while retaining only
the token arrays used by evaluation.

The ordering is deliberately leakage-safe for each configured detector:

1. `_clean` extracts tuning human and clean LLM rows.
2. `orientation` learns score direction from those tuning rows for the general
   detectors. With the revision enabled, LRR is fixed to `+1`; Binocular-origin
   is always fixed to `-1` (smaller un-oriented ratio is more machine-like).
3. `tune_clipping_spec` searches the configured quantiles using only the
   configured tuning mixture. Generic detectors floor oriented local
   contributions at a lower bound; LRR separately caps NLL and log-rank before
   recomputing the ratio. Revised origin caps numerator NLL only. Revised
   candidate guards reject nonempty constant-score candidates and invalid LRR
   zero-denominator caps; the no-clipping candidate remains available.
4. Only after selection is complete,
   [`precompute_evaluation_scores`](../experiment_core/analysis/evaluation.py) traverses every
   row once and creates contiguous `float64` raw and clipped arrays plus
   `sample_id -> int64 row-index array` groups.
5. Calibration and test selection use row indices into those arrays.
   Bootstraps never reconvert token arrays, reapply clipping, or recompute LRR.

One machine-readable stdout event is emitted per detector and analysis:

```text
evaluation_precompute_timing={"analysis":"primary_frozen_mixture",...}
```

It includes detector, elapsed seconds, row count, and raw/clipped score counts.
Timing is instrumentation only; it is not stored in `metrics.csv`. No controlled
before/after speedup benchmark was established by this documentation audit;
ordinary elapsed-job logs alone cannot establish an optimization speedup.

Thresholds are conservative upper order statistics from calibration-human
scores. Final metrics use test humans and the requested clean/attacked LLM rows.
The primary output ordering follows configured detector order, then attack mode,
ratio, target FPR, and raw/clipped aggregation.
[`write_tidy_csv`](../experiment_core/analysis/evaluation.py) writes to a temporary file and
atomically replaces `metrics.csv`.

## Bootstrap implementation

`_metric_bootstrap` and `_robustness_bootstrap` use the precomputed scalar
arrays. `_selected_source_indices` groups the requested row indices by
`sample_id`; each repetition samples sorted source IDs with replacement using
`random.Random`. Every row belonging to a sampled source is concatenated,
preserving random-draw clustering. Raw and clipped metrics use the same sampled
IDs. Percentile intervals use the 2.5% and 97.5% quantiles.

The selected bounds and calibration thresholds are **held fixed** inside these
test-source bootstraps. The intervals quantify test-source sampling uncertainty
conditional on the fitted procedure, not uncertainty from retuning or
recalibrating the entire experiment. Report actual held-out FPR alongside the
nominal calibration target; a target of 5% does not force test FPR to equal 5%.

Condition-specific seeds equal the configured bootstrap base seed plus a stable
hash of detector/condition fields. The CSV records the base seed; the derivation
is deterministic in code. Bootstrap loops are serial and metrics are not
vectorized.

## Inputs, outputs, and row validation

With the paper's relative paths, `--workspace <root>` produces:

| Stage | Required input | Main output | Full-scale count per cell |
|---|---|---|---:|
| Select | dataset stream or existing shared manifest | `runs/source_manifests/<signature>.jsonl` | 3,000 |
| Prepare | source manifest, target generator/tokenizer | `base_generations.jsonl` | 3,000 |
| Prepare | base generations, target NLL scorer | `tail_candidate_cache.jsonl` | 3,000 |
| Prepare | base generations and tail cache | `data.jsonl` | 78,000 |
| Score | `data.jsonl`, target model | `target_scores.jsonl` | 78,000 |
| Score | `data.jsonl`, Falcon pair | `binoculars_scores.jsonl` | 78,000 |
| Evaluate | target scores and configured Binoculars scores | `results/<run-id>/metrics.csv` | 392 primary-analysis rows |

The 392 metrics rows are
`7 detectors x 2 modes x 7 ratios x 2 target FPRs x 2 aggregations`.
Ratio-zero prepared text is stored once but evaluated as the common clean
baseline on both random and tail curves.

The following illustrate the **general base pipeline**, not the accepted
detector-revision replay. They can create/update artifacts and are not a
recommended next step for completed cells. Use an approved new run ID and
verify storage before generation/scoring. Source selection may require network;
evaluation can reuse saved features without inference:

```bash
python run_experiment.py --config configs/paper.json --dataset xsum \
  --model Qwen/Qwen2.5-32B --run-id paper-xsum-qwen32 --stage prepare
python run_experiment.py --config configs/paper.json --dataset xsum \
  --model Qwen/Qwen2.5-32B --run-id paper-xsum-qwen32 --stage score
python run_experiment.py --config configs/paper.json --dataset xsum \
  --model Qwen/Qwen2.5-32B --run-id paper-xsum-qwen32 --stage evaluate
```

Validate counts without loading whole JSONL files into shell variables:

```bash
wc -l "runs/source_manifests/<manifest>.jsonl" \
  runs/paper-xsum-qwen32/base_generations.jsonl \
  runs/paper-xsum-qwen32/tail_candidate_cache.jsonl \
  runs/paper-xsum-qwen32/data.jsonl \
  runs/paper-xsum-qwen32/target_scores.jsonl \
  runs/paper-xsum-qwen32/binoculars_scores.jsonl
```

`prepare_run_data` validates unique base/tail/data keys and exact expected
counts before `prepare.complete.json`. `score_jsonl` rejects duplicate completed
keys and requires output count to match input count before
`score.complete.json`. Evaluation writes `evaluate.complete.json` only after
the CSV is created.

## Paired Granite-XSum splice-artifact audit

[`archive/studies/splice_audit/run_splice_artifact_audit.py`](../archive/studies/splice_audit/run_splice_artifact_audit.py) is a bounded,
isolated historical follow-up to the broader proposed model matrix. It reuses the completed
Granite-XSum cell named in
[`archive/studies/splice_audit/splice_artifact_audit_granite_xsum.json`](../archive/studies/splice_audit/splice_artifact_audit_granite_xsum.json)
and is designed not to modify that cell. The archived default protocol selects
500 test sources, generates one seed-404 alternative Granite continuation for
each source, and constructs one common clean row plus four paired conditions at
10/30/50 percent contamination. Thus the default audit writes 500 alternative
generations and `500 x (1 + 4 x 3) = 6,500` data, target-score, and Binoculars
rows.

The token-level human condition uses the exact primary-study draw-zero random
replacement plan. Its LLM-donor pair reuses the same recipient windows and
replacement-token budget. The sentence-level human/LLM pair reuses recipient
sentence positions and inserts only complete donor sentences. Evaluation
imports the source cell's frozen detector directions, clipping specifications,
and 1/5-percent-FPR thresholds; it never tunes them on audit rows. Outputs live
under `runs/splice_artifact_audits/<audit-id>` and
`results/splice_artifact_audits/<audit-id>`.

This archived diagnostic is not part of the final nine-cell + RAID result.
The following commands are historical reference, not a next action or a
requirement for report writing. They require an independently preserved local
archive; a fresh clone does not contain the ignored scripts.
The following expected counts describe the archived protocol, not evidence
that its Delta gate or full study passed. Verify historical sources and script
compatibility separately before executing them.
For a deliberately resumed diagnostic, run the all-detector gate first. These commands are for Delta after the user's
Git pull and after verifying that both `runs` and `results` resolve below
`/work/hdd`:

```bash
cd ~/LLM-detection
source tools/delta/activate_environment.sh
mkdir -p logs

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
AUDIT_ID="granite-splice-smoke-$STAMP"
JOB_ID=$(sbatch --parsable \
  --account=bhuc-delta-gpu \
  --partition=gpuA100x4 \
  --gpus-per-node=2 \
  --cpus-per-task=16 \
  --mem=160G \
  --time=02:00:00 \
  --job-name=granite-splice-smoke \
  --export=ALL,AUDIT_ID="$AUDIT_ID",SOURCE_RUN_ID=granite8b-xsum-full-20260730T063720Z,SAMPLE_COUNT=12,BOOTSTRAP_REPETITIONS=100,DEBUG_ONLY=1 \
  archive/studies/splice_audit/delta_splice_artifact_audit.sbatch)
echo "submitted_job_id=$JOB_ID audit_id=$AUDIT_ID"
```

This gate deliberately includes Binoculars. It should produce 12 alternative
rows and `12 x 13 = 156` data/target/Binoculars rows. After it passes, launch
the frozen 500-source audit with a new ID:

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
AUDIT_ID="granite-splice-audit-$STAMP"
JOB_ID=$(sbatch --parsable \
  --account=bhuc-delta-gpu \
  --partition=gpuA100x4 \
  --gpus-per-node=2 \
  --cpus-per-task=16 \
  --mem=160G \
  --time=06:00:00 \
  --job-name=granite-splice-audit \
  --export=ALL,AUDIT_ID="$AUDIT_ID",SOURCE_RUN_ID=granite8b-xsum-full-20260730T063720Z \
  archive/studies/splice_audit/delta_splice_artifact_audit.sbatch)
echo "submitted_job_id=$JOB_ID audit_id=$AUDIT_ID"
```

Validate Slurm and scientific completion independently:

```bash
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
tail -n 100 "logs/granite-splice-audit-$JOB_ID.err"
wc -l \
  "runs/splice_artifact_audits/$AUDIT_ID/alternative_generations.jsonl" \
  "runs/splice_artifact_audits/$AUDIT_ID/data.jsonl" \
  "runs/splice_artifact_audits/$AUDIT_ID/target_scores.jsonl" \
  "runs/splice_artifact_audits/$AUDIT_ID/binoculars_scores.jsonl"
python - "$AUDIT_ID" <<'PY'
import json, sys
from pathlib import Path
audit_id = sys.argv[1]
run = Path("runs/splice_artifact_audits") / audit_id
result = Path("results/splice_artifact_audits") / audit_id
manifest = json.loads((run / "manifest.json").read_text())
summary = json.loads((result / "summary.json").read_text())
print("status:", manifest["completion_status"])
print("stages:", manifest["completed_stages"])
print("summary:", summary)
PY
```

The expected full counts are 500/6,500/6,500/6,500. The result directory also
contains `metrics.csv`, `artifact_share.csv`, `boundary_diagnostics.csv`, and
three compact PNGs. Alternative generation uses the same checkpoint semantics
as the main Transformers backend: completed rows survive interruption, but a
mid-generation resume does not promise RNG equivalence to an uninterrupted
run. Use a new audit ID if the frozen protocol changes.

## Beemo expert-edit benchmark

[`archive/studies/beemo/run_beemo.py`](../archive/studies/beemo/run_beemo.py) is the authoritative entry point for
the separate archived realistic-edit study, not an active final-report
dependency. The description below gives its designed contract; inspect its own
historical manifests before claiming those counts or completion. It orchestrates five stages:

1. download and normalize all nine Beemo variants per record;
2. score output text with GPT2-XL, OPT-1.3B, Falcon-7B, Qwen2-7B, and, by
   default, the shared Binoculars pair; optional Granite is an explicit flag;
3. tune, calibrate, and evaluate under record-group isolation;
4. create one compact two-by-seven plot per primary scorer;
5. validate row keys, counts, detectors, and markers.

Large artifacts live below `runs/beemo/<run-id>` and small results below
`results/beemo/<run-id>`. A full run contains 2,187 × 9 = 19,683 prepared rows
and the same number of rows in each of four target-scorer packs and the shared
Binoculars pack. Six single-model detectors × four scorers plus one shared
Binoculars configuration produce 25 scorer-detector configurations. Across ten
reported conditions, two aggregations, and two FPRs, this produces 1,000 metric
rows. Optional Granite adds six configurations and 240 rows.

The Delta gate and full commands are maintained in
[`archive/studies/beemo/DELTA_GUIDE.md`](../archive/studies/beemo/DELTA_GUIDE.md). The two-GPU wrapper is
[`archive/studies/beemo/delta_beemo.sbatch`](../archive/studies/beemo/delta_beemo.sbatch); it enforces CUDA 12.8
and work-filesystem links before execution. Convenience single-stage scripts
exist for preparation, scoring, evaluation, plotting, and validation. A new
run ID is mandatory when the Beemo split, context policy, scorer, variants,
clipping objective, calibration, or bootstrap protocol changes.

## Manifests, completion markers, and resume behavior

For non-selection stages, `run_experiment.py` creates
`runs/<run-id>/manifest.json` with configuration groups, revisions, Git/software/
accelerator provenance, paths, and host/Slurm metadata. Each completed stage is
appended to `completed_stages`; exceptions set `completion_status: failed`.
`--stage all` ends at `complete`, whereas a standalone stage ends at
`stage-<name>-complete`.

[`AppendSafeJsonlWriter`](../experiment_core/infrastructure/io.py) appends complete sorted-key JSON
records and fsyncs at checkpoints. On resume,
[`repair_partial_jsonl`](../experiment_core/infrastructure/io.py) truncates only an invalid,
non-newline-terminated final record. A malformed complete line is a hard error.
[`completed_keys`](../experiment_core/infrastructure/io.py) rejects duplicates and lets each stage
skip complete provenance keys. CSV and JSON markers use temporary-file atomic
replacement.

Resume is key-safe, not a universal configuration cache. Base/tail/data keys do
not encode all generation or algorithm settings. General manifests record the
generation, contamination, scoring and evaluation groups, among other
provenance, but the base runner does not compare the complete resolved
configuration on resume. Dedicated revision identities provide stricter
configuration/content checks. Also, the Transformers backend keeps per-seed RNG state
only in memory; after an interrupted process, completed generations remain
unchanged but the remaining examples restart that seed's RNG stream and need
not match an uninterrupted reference run. The dedicated Delta smoke compares a
completed first pass with a no-op second pass; it does not test mid-generation
RNG equivalence. Use a new run ID for changed settings and preserve the original
manifest/config beside released artifacts.

## Plotting

The canonical final primary + RAID exporter is
[`tools/exports/export_final_publication_bundle.py`](../tools/exports/export_final_publication_bundle.py),
with Delta wrapper
[`tools/delta/delta_export_final_publication.sbatch`](../tools/delta/delta_export_final_publication.sbatch).
It validates corrected source results, creates 99 primary plots and three
paper-facing RAID plots, and exports metrics/provenance without large score packs.
The older exporter documented below is retained for historical reproduction.


[`plot_tpr_contamination.py`](../plot_tpr_contamination.py) is the canonical plot
entry point. It reads tidy CSV rows from `primary_frozen_mixture`, creates one
row of panels per detector and columns for 1%/5% FPR, distinguishes attack mode
by color and aggregation by line style, and draws bootstrap bands unless
`--no-ci` is passed.

```bash
python plot_tpr_contamination.py \
  --metrics results/paper-xsum-qwen32/metrics.csv \
  --output results/paper-xsum-qwen32/tpr_contamination.png
```

Use `--detectors log_likelihood,lrr,binocular_origin` to select detectors or
`--analysis` for a deliberately labeled non-primary analysis.

[`archive/exporters/export_completed_plots.py`](../archive/exporters/export_completed_plots.py)
discovers the newest completed, non-debug run for every available paper cell,
creates an overview and one compact plot per detector, copies small metrics and
manifest artifacts, lists missing/skipped cells, and produces a ZIP without
reading the large JSONL score packs. Each cell also receives a 2-by-7 overview
and separate seven-detector horizontal strips for 1% and 5% FPR:

```bash
python archive/exporters/export_completed_plots.py
```

By default the export and archive are written below
`results/plot_exports/`. It resolves under `/work/hdd` only if that checkout's
`results` link is correctly configured; verify it rather than assuming it.

## Delta setup, storage, and Slurm

The active Delta instructions are [`docs/delta/DELTA.md`](delta/DELTA.md).
Use [REPOSITORY_SMOKE.md](delta/REPOSITORY_SMOKE.md) for no-download layout/environment
verification, [DELTA_SMOKE.md](delta/DELTA_SMOKE.md) for the bounded real Qwen-0.5B
pipeline, and the detector-revision guide for the separate real Falcon gate.
None needs repeating merely to read or plot completed validated results.

Recommended filesystem responsibilities are:

| Delta path | Responsibility |
|---|---|
| `/u/<user>` | small checkout/launch scripts and lightweight logs; not full token-feature outputs |
| `/projects/<project>` | virtual environments and deliberately retained/shared caches |
| `/work/hdd/<project>/<user>` | high-volume run JSONL, feature packs, results, and temporary experiment workspace |

In `run_experiment.py`, both relative output paths and the relative `--config`
filename resolve below `--workspace`; use an absolute config filename when the
workspace differs from the checkout. Revision runners instead reconstruct their
settings from the preserved source manifest. The general
`delta_experiment.sbatch` defaults to `$PWD/.venv` and `$PWD/.cache/huggingface`
unless `VENV_PATH`/`HF_HOME` are exported. It does not enforce storage placement.
For this installation set those variables to the existing project environment
and shared work cache, and verify `runs`/`results` links into `/work/hdd` before
launch. The revision launcher already uses the shared cache and environment,
offline mode, and explicitly checks the links. Do not generalize the older
launcher's defaults to all current wrappers.

**New installation only:** cluster environment setup creates the requested venv, loads
`miniforge3-python`, installs PyTorch 2.11.0 from the CUDA 12.8 wheel index, then
installs [`requirements.txt`](../requirements.txt):

The existing installation must instead be activated with
`source tools/delta/activate_environment.sh`; its display name is
`llm-detection`, while its physical path remains `.../venvs/delta-smoke`.
Do not reinstall packages as a routine result-inspection step.

```bash
VENV_PATH="/projects/<project>/$USER/venvs/llm-detection" \
  bash tools/delta/setup_delta_env.sh
```

[`tools/delta/delta_experiment.sbatch`](../tools/delta/delta_experiment.sbatch) defaults to
`gpuA100x4`, four GPUs, 32 CPUs, 240 GB host memory, and 24 hours. It uses
Transformers unless `GENERATION_BACKEND=vllm`, passes `device_map="auto"` for
target generation/scoring, and can skip Binoculars. The wrapper
[`tools/delta/submit_delta_matrix.sh`](../tools/delta/submit_delta_matrix.sh) accepts
`primary`, `replication`, `qwen-scaling`, or `all`, requires an explicit account,
and permits only `gpuA100x4` or explicit `gpuH200x8`. Qwen 72B is skipped unless
H200 is selected. `all` deduplicates Qwen 32B; separate primary/scaling wrapper
calls do not share run IDs and would duplicate it.

The dedicated smoke wrapper and sbatch request one A100, 8 CPUs, 32 GB, and 45
minutes, skip Binoculars, run the pipeline twice, validate exact keys and
metrics, and write `smoke_validation.complete.json` only after the resume
snapshot agrees. These commands submit jobs and must be run only by an
authorized operator:

```bash
ACCOUNT=<delta-account> VENV_PATH=/projects/<project>/$USER/venvs/delta-smoke \
  SMOKE_CACHE_ROOT=/work/hdd/<project>/$USER/cache \
  bash tools/delta/submit_delta_smoke.sh
```

No local test establishes CUDA correctness, model placement, vLLM behavior,
Slurm submission, or paper-scale throughput.

## Extending the repository

Use a new run ID and add focused tests for every extension.

- **New dataset:** add its ID/split/text and ID fields to the JSON, extend the
  hardcoded CLI dataset choices, and test normalization, deduplication, packing,
  manifest determinism, and source sufficiency. A new dataset requires new
  manifests, preparation, scoring, and evaluation.
- **New target model:** add it to the appropriate config and matrix-wrapper
  list, decide revision/tokenizer/device/dtype settings, and validate generation
  length, feature schema, and memory on a bounded gate.
- **New detector:** implement its token/document formula in `scoring.py`, add
  evaluation local contributions and clipping semantics, update detector lists,
  plotting labels, smoke validation as appropriate, and numerical-reference,
  precomputation, bootstrap, and end-to-end tests.
- **New contamination method:** implement deterministic construction and
  provenance in `data.py`/`pipeline.py`; extend expected-row math, modes in
  evaluation and plotting, tuning-mixture validation, row keys, and clustered
  tests. Do not silently reuse old rows with matching keys.
- **New external benchmark:** predeclare whether it is a frozen-transfer test
  or a benchmark with its own tuning/calibration/test split. A frozen-transfer
  test must not retune on external rows. RAID is the latter: it has its own
  source-disjoint tuning and domain-wise human calibration, not thresholds
  transferred from the nine primary cells. Neither design permits test-set
  tuning.

## Tests by subsystem

Run the complete no-download suite with:

```bash
python -m unittest discover -s tests -v
```

| Test module | Coverage |
|---|---|
| [`tests/test_config_and_data.py`](../tests/test_config_and_data.py) | frozen paper values, split/seed assignment, ratio-zero count, exact token budgets, tail-cache reuse, SQuAD packing |
| [`tests/test_scoring.py`](../tests/test_scoring.py) | exact NumPy/Torch features, top-k/margins, single-forward document scores, schema resume guards, optional pooling, Binoculars roles/formula |
| [`tests/test_evaluation.py`](../tests/test_evaluation.py) | compact rows, primitives, all-detector raw/clipped precompute equivalence, leakage-safe ordering, indexed-vs-serial bootstrap equivalence, deterministic output/provenance |
| [`tests/test_io.py`](../tests/test_io.py) | torn-final-line repair, duplicate rejection, durable append checkpoints, completion markers |
| [`tests/test_delta_and_manifest.py`](../tests/test_delta_and_manifest.py) | Delta partitions/matrix and required manifest provenance |
| [`tests/test_delta_smoke.py`](../tests/test_delta_smoke.py) | exact bounded config, static Slurm guards, no-network two-pass synthetic smoke and validator |
| `tests/test_raid_data.py`, `tests/test_raid_scoring.py`, `tests/test_raid_evaluation.py`, `tests/test_raid_pipeline.py`, `tests/test_raid_delta.py` | RAID selection/indexing, feature references, fitted procedures, source-cluster evaluation and sharded orchestration |
| `tests/test_detector_revision.py` | revised detector arithmetic, gate/score/evaluate recovery, constant and zero-cap safeguards, conditional primary and official RAID distinctions |
| `tests/test_final_publication_export.py` | source artifact validation, filtered seven-detector export, plots and bundle integrity |

Torch-specific unit paths skip when Torch is unavailable. The synthetic smoke
validates contracts and deterministic resume keys, not actual dataset/model
downloads or GPU behavior.

## Current limitations and planned optimizations

- The final nine-cell + RAID bundle provides artifact-level evidence for those
  completed runs. The user-reported repository smoke `21883773` passed all 105
  active tests without skips on Delta. Neither establishes untested vLLM,
  72B/H200, arbitrary future models, or equivalence under changed dependencies.
- Configured revisions are `null` until runtime resolution; pin released runs
  explicitly for immutable replication.
- General resume records configuration groups but does not compare the full
  configuration before reusing prepared artifacts, and
  stochastic RNG state is not persisted across interrupted generation.
- The evaluator holds compact token-feature rows in memory. Scalar
  precomputation removes repeated aggregation inside bootstrap but does not make
  initial token-array loading streaming.
- Bootstrap repetitions and detector analyses are intentionally serial; planned
  optimization may vectorize bootstrap metrics or parallelize detectors only
  after numerical equivalence is preserved.
- Precompute timing is logged; this audit did not establish a controlled
  before/after speedup benchmark. Do not turn elapsed-job anecdotes into one.
- The matrix wrapper enforces Qwen-32B reuse only within one invocation; it has
  no cross-run registry.
- `/work/hdd` placement is not enforced by the general config validator or
  primary full-generation wrapper; corrected revision and RAID wrappers have
  explicit checks. Validate actual links before each large launch.
- The current RAID evaluator is internally split, not a frozen-transfer
  evaluator. Do not describe its results as zero-tuning external transfer.
- Historical source is retained under `archive/`; generated Python caches are ignored.
