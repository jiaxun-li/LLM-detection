# Requirement checklist

Status values:

- **Completed / locally validated**: implementation and no-download automated
  tests pass in this workspace.
- **Completed / static validation**: code/configuration is present and parsed or
  inspected locally, but the GPU/Slurm behavior needs Delta.
- **Awaiting Delta validation**: the only missing verification requires model
  weights, CUDA GPUs, vLLM, Hugging Face datasets, or the Delta scheduler.

No requested requirement is intentionally omitted or blocked.

## Scientific design

| Requirement | Implementation | Test/status |
|---|---|---|
| XSum, SQuAD, WritingPrompts | `configs/paper.json`, `llm_detection/pipeline.py` | Config parsed; dataset loading awaiting Delta/network validation |
| Primary 8B/24B/32B targets | Granite 3.3 8B Base, Mistral Small 24B Base, and Qwen 2.5 32B in `configs/paper.json` and `scripts/submit_delta_matrix.sh` | Tests cover the matrix; Llama 3.1 8B was provider-blocked by geographic access review and is explicitly replaced by public Apache-2.0 Granite 3.3 8B Base; weights awaiting Delta |
| Erebus replication | Same files | Static validated; awaiting Delta |
| Qwen 7B/14B/32B/72B scaling | Same files | Static validated; 72B explicitly gated to H200; awaiting Delta |
| Seven detectors including LRR | `llm_detection/scoring.py`, `llm_detection/evaluation.py` | `tests/test_scoring.py`, `tests/test_evaluation.py` |
| Target model scores all single-model detectors | `TargetModelScorer` | One-forward code path compiled; exact formulas locally tested; CUDA awaiting Delta |
| Falcon Binoculars roles and formula | `BinocularsScorer`, `binoculars_score` | Formula/roles tested in `tests/test_scoring.py`; two-GPU execution awaiting Delta |
| Raw versus one-sided clipped aggregation | `oriented_score`, `tune_clipping_spec` | Synthetic end-to-end evaluation test passes |

## Data construction

| Requirement | Implementation | Test/status |
|---|---|---|
| Model-independent IDs shared across targets | shared `source_manifest_path`, `select_source_manifest` | Determinism test passes; real dataset manifests awaiting network |
| Three deterministic, disjoint splits; 500/500/2000 defaults | `assign_splits_and_generation_seeds`, paper config | Tested |
| Three generation seeds divide rather than multiply total | same function/config | Tested |
| Explicit 30/220/0.8/0.95 settings | paper/smoke configs and manifest | Tested |
| Ratios 0/5/10/20/30/40/50 | paper config | Tested |
| Tokenizer-token target, not sentence rounding | `random_token_contamination`, `tail_token_contamination` | Exact budget test passes |
| Requested/realized/original/human/replaced counts | `contamination_counts`, data rows | Tested |
| Fixed/tightly matched continuation length | one-for-one token replacement plus round-trip tolerance | Tested at token-function level; tokenizer round-trip awaiting model |
| Ratio zero stored once | `expected_rows_per_source`, clean row construction | Tested |
| Configurable random draws, seed/draw recorded | config, `corruption_seed`, data/evaluation rows | Tested/config parsed |
| White-box highest-NLL tail spans, model revisions | `build_tail_cache`, tail data provenance | Once-only order test passes; real-model NLL awaiting Delta |
| Score tail candidates once and reuse cache | `tail_candidate_cache.jsonl`, resume key | Call-count/order reuse tested |

## Generation and scoring optimization

| Requirement | Implementation | Test/status |
|---|---|---|
| Length-bucketed batched generation/scoring | `generation.length_bucketed`, both scorers | Code compiled; CUDA throughput awaiting Delta |
| Transformers default; optional vLLM | `make_generation_backend`, config | Static validated; vLLM awaiting Delta |
| Batch/microbatch/dtype/device/map/TP/max tokens/revisions | JSON configs, loaders | Config validation passes |
| One target forward supplies six detectors | `TargetModelScorer.score_batch` | Feature set/formulas tested; forward awaiting Delta |
| Exact rank and entropy with memory-aware vocab chunks | `exact_token_features` | NumPy exact reference tested; Torch/CUDA parity awaiting Delta |
| Compact reusable top-k/margin features and optional pooled hidden state | `exact_token_features`, `TargetModelScorer`, scoring config | NumPy/Torch parity and pooled-hidden unit test; dedicated GPU smoke awaiting rerun |
| Falcon pair on separate GPUs and explicit formula | `BinocularsScorer` | Formula tested; placement awaiting Delta |
| Streaming where practical | streaming dataset selection/JSONL scoring and append outputs | Partial/restart tests pass |
| Restart, completed keys, partial line repair, completion markers | `llm_detection/io.py`, pipeline markers | `tests/test_io.py` |
| Throughput, elapsed, ETA, peak memory | `runtime.Throughput` | Static validated; GPU values awaiting Delta |

## Evaluation

| Requirement | Implementation | Test/status |
|---|---|---|
| Tune only on clipping split | `_clean(..., clipping_tuning)`, `tune_clipping_spec` | Synthetic protocol test |
| One dataset × model × detector frozen mixture spec | `evaluate` primary analysis | Same spec across modes/ratios tested |
| Optional labeled mode-specific oracle | config flag and `analysis=oracle_mode_specific` | Static validated |
| Threshold only from calibration-human split | `calibration_threshold` call site | Threshold/order-statistic and protocol tests |
| Frozen threshold on final test | `evaluate` | Synthetic protocol test |
| Actual FPR, TPR@1/5, AUROC, pAUROC 0–5%, paired differences, robustness AUC, counts | tidy metrics rows | Synthetic end-to-end test |
| Paired clustered bootstrap 95% CIs | `_metric_bootstrap`, `_robustness_bootstrap` keyed by `sample_id` | Synthetic execution test |
| LRR in new outputs | scorer/evaluator | Explicitly tested |
| Tidy provenance-rich CSV | `write_tidy_csv`, output columns | Synthetic CSV run passes |

## Configuration, provenance, and Delta

| Requirement | Implementation | Test/status |
|---|---|---|
| Clear JSON configuration | `configs/paper.json`, `configs/smoke.json`, `config.py` | Parsed/validated in tests |
| Full machine-readable manifest | `runtime.build_manifest`, `run_experiment.py` | Required groups tested |
| Delta `gpuA100x4`, optional `gpuH200x8` | Delta Slurm job/wrapper | Static test passes; submission awaiting Delta |
| Isolated one-GPU Qwen 0.5B gate | `configs/delta_smoke_qwen_0_5b.json`, `scripts/delta_smoke_qwen_0_5b.sbatch`, `scripts/submit_delta_smoke.sh`, `smoke_validation.py` | Synthetic two-pass run and static resource/account tests pass; real Slurm/CUDA execution awaiting Delta |
| Requirement-to-file/test audit | this file | Completed |

## Remaining cluster-only validation

All locally verifiable requirements pass. The following are **awaiting Delta
validation**: Hugging Face dataset revisions and access, target-model generation,
Torch exact-feature parity on CUDA, model sharding/memory for 24B–72B targets,
optional vLLM, Falcon two-GPU placement, Slurm submission/resume, and measured
throughput/peak GPU memory. Follow the gate in `DELTA.md`; do not mark paper runs
complete until those checks pass.
