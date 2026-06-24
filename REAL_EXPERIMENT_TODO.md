# Real Experiment TODO

Goal: evaluate raw vs clipped sum-based LLM detectors on real text, using XSum/SQuAD/WritingPrompts-style human continuations and generated LLM continuations.

## 1. Generate Real Contamination Data

Start with XSum. Use more than the smoke-test size.

```powershell
python prepare_real_contamination.py --dataset xsum --n-samples 200
```

This writes:

```text
real_data/xsum/Qwen__Qwen2.5-0.5B/clean.jsonl
real_data/xsum/Qwen__Qwen2.5-0.5B/contaminated.jsonl
```

`clean.jsonl` contains human clean and LLM clean rows.  
`contaminated.jsonl` contains LLM rows with random and tail human contamination.

## 2. Score Clean Data

```powershell
python score_real_text.py `
  --input real_data/xsum/Qwen__Qwen2.5-0.5B/clean.jsonl `
  --output real_data/xsum/Qwen__Qwen2.5-0.5B/clean_scores.jsonl `
  --observer-model Qwen/Qwen2.5-0.5B `
  --reference-model Qwen/Qwen2.5-1.5B
```

## 3. Score Contaminated Data

```powershell
python score_real_text.py `
  --input real_data/xsum/Qwen__Qwen2.5-0.5B/contaminated.jsonl `
  --output real_data/xsum/Qwen__Qwen2.5-0.5B/contaminated_scores.jsonl `
  --observer-model Qwen/Qwen2.5-0.5B `
  --reference-model Qwen/Qwen2.5-1.5B
```

The scoring script computes token-level features:

```text
logp_a
rank_a
logrank_a
entropy_a
cross_entropy_ba
```

and document scores:

```text
log_likelihood
rank
log_rank
entropy
entropy_gap
binoculars
```

## 4. Evaluate Raw vs Clipped Methods

```powershell
python evaluate_real_clipping.py `
  --clean real_data/xsum/Qwen__Qwen2.5-0.5B/clean_scores.jsonl `
  --contaminated real_data/xsum/Qwen__Qwen2.5-0.5B/contaminated_scores.jsonl `
  --output-dir real_results/xsum_qwen
```

This does a 50/50 split by `sample_id`:

```text
50% tune clipping bounds
50% test final TPR
```

FPR thresholds are set using clean human test scores.

Outputs:

```text
real_results/xsum_qwen/metrics.csv
real_results/xsum_qwen/random_contamination_tpr_at_1_fpr.png
real_results/xsum_qwen/random_contamination_tpr_at_5_fpr.png
real_results/xsum_qwen/tail_contamination_tpr_at_1_fpr.png
real_results/xsum_qwen/tail_contamination_tpr_at_5_fpr.png
```

## 5. Important Sample Size Note

The 5-example smoke test is only for checking the pipeline.

For meaningful curves:

```text
minimum: 100-200 samples
better: 500+ samples
```

At very low FPR, small test sets are unstable:

```text
TPR@5%FPR needs many human examples
TPR@1%FPR needs even more
```

If the plot only takes values like `0`, `0.33`, `0.67`, `1.0`, the test set is too small.

## 6. Next Datasets

After XSum works:

```powershell
python prepare_real_contamination.py --dataset squad --n-samples 200
python prepare_real_contamination.py --dataset writingprompts --n-samples 200
```

Then repeat scoring and evaluation with the corresponding paths.

## 7. Methods To Report

Primary:

```text
log_likelihood
log_rank
binoculars
entropy_gap
```

Secondary:

```text
rank
entropy
```

Compare:

```text
raw aggregation
clipped aggregation
```

Main expected story:

```text
random contamination: clipping may be neutral
tail contamination: clipping should improve robustness
```
