# Sequence Model Benchmark Result Analysis

## Analysis Scope

This document records the result interpretation for benchmark v2, whose goal is to align `health` with `remaining-life progress` instead of `DP` proxy severity.

The relevant runs are:

- formal v2 benchmark run:
  - `outputs/sequence_model_benchmark_v2/20260313_135556`
- implementation smoke run:
  - `outputs/sequence_model_benchmark_v2_smoke/20260313_135428`

The smoke run is only a pipeline check:

- `A2` only
- `2-fold`
- `1` pretrain epoch
- `1` fine-tune epoch
- limited sequences and windows

So it is useful for verifying semantics and output structure, but it is not suitable for model ranking.

## Why The Previous Version Was Misaligned

The previous benchmark used:

- `health = clip((600 - DP) / (600 - 25), 0, 1)`

That target is easy to regress because it is just a rescaled `Differential_pressure`.

But that target is not equivalent to:

- remaining life
- life progress
- distance to failure in normalized time

So the old pipeline had a structural mismatch:

1. sequence models learned `DP` proxy health very well
2. `RUL` was then reconstructed from local health slope
3. small local slope noise exploded into large `RUL` spikes

This is why the previous `RUL` curves kept bouncing even when `health` scatter looked almost perfect.

## What v2 Changes

Benchmark v2 changes the target and the training protocol:

- supervised target:
  - `health_remaining_true = RUL / total_life`
- per-sequence constant:
  - `total_life = Time + RUL`
- `Train` split:
  - self-supervised pretraining only
- `Test` split:
  - grouped supervised `5-fold` evaluation only
- benchmark `RUL`:
  - `rul_pred = health_remaining_pred * total_life`

This means benchmark v2 is now testing:

- whether the model can learn normalized life progress directly
- whether `RUL` becomes stable once it is derived from a target that already means remaining life

It is not testing simulator-side inference yet, because true `total_life` is only available inside benchmark evaluation.

## What Has Already Been Verified

### 1. Target semantics are now aligned with life progress

The new code explicitly separates:

- `health_remaining_true`
- `health_remaining_pred`
- `dp_proxy_health`

This removes the previous label ambiguity. `DP` proxy is now only a diagnostic feature, not the main supervised target.

### 2. `Test` labels now behave like remaining life

The v2 tests verify that for each `Test` sequence:

- `total_life = Time + RUL` is constant
- `health_remaining_true = RUL / total_life` stays in `[0, 1]`
- `health_remaining_true` is monotone non-increasing over time

This is the key data-level alignment that the previous version did not have.

### 3. `Train` no longer leaks fake life labels

The v2 pipeline now uses:

- `Train` for next-step `DP + Flow` self-supervised pretraining
- `Test` for life-progress supervision

This is important because `Train` does not have true `RUL`, so it should not be assigned pseudo life labels.

### 4. Grouped evaluation is now consistent with the intended supervision boundary

The v2 tests also verify:

- grouped folds are created from `Test` sequences only
- held-out `Test` sequences never appear in pretraining
- `val_fold` and `test_fold` are excluded from pretraining

So the new benchmark is much cleaner from a leakage perspective.

### 5. `RUL` no longer comes from slope inversion

The old EWMA slope-based `RUL` conversion is fully removed.

The new benchmark prediction rule is:

```text
rul_pred = health_remaining_pred * total_life
```

This means the old source of bouncing has been removed at the benchmark-definition level.

## Preliminary Run Interpretation

At the time this analysis was updated:

- the formal run at `outputs/sequence_model_benchmark_v2/20260313_135556` is producing fold artifacts
- the smoke run has completed
- the smoke run confirms that all four backends can finish:
  - pretraining
  - fine-tuning
  - held-out prediction
  - metrics export
  - reference-tail plotting

From the smoke run, the absolute metrics are poor, which is expected because the run is intentionally tiny and undertrained.

The smoke run is therefore only evidence of:

- correct semantics
- correct data flow
- correct output structure

It is not evidence for final model quality.

## Direct Answers To The Two Core Questions

### 1. Does `health` now really correspond to life progress?

Yes, at the benchmark-definition level it does.

Why:

- `health_remaining_true` is derived directly from true `RUL`
- it is normalized by per-sequence `total_life`
- it is monotone toward `0` at failure
- it is no longer a pressure proxy

So this version is aligned with life progress in a way the previous version was not.

What is still not solved:

- simulator-side inference does not know true `total_life`
- so benchmark v2 proves target alignment, not deployment readiness

### 2. Are the `RUL` curves still bouncing?

The previous source of bouncing has been removed from the benchmark definition.

Why:

- v1 reconstructed `RUL` from local slope estimates
- v2 computes `RUL` directly from predicted remaining-life fraction

So the old "local slope noise turns into huge `RUL` spikes" mechanism no longer exists here.

What still needs formal confirmation:

- the completed full v2 run should be checked to confirm that the new scatter and representative curves are visually smooth under normal training budget

That confirmation depends on the final outputs from:

- `metrics_overall.csv`
- `metrics_by_family.csv`
- `plots/scatter_health_pred.png`
- `plots/scatter_rul_pred.png`
- representative `health_curve_*` and `rul_curve_*` plots

## What To Check When The Full Run Finishes

The formal v2 run should be accepted only if all of the following hold:

### Health target quality

- `health_remaining` remains learnable across `A2 / A3 / A4`
- overall and family-level `health_rmse` are meaningfully better than naive baselines
- ranking across `GRU / LSTM / TCN / ATTENTION` is stable enough to choose a default encoder

### RUL stability

- `rul_pred` scatter no longer shows the large vertical explosion from v1
- representative `RUL` curves are smoother and mostly monotone
- no frequent upward spikes appear from one neighboring point to the next

### A2 distribution-shift behavior

- `A2` still needs special attention because train/test feed coverage is asymmetric
- any severe degradation of `A2` relative to `A3 / A4` should be interpreted partly as distribution shift, not only architecture weakness

## Practical Conclusion

This batch fixes the main semantic mismatch in the benchmark.

What can now be claimed:

- the benchmark target is aligned with remaining-life progress
- the data protocol is cleaner
- `Train` and `Test` roles are now consistent with the available labels
- the previous slope-based `RUL` instability mechanism has been removed

What cannot yet be claimed until the full run completes:

- which model is the final winner under v2
- how much `RUL` stability improved numerically
- whether benchmark v2 is ready to be connected back to the simulator

## Next Step

Once the formal run finishes, this file should be updated with:

- overall metrics from `metrics_overall.csv`
- family-level winners from `metrics_by_family.csv`
- explicit judgement on `RUL` smoothness from the representative plots
- a decision on which single sequence model should move forward into simulator integration
