# Sequence Model Benchmark Result Analysis

## Analysis Scope

This analysis uses the completed formal benchmark run at:

- `outputs/sequence_model_benchmark/20260313_141821`

Although the directory name still uses `sequence_model_benchmark`, the run itself is already the v2 benchmark. This is confirmed by:

- `health_target_type = remaining_life_fraction`
- `pretrain_task = next_step_dp_flow`
- `rul_derivation = health_hat_times_total_life`

from:

- `outputs/sequence_model_benchmark/20260313_141821/benchmark_config.json`
- `outputs/sequence_model_benchmark/20260313_141821/dataset_summary.json`

The partially completed run at:

- `outputs/sequence_model_benchmark_v2/20260313_135556`

is explicitly excluded from this analysis because it did not finish all families/models and did not emit final aggregate metrics.

This document combines four evidence sources:

1. benchmark outputs under `outputs/sequence_model_benchmark/20260313_141821`
2. `dataset_summary.json` from the same run
3. raw `Train_Data_CSV.csv` and `Test_Data_CSV.csv`
4. the existing dataset study in [dataset_analysis.md](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/dataset_analysis.md)

## Overall Result

The short answer is:

- yes, the result matches the main expectation of benchmark v2
- but no, it does not yet fully solve the problem

Why it matches expectation:

- `health_remaining` is no longer behaving like an almost-trivial `DP` rescaling target
- `RUL` no longer shows the large slope-inversion explosions from the previous version
- the model ranking is stable and interpretable

Why it does not fully match expectation:

- `A2` still has a clear subgroup mismatch
- local trajectory jaggedness is still visible
- benchmark-side `RUL` still relies on true `total_life`

Overall model ranking from `metrics_overall.csv` is:

| Rank | Model | Health RMSE | Health R2 | RUL RMSE |
| --- | --- | ---: | ---: | ---: |
| 1 | `LSTM` | `0.0638` | `0.8951` | `8.7127` |
| 2 | `GRU` | `0.0682` | `0.8792` | `9.2312` |
| 3 | `ATTENTION` | `0.0689` | `0.8650` | `9.5683` |
| 4 | `TCN` | `0.0713` | `0.8564` | `10.0200` |

Main conclusion:

- `LSTM` is the best current default under the new remaining-life target
- the ranking is coherent on both `health` and `RUL`
- this is a good sign that the new target semantics are doing useful work

## Dataset-Coupled Interpretation

The benchmark result only makes sense when read together with the dataset structure.

### 1. Family material differences still show up in model difficulty

Per [dataset_analysis.md](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/dataset_analysis.md), the particle-size summaries are:

| Family | `D50` (`um`) | Mean total life |
| --- | ---: | ---: |
| `A2` | `8.250` | `65.67` |
| `A3` | `14.214` | `119.49` |
| `A4` | `34.620` | `166.87` |

This dataset-level pattern is important:

- `A2` has the finest particles and the shortest average life
- `A4` has the coarsest particles and the longest average life

The benchmark results follow the same direction:

- `A4` is the easiest family
- `A3` is intermediate
- `A2` is the hardest

So the benchmark is not producing an arbitrary ranking. It is consistent with the material and lifespan patterns already documented in the dataset analysis.

### 2. A2 is not just a generic "hard family"; it is distribution-shifted

From `dataset_summary.json`:

- `A2` train feeds: `59.107235, 79.246269, 118.214470, 158.492538`
- `A2` test feeds: `59.107235, 177.321701, 236.428940, 237.738800, 316.985077`
- overlap: only `59.107235`

This means `A2` is the only family with severe train/test feed asymmetry.

However, the benchmark failure pattern shows that the problem is not only "unseen high feed".

In `A2` fold 3, the worst held-out sequences are:

- `test_19`: `feed = 59.1072`, `total_life = 202.4`
- `test_20`: `feed = 59.1072`, `total_life = 189.2`

These are low-feed sequences, not unseen high-feed sequences.

So the true issue is:

- `A2` contains a rare low-feed, very-long-life subgroup
- when that subgroup is concentrated in the held-out fold, all models underpredict it

This is a much more precise diagnosis than simply saying "A2 has unseen feeds".

### 3. Many test sequences still end well before failure

From `dataset_summary.json`, the terminal observed `health_remaining` ranges are:

| Family | terminal `health_remaining` min | terminal `health_remaining` max | observed life fraction range |
| --- | ---: | ---: | ---: |
| `A2` | `0.0404` | `0.6200` | `38.00% - 95.96%` |
| `A3` | `0.0201` | `0.6402` | `35.98% - 97.99%` |
| `A4` | `0.1199` | `0.5299` | `47.01% - 88.01%` |

This matters for interpretation:

- many sequences do not actually run to observed failure
- the "tail to zero" in the representative plots is a true-label reference extension
- it is not an observed continuation of the raw signal

So any visual comparison beyond the dotted "observed end" line must be read as:

- true reference tail only
- not a learned predicted tail

## Family-Level Findings

### A2

`A2` is the weakest family, and the weakness is structurally meaningful.

For `LSTM`, family-level metrics are:

- `health_rmse = 0.0874`
- `health_r2 = 0.7945`
- `rul_rmse = 9.0349`

This is still acceptable, but it is clearly worse than `A3` and `A4`.

The biggest issue is fold volatility. In `metrics_by_fold.csv`, the worst `A2` fold is fold `3`:

- `LSTM health_rmse = 0.1855`
- `LSTM rul_rmse = 35.6426`

The same fold is also the worst one for `GRU`, `ATTENTION`, and `TCN`, which means:

- this is not a single-model failure
- this is a dataset subgroup failure

The held-out `A2` fold-3 sequences are:

- `test_19` at `59.1072`
- `test_20` at `59.1072`
- `test_41` at `237.7388`

Under `LSTM`, the large underestimation is concentrated in the first two long-life low-feed sequences, not in the high-feed one.

Interpretation:

- `A2` difficulty is driven by subgroup structure, not just feed extrapolation
- the current representation still compresses long-life low-feed `A2` behavior too aggressively

### A3

`A3` is intermediate.

For `LSTM`, family-level metrics are:

- `health_rmse = 0.0703`
- `health_r2 = 0.9170`
- `rul_rmse = 11.0410`

Important dataset fact:

- `A3` has full `8`-feed coverage in both train and test

So `A3` shows an important contrast with `A2`:

- feed coverage is complete
- but it is still clearly harder than `A4`

Interpretation:

- the remaining error is not only a feed-coverage problem
- `A3` likely has higher intra-family degradation heterogeneity
- the model can track life progress, but it still overestimates some high-health segments and keeps visible local jaggedness

### A4

`A4` is the strongest family.

For `LSTM`, family-level metrics are:

- `health_rmse = 0.0336`
- `health_r2 = 0.9739`
- `rul_rmse = 6.0623`

This matches the dataset facts documented in [dataset_analysis.md](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/dataset_analysis.md):

- `A4` is the coarsest dust family
- `A4` has the largest median and mean total life
- `A4` reaches high pressure thresholds later than `A2` and `A3`

Interpretation:

- `A4` is the cleanest demonstration that benchmark v2 is aligned with remaining-life semantics
- the predicted curves are smooth, close to the true trajectory, and only lightly biased

## Figure-Based Interpretation

### Health scatter

See:

- [scatter_health_remaining_pred.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_141821/plots/scatter_health_remaining_pred.png)

What the plot shows:

- points cluster around the diagonal for all four models
- the cloud is no longer unrealistically perfect
- family-specific band structure is still visible

Interpretation:

- the model is learning remaining-life progress, not merely a trivial `DP` transform
- this is exactly what benchmark v2 was supposed to achieve

### RUL scatter

See:

- [scatter_rul_pred.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_141821/plots/scatter_rul_pred.png)

What the plot shows:

- the previous large vertical blow-up is gone
- points broadly follow the diagonal
- some banded bias remains in the medium-to-high RUL region

Interpretation:

- `RUL` stability has improved substantially
- the improvement comes from target-semantic alignment, not from simply using a stronger model

### A2 LSTM RUL curve

See:

- [rul_curve_A2_LSTM.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_141821/plots/rul_curve_A2_LSTM.png)

What the plot shows:

- the trajectory is no longer violently exploding
- but it stays systematically below the true line
- the bias is strongest for long-life low-feed `A2` behavior

Interpretation:

- v2 fixed the old bouncing failure mode
- but `A2` still suffers from structured underestimation

### A4 LSTM RUL curve

See:

- [rul_curve_A4_LSTM.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_141821/plots/rul_curve_A4_LSTM.png)

What the plot shows:

- predicted `RUL` decreases almost along the true line
- the trajectory is smooth and stable across the observed interval

Interpretation:

- this is the best evidence that the new target definition is working as intended
- `A4` is currently the strongest benchmark family for downstream integration confidence

### A3 LSTM health curve

See:

- [health_curve_A3_LSTM.png](/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1/outputs/sequence_model_benchmark/20260313_141821/plots/health_curve_A3_LSTM.png)

What the plot shows:

- overall shape is correct
- the high-health region is somewhat overestimated
- local sawtooth behavior still appears across the observed segment

Interpretation:

- the model has learned the correct direction of life progress
- but local monotonicity is still not well controlled

## Does It Match Expectation

Yes, in the main sense it does.

What matches expectation:

- `health_remaining` is no longer behaving like a disguised `DP` proxy
- `RUL` no longer inherits the previous slope-based bouncing pathology
- family difficulty roughly matches the known dataset physics: `A4` easiest, `A3` medium, `A2` hardest
- `LSTM` emerges as a defensible default winner

What does not fully match expectation:

- `A2` still has a major subgroup mismatch
- local pointwise monotonicity is still weak
- benchmark-side `RUL` is still evaluation-only because it uses true `total_life`

So the correct conclusion is:

- benchmark v2 is directionally successful
- but it is not yet ready to be interpreted as a finished simulator-side solution

## Open Problems

1. `A2` fold 3 fails because of a rare low-feed, long-life subgroup, not because of unseen high-feed extrapolation alone.

2. The curves no longer have large bouncing, but local upward steps remain common enough that monotonicity is still visually imperfect.

3. Benchmark `RUL` still depends on true `total_life`, so the current numbers cannot yet be interpreted as simulator-ready inference quality.

4. Many test sequences do not reach observed end-of-life, so the true tail shown in the plots is a reference extension, not an observed continuation.

## Next Analytical Step

The next analysis should verify four things before any simulator reintegration is trusted:

1. Error by `A2` subgroup:
   - split `A2` by feed and life-length regime
   - quantify whether low-feed long-life sequences are the dominant failure source across all folds

2. Error by observation horizon:
   - compare sequences that end near failure against sequences that stop early
   - test whether early-cut sequences systematically increase remaining-life bias

3. Monotonicity diagnostics:
   - quantify local upward-step frequency and magnitude for each family/model
   - separate harmless micro-jitter from meaningful trajectory reversals

4. `total_life` dependence:
   - measure how much of the current `RUL` quality comes from the accurate `health_remaining` target
   - versus how much is simply inherited from knowing the true total lifetime during benchmark evaluation
