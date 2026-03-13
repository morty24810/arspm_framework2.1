# Sequence Model Benchmark Result Analysis

## Run Scope

This analysis is based on the formal benchmark run at:

- `outputs/sequence_model_benchmark/20260313_084355`

The analyzed run used:

- pooled `Train_Data_CSV.csv` + `Test_Data_CSV.csv`
- family-specific training for `A2 / A3 / A4`
- models:
  - `GRU`
  - `LSTM`
  - `TCN`
  - `ATTENTION`
- grouped `5-fold` evaluation by `sequence_id`
- window length `32`
- batch size `64`
- epoch budget `100`
- patience `10`
- health label:
  - `clip((600 - Differential_pressure) / (600 - 25), 0, 1)`
- derived `RUL`:
  - computed from predicted health with `5`-step EWMA decay

## Main Findings

### 1. Health regression results are very strong

Across all four models, health prediction quality is already very high:

| Model | Health MAE | Health RMSE | Health R2 |
| --- | ---: | ---: | ---: |
| LSTM | `0.00157` | `0.00262` | `0.99968` |
| GRU | `0.00183` | `0.00304` | `0.99960` |
| ATTENTION | `0.00182` | `0.00310` | `0.99946` |
| TCN | `0.00220` | `0.00350` | `0.99952` |

Interpretation:

- all four sequence models can already fit the `normalized health` target extremely well
- the new pooled-data, family-specific sequence-training setup is technically valid
- this part of the redesign is successful

### 2. Overall best model for health is LSTM

By the benchmark's current primary target, `LSTM` is the best overall model:

- best overall `health_rmse = 0.00262`
- best overall `health_mae = 0.00157`
- best overall `health_r2 = 0.99968`

This means:

- if the next batch only needs the most accurate `health` regressor, `LSTM` is the current default winner

### 3. Family-specific winners are not the same

Best model by health RMSE in each family:

| Family | Best model | Health RMSE |
| --- | --- | ---: |
| A2 | `LSTM` | `0.00218` |
| A3 | `ATTENTION` | `0.00265` |
| A4 | `GRU` | `0.00235` |

Interpretation:

- no single architecture dominates every family
- the sequence benchmark is doing something meaningful rather than collapsing to trivial ties
- there is still value in keeping the architecture-comparison path, even though `LSTM` is currently the best overall average

### 4. Derived RUL results are still not acceptable

The derived `RUL` metrics are much worse than the health metrics:

| Model | RUL MAE | RUL RMSE | RUL sMAPE |
| --- | ---: | ---: | ---: |
| TCN | `328.32` | `490.90` | `0.995` |
| LSTM | `339.52` | `500.03` | `1.021` |
| GRU | `346.40` | `506.27` | `1.017` |
| ATTENTION | `344.84` | `507.82` | `1.017` |

Interpretation:

- health prediction is already excellent
- but the current `health -> derived RUL` conversion is still unstable
- therefore the benchmark currently validates the `health-modeling` idea, not the current `RUL` derivation method

## Family-Level Interpretation

### A2

Observed facts:

- best health model: `LSTM`
- best RUL model: `TCN`
- `A2` has asymmetric feed coverage:
  - train-only feeds: `79.246269`, `118.214470`, `158.492538`
  - test-only feeds: `177.321701`, `236.428940`, `237.738800`, `316.985077`
  - overlap: only `59.107235`

Interpretation:

- `A2` is the hardest family from a distribution-shift perspective
- even though health RMSE is still low, the feed mismatch likely hurts RUL stability
- `A2` results should be treated as evidence that pooled supervision works, but also as evidence that future feed-conditioned generalization needs special care

### A3

Observed facts:

- best health model: `ATTENTION`
- best RUL model: `TCN`
- A3 has full `8`-feed coverage in both train and test
- yet A3 gives the worst RUL RMSE among the three families

Interpretation:

- A3 is not suffering from the same feed-coverage problem as A2
- the weakness is more likely in the current RUL derivation itself, not in the sequence encoder
- A3 likely contains more heterogeneous degradation-rate behavior, so the current EWMA slope mapping is too fragile

### A4

Observed facts:

- best health model: `GRU`
- best RUL model: `LSTM`
- A4 gives the best derived-RUL results overall:
  - best family RUL RMSE `435.62`

Interpretation:

- A4 appears to be the easiest family for stable health-to-RUL conversion
- this likely comes from smoother or more monotone degradation behavior in the coarse-dust family
- A4 is the strongest current candidate for debugging the derivation pipeline because the signal is the cleanest there

## Figure-Based Interpretation

### Health scatter

From:

- `outputs/sequence_model_benchmark/20260313_084355/plots/scatter_health_pred.png`

Observation:

- all four models lie almost perfectly on the diagonal

Conclusion:

- the benchmark's current health target is being learned correctly
- there is no evidence that the pooled-data training setup itself is broken

### RUL scatter

From:

- `outputs/sequence_model_benchmark/20260313_084355/plots/scatter_rul_pred.png`

Observation:

- predicted `RUL` points are highly dispersed
- many points overshoot far above the diagonal
- there is strong vertical spread for the same true `RUL`

Conclusion:

- the main error source is not sequence-model fitting
- the main error source is the current post-processing step that converts health trajectories into `RUL`

### Representative A4 LSTM RUL curve

From:

- `outputs/sequence_model_benchmark/20260313_084355/plots/rul_curve_A4_LSTM.png`

Observation:

- the true `RUL` curve decreases smoothly
- the derived `RUL` curve has repeated spikes and strong amplitude instability

Conclusion:

- even when family-level sequence prediction is good, the current EWMA-based inverse-slope mapping is too noisy
- this confirms that the current bottleneck is `RUL derivation`, not `health regression`

## What This Means For The Project

### What can already be accepted

- the old "pick a few `Test Data_No` as fixed machine lifespan" interpretation can be abandoned
- pooled `Train + Test` supervision is workable
- family-specific sequence modeling is a valid direction
- the benchmark framework itself is useful and should be kept

### What should not yet be accepted

- the current derived `RUL` should not yet be treated as production-ready
- the current `health -> RUL` mapping should not yet be plugged into RL as the final source of truth
- model ranking by health and model ranking by derived RUL are not aligned enough to treat the pipeline as closed

## Recommended Next Step

The next batch should keep the current sequence benchmark and change only the `RUL` derivation layer.

Priority order:

1. Keep `LSTM` as the current default health baseline.
2. Keep `TCN` as the current best derived-RUL reference baseline.
3. Redesign the `health -> RUL` conversion before integrating benchmark output back into the simulator.
4. Re-run the same benchmark protocol after changing only the derivation logic, so improvements can be attributed cleanly.

## Practical Decision For Now

For immediate downstream integration:

- use this batch as evidence that the new sequence-learning setup works
- treat `LSTM` as the current preferred health model
- do not claim that the current derived `RUL` is already satisfactory
- delay RL integration until the derived-RUL instability is reduced
