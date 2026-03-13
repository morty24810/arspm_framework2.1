# Sequence Model Benchmark Task Report

## Task Goal

This batch redefines the standalone sequence benchmark so that `health` matches `remaining-life progress`, not `DP`-proxy clogging severity.

The benchmark now exists to answer one question first:

- can we learn a family-specific sequence representation whose main supervised target is `remaining-life health`, and can that remove the previous `RUL` bouncing problem?

## Implemented In This Batch

### Core Semantic Change

- The main supervised target is now:
  - `health_remaining = RUL / total_life`
- `total_life` is defined per `Test` sequence as:
  - `Time + RUL`
- `health_remaining` is therefore:
  - close to `1` near sequence start
  - close to `0` at failure

This replaces the previous benchmark target:

- `DP`-proxy `health = clip((600 - DP) / (600 - 25), 0, 1)`

### Current Program Behavior

- The standalone entrypoint remains:
  - `benchmark_sequence_models.py`
- The reusable benchmark module remains:
  - `src/sequence_benchmark.py`
- The benchmark still compares:
  - `GRU`
  - `LSTM`
  - `TCN`
  - `ATTENTION`
- The benchmark still runs family-specific experiments for:
  - `A2`
  - `A3`
  - `A4`

### Data Role Redefinition

- `Train_Data_CSV.csv`
  - no longer contributes life-label supervision
  - now serves as family-specific self-supervised pretraining data
- `Test_Data_CSV.csv`
  - now serves as the only source of life-progress supervision
  - also remains the only source of true `RUL` evaluation

### Training Pipeline

For each `family x model x fold`, the benchmark now runs two stages:

1. Self-supervised pretraining
   - data:
     - all selected family `Train` sequences
     - plus the current fold's `train_fold` `Test` sequences
   - task:
     - predict next-step `Differential_pressure`
     - predict next-step `Flow_rate`

2. Supervised fine-tuning
   - data:
     - current fold's `train_fold` `Test` sequences
   - target:
     - window-end `health_remaining`

Validation is drawn only from the supervised `Test` training side of the fold.

### RUL Definition In This Benchmark

The old slope-based `health -> RUL` derivation has been removed.

Current benchmark `RUL` prediction is now defined as:

```text
rul_hat = health_remaining_hat * total_life
```

Important note:

- this works only inside the benchmark because true `total_life` is known from `Test`
- this is a benchmark-side alignment check
- this is not yet the final simulator-side inference rule

### Output Changes

Each prediction CSV now distinguishes:

- `health_remaining_true`
- `health_remaining_pred`
- `dp_proxy_health`
- `rul_true`
- `rul_pred`
- `total_life`
- `is_observed`
- `is_tail_reference`

The benchmark also records:

- `health_target_type = remaining_life_fraction`
- `pretrain_task = next_step_dp_flow`
- `rul_derivation = health_hat_times_total_life`

### Visualization Changes

Representative plots now use two layers:

- observed segment
  - true vs predicted `health_remaining`
  - true vs predicted `RUL`
- reference full-life view
  - true tail is extended to failure using true `RUL`
  - predicted tail is not fabricated beyond the observed segment

## Default Settings

- Models:
  - `GRU,LSTM,TCN,ATTENTION`
- Families:
  - `A2,A3,A4`
- Grouped evaluation:
  - `5-fold`
- Window length:
  - `32`
- Batch size:
  - `64`
- Fine-tuning epochs:
  - `100`
- Fine-tuning patience:
  - `10`
- Pretraining epochs:
  - `20`
- Pretraining patience:
  - `5`
- Optimizer:
  - `Adam`
- Learning rate:
  - `1e-3`

## Explicitly Not Done In This Batch

- No change to `run_experiment.py`
- No change to the current RL path
- No pseudo life labels for `Train`
- No unified conditional model
- No feed interpolation
- No particle-size curve input features
- No direct raw `RUL` regression head
- No simulator-side inference redesign

## How To Run

Default run:

```bash
python benchmark_sequence_models.py
```

Small smoke run:

```bash
python benchmark_sequence_models.py \
  --families A2 \
  --kfolds 2 \
  --epochs 1 \
  --patience 1 \
  --pretrain-epochs 1 \
  --pretrain-patience 1 \
  --batch-size 16 \
  --max-sequences-per-family 4 \
  --max-windows-per-sequence 16 \
  --device cpu
```

## How To Interpret The Result

This benchmark is successful when:

- `health_remaining` is still learnable under grouped `Test`-only supervision
- `RUL` plots become much smoother than the previous slope-based version
- `RUL` scatter no longer shows the previous large vertical explosion
- the benchmark can clearly separate:
  - `DP`-proxy health
  - life-progress health

## Next Step After This Batch

If this version behaves well, the next step is:

- choose the strongest family/model baseline under `health_remaining`
- decide how simulator-side inference will estimate `total_life` or equivalent latent quantity without using true `Test` labels
- only then reconnect benchmark output back to the simulator / RL pipeline
