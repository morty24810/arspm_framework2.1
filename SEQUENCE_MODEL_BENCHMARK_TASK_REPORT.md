# Sequence Model Benchmark Task Report

## Task Goal

This task adds an independent sequence-model benchmark program for the new time-series training direction.

The goal of this batch is:

- stop using the old interpretation of "pick a few `Test Data_No` entries as fixed machine lifespans"
- pool `Train_Data_CSV.csv` and `Test_Data_CSV.csv` into one sequence sample space
- train `family-specific` sequence regressors for `A2 / A3 / A4`
- compare `GRU / LSTM / TCN / Attention`
- evaluate whether `normalized health` regression and derived `RUL` behavior are reasonable before reconnecting anything to the simulator or RL pipeline

## Completed In This Task

### Implemented

- Added a new standalone entrypoint:
  - `benchmark_sequence_models.py`
- Added a reusable benchmark module:
  - `src/sequence_benchmark.py`
- Implemented pooled data loading from:
  - `Train_Data_CSV.csv`
  - `Test_Data_CSV.csv`
- Implemented sequence identity as:
  - `source_split + Data_No`
- Implemented `family-specific` training for:
  - `GRU`
  - `LSTM`
  - `TCN`
  - `ATTENTION`
- Implemented grouped cross-validation by `sequence_id`
- Implemented validation split inside each outer fold
- Implemented standardized many-to-one window regression for `normalized health`
- Implemented derived `RUL` calculation from predicted health trajectories
- Implemented benchmark outputs:
  - metrics tables
  - model rankings
  - fold prediction artifacts
  - representative health/RUL trajectory plots
  - metric bar plots
  - predicted-vs-true scatter plots

### Default Settings

- Models:
  - `GRU,LSTM,TCN,ATTENTION`
- Families:
  - `A2,A3,A4`
- Cross-validation:
  - grouped `5-fold`
- Window length:
  - `32`
- Batch size:
  - `64`
- Epoch budget:
  - `100`
- Early stopping patience:
  - `10`
- Optimizer:
  - `Adam`
- Learning rate:
  - `1e-3`
- Training target:
  - window-end `normalized health`
- Derived RUL:
  - computed from predicted health with `5`-step EWMA decay

### Explicitly Not Done In This Task

- No change to `run_experiment.py`
- No change to the current RL training or evaluation path
- No reconnection to `DegradationReplay`
- No reuse of `RULPredictorWrapper`
- No unified conditional model
- No feed interpolation
- No particle-size curve input features
- No direct raw `RUL` supervision
- No hyperparameter sweep
- No replacement of the current simulator-side health source

## Data Scope And Label Definition

### Data Used

- `Train_Data_CSV.csv` and `Test_Data_CSV.csv` are both used for health supervision
- only `Test_Data_CSV.csv` contributes true `RUL` values for derived-RUL evaluation

### Sequence Key

- each sequence is keyed as `train_XX` or `test_XX`
- this avoids collisions because both files use `Data_No = 1..50`

### Input Features

- `Differential_pressure`
- `d(Differential_pressure)/dt`
- `Flow_rate`
- `Dust_feed`

### Health Label

The benchmark uses:

```text
health = clip((600 - Differential_pressure) / (600 - 25), 0, 1)
```

Interpretation:

- `25 Pa` is the clean-filter baseline
- `600 Pa` is the failure threshold
- the window-end health value is the supervised regression target

### Dataset Constraint Recorded

`A2` has asymmetric feed coverage between `Train` and `Test`.

The benchmark records this asymmetry in the dataset summary outputs and prints it in the console summary, but does not interpolate missing feed levels in this batch.

## How To Run

Default run:

```bash
python benchmark_sequence_models.py
```

Example smoke run:

```bash
python benchmark_sequence_models.py \
  --families A2 \
  --kfolds 2 \
  --epochs 1 \
  --patience 1 \
  --batch-size 16 \
  --max-sequences-per-family 4 \
  --max-windows-per-sequence 24 \
  --device cpu
```

## Output Structure

Each run writes to:

```text
outputs/sequence_model_benchmark/<timestamp>/
```

Important files:

- `benchmark_config.json`
- `dataset_summary.json`
- `dataset_summary.csv`
- `metrics_by_fold.csv`
- `metrics_by_family.csv`
- `metrics_overall.csv`
- `metrics_overall.json`
- `model_rankings.csv`

Important directories:

- `artifacts/<family>/<model>/`
  - fold-level checkpoints
  - fold history CSV
  - fold summary JSON
  - fold prediction CSV
- `plots/`
  - representative `health_curve_*`
  - representative `rul_curve_*`
  - `metric_bars_overall.png`
  - `scatter_health_pred.png`
  - `scatter_rul_pred.png`

## How To Judge Whether The Result Matches Expectations

The result is considered aligned with this batch's goal when:

- all four sequence models can complete the benchmark run
- grouped folds show no sequence leakage
- health metrics are produced for every family/model/fold
- derived RUL metrics are produced on held-out `Test` windows
- representative trajectory plots show predicted health following the major degradation trend
- derived RUL curves are directionally reasonable and not dominated by obvious instability
- the ranking table clearly identifies a best current baseline for the next integration step

This batch is not trying to prove final production quality. It is trying to verify that the new pooled-data, family-specific sequence training setup is coherent and measurable.

## Next Integration Step

If the benchmark result is acceptable, the next batch should:

- choose the winning sequence backend from this benchmark
- expose its artifact format as a stable inference interface
- connect that inference path to the simulator's health source
- then update maintenance-state construction and downstream RL experiments to consume the new health output

That integration is intentionally left out of this task.
