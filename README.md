# ARSPM-style Event-driven Scheduling + PdM (THDQN + POMCP/DQN Maintenance) — Minimal, Reproducible Framework

This project is a *runnable* reference implementation for an event-driven shop-floor simulator that integrates:
- A **RUL predictor** (pluggable; can load a trained GRU artifact, or use a simple fallback)
- A **maintenance policy** (supports `POMCP` or `DQN`; actions: DN / IM / CM, default `POMCP`)
- A **hierarchical scheduling agent** (High DQN selects goal; Low DQN selects dispatching rule)

It supports **non-stationary Poisson arrivals (rush hour)** via time-varying λ(t) and time-varying due-date tightness DDT(t).

## Quick start

1) Put your files next to the project root (or edit paths in `config.py`):
- `Train_Data_CSV.csv`  (required for degradation replay)
- optional: `rul_model_artifact/` (from your GRU training notebook)

2) Run:

```bash
python run_experiment.py
```

It will:
- Run a paired experiment for each seed in `EXPERIMENT_SEEDS`
- Train both `DQN` and `POMCP` maintenance modes on the same locked scenarios
- Run final evaluation for both `region_on` and `region_off`
- Save per-run artifacts under `outputs/paired_<timestamp>/seed_<seed>/maint_<mode>/`
- Save per-seed `DQN vs POMCP` compare artifacts under `outputs/paired_<timestamp>/seed_<seed>/compare/`

## Final Eval / Inference Output Modes

Final evaluation and `infer_demo.py` can produce two policy routes and two maintenance-mode routes with explicit filename tags:

- `region_on_hx0p3_hy0p1`: Region constraints are enforced (`Hx=0.3`, `Hy=0.1`)
- `region_off_unrestricted`: no Region A/B/C hard enforcement
- `maint_pomcp`: maintenance decisions come from `POMCP`
- `maint_dqn`: maintenance decisions come from `DQN`

Generated artifacts include the policy tag in filename, for example:

- `decision_log_<timestamp>_maint_pomcp_region_on_hx0p3_hy0p1.csv`
- `gantt_<timestamp>_maint_dqn_region_off_unrestricted.png`
- `rul_curves_<timestamp>_maint_pomcp_region_on_hx0p3_hy0p1.png`
- `summary_<timestamp>_maint_dqn_region_off_unrestricted.json` and `.csv`
- `maint_compare_<timestamp>_<policy_tag>.json`, `.csv`, `.png` for `POMCP` vs `DQN` decision diffs
- `aggregate_compare_<policy_tag>.json`, `.csv` for multi-seed aggregate comparison

Each visualization also contains an in-plot policy label:

- `Policy: Region constrained (Hx=0.3, Hy=0.1) | Maintenance: POMCP`
- `Policy: Unrestricted (no Hx/Hy enforcement) | Maintenance: DQN`

For unrestricted runs, the RUL chart still draws `Hx/Hy` lines as reference only (not enforced).

`decision_log` also records:

- `maint_seq_global`
- `maint_seq_machine`

These are used to align `POMCP` vs `DQN` maintenance decisions in `maint_compare_*`.

## Inference Compare Mode

`infer_demo.py` defaults to `--maint_mode POMCP`.

To compare `POMCP` against a dedicated `DQN` maintenance checkpoint while keeping the scheduler fixed from the primary checkpoint:

```bash
python infer_demo.py \
  --ckpt checkpoints/latest.pt \
  --maint_mode POMCP \
  --compare_maint_modes 1 \
  --dqn_ckpt checkpoints/maint_dqn_best.pt
```

This runs both maintenance modes on the same seed / combo plan and emits:

- per-mode `decision_log_*`, `summary_*`, `gantt_*`, `rul_curves_*`
- `maint_compare_*` artifacts showing metric deltas, action distributions, and aligned decision diffs

## Notes
- This is an MVP designed to be extended (better state features, richer rules, better reward shaping).
- The simulator is **event-driven**: decision points occur at job arrivals and operation completions.

## Sequence Model Benchmark

An independent pooled-data sequence benchmark is also available for validating the new time-series training path before reconnecting it to the simulator:

```bash
python benchmark_sequence_models.py
```

This benchmark compares `GRU / LSTM / TCN / ATTENTION` as `family-specific` health regressors on pooled `Train_Data_CSV.csv` and `Test_Data_CSV.csv`, then derives `RUL` from predicted health trajectories.

Implementation details and task-completion notes are recorded in:

- [SEQUENCE_MODEL_BENCHMARK_TASK_REPORT.md](SEQUENCE_MODEL_BENCHMARK_TASK_REPORT.md)
- [SEQUENCE_MODEL_BENCHMARK_RESULT_ANALYSIS.md](SEQUENCE_MODEL_BENCHMARK_RESULT_ANALYSIS.md)
