# ARSPM-style Event-driven Scheduling + PdM (THDQN + Maintenance DDQN) — Minimal, Reproducible Framework

This project is a *runnable* reference implementation for an event-driven shop-floor simulator that integrates:
- A **RUL predictor** (pluggable; can load a trained GRU artifact, or use a simple fallback)
- A **maintenance agent** (DDQN; actions: DN / IM / CM)
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
- Train agents for a small number of episodes (fast default)
- Run one evaluation episode
- Save plots into `outputs/`:
  - Gantt chart with maintenance blocks
  - RUL curves with maintenance markers and thresholds
  - Rule/goal selection vs features

## Final Eval / Inference Output Modes

Final evaluation and `infer_demo.py` can produce two policy routes with explicit filename tags:

- `region_on_hx0p3_hy0p1`: Region constraints are enforced (`Hx=0.3`, `Hy=0.1`)
- `region_off_unrestricted`: no Region A/B/C hard enforcement

Generated artifacts include the policy tag in filename, for example:

- `decision_log_<timestamp>_region_on_hx0p3_hy0p1.csv`
- `gantt_<timestamp>_region_off_unrestricted.png`
- `rul_curves_<timestamp>_region_on_hx0p3_hy0p1.png`
- `summary_<timestamp>_region_off_unrestricted.json` and `.csv`

Each visualization also contains an in-plot policy label:

- `Policy: Region constrained (Hx=0.3, Hy=0.1)`
- `Policy: Unrestricted (no Hx/Hy enforcement)`

For unrestricted runs, the RUL chart still draws `Hx/Hy` lines as reference only (not enforced).

## Notes
- This is an MVP designed to be extended (better state features, richer rules, better reward shaping).
- The simulator is **event-driven**: decision points occur at job arrivals and operation completions.
