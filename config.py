from dataclasses import dataclass

@dataclass
class SimConfig:
    # --- data / artifact paths ---
    TRAIN_CSV: str = "Train_Data_CSV.csv"
    RUL_ARTIFACT_DIR: str = "rul_model_artifact"  # optional

    # --- shop / jobs ---
    NUM_MACHINES: int = 6
    MACHINE_CURVE_IDS: tuple = (7, 1, 35, 42, 28, 48)
    JOBS_TARGET: int = 40           # how many jobs arrive in an episode (approx., eval default)
    MAX_TIME: float = 2000.0        # legacy cap (not used when running by job count only)
    TRAIN_JOBS_TARGET: int = 100     # doubled to increase training operations
    EVAL_JOBS_TARGET: int = 40

    OPS_PER_JOB_MIN: int = 2
    OPS_PER_JOB_MAX: int = 5

    # processing time distribution for feasible machines
    PT_MIN: float = 5.0
    PT_MAX: float = 50.0
    MACHINE_PT_SCALE_MIN: float = 0.7
    MACHINE_PT_SCALE_MAX: float = 1.3

    # flexibility: how many feasible machines per operation
    FEASIBLE_M_MIN: int = 1
    FEASIBLE_M_MAX: int = 4

    # --- arrival / due-date settings ---
    # lam is mean inter-arrival time for exponential arrivals (mean = lam).
    ARRIVAL_LAM_VALUES: tuple = (20.0, 40.0, 60.0)
    # DDT (due-date tightness) is sampled from these discrete values.
    DDT_VALUES: tuple = (1.0, 1.5, 2.0)
    # Number of jobs per scenario segment (combo).
    COMBO_SEGMENT_JOBS: int = 9
    # Randomize lambda/DDT combos during training.
    COMBO_RANDOMIZE: bool = True
    # Used when COMBO_RANDOMIZE is False.
    DEFAULT_COMBOS = [
        (40.0, 1.5),
    ]
    # Rush indicator threshold: smaller mean inter-arrival means heavier load.
    RUSH_LAM_MEAN_THRESH: float = 80.0
    RUSH_SLACK_THRESH: float = 0.7
    DEGRAD_LOW: float = 35.0
    DEGRAD_HIGH: float = 75.0
    CURR_FRAC: float = 0.35
    IM_WEAR_NOISE: float = 0.02  # legacy / unused
    FAIL_COST_NOISE: float = 0.10  # legacy / unused

    # --- degradation / RUL ---
    RUL_WINDOW: int = 30
    DEGRAD_NOISE_STD: float = 0.02
    BASE_DEGRADATION_RATE: float = 56.0
    DEGRAD_ALPHA: float = 1.0
    PT_REF: float = 50.0
    ARRIVAL_WINDOW: float = 200.0
    OBS_WINDOW: float = 200.0
    OBS_PROC_DEFAULT: float = 30.0
    OBS_EPS: float = 1e-6
    RUL_OBS_NOISE: float = 0.01
    SLACK_REF: float = 150.0
    SLACK_SCALE_ALPHA: float = 0.05
    SLACK_SCALE_MIN: float = 80.0
    SLACK_PRESSURE_GAMMA: float = 2.0
    UTIL_REF: float = 0.7
    STRESS_W_SLACK: float = 1.0
    STRESS_W_UTIL: float = 0.5

    # thresholds (Region A/B/C)
    Hx: float = 0.3
    Hy: float = 0.1
    ENFORCE_REGION_POLICY: bool = True

    # maintenance modeling
    MT_CM: float = 20.0                 # fixed CM duration
    MT_IM_BASE: float = 10.0            # a_i in IM duration = a_i + b_i * (T_t - T_x)
    MT_IM_LINEAR: float = 0.10          # b_i in IM duration = a_i + b_i * (T_t - T_x)
    MAINT_IM_RATIO: float = 0.5         # legacy (unused in action-based maintenance duration)
    MAINT_CM_RATIO: float = 1.0         # legacy (unused in action-based maintenance duration)
    IM_RESET: float = 0.80              # legacy (unused in new IM model)
    CM_RESET: float = 1.00              # legacy (unused in new CM model)
    IM_RESET_MIN: float = 0.85          # legacy
    IM_RESET_MAX: float = 0.92          # legacy
    IM_GRACE_TIME: float = 80.0         # legacy
    IM_GRACE_FACTOR: float = 0.4        # legacy
    IM_WEAR: float = 0.06               # legacy / unused
    MIN_MAX_H: float = 0.65             # legacy
    IM_DEGR_BOOST: float = 0.08         # legacy
    IM_DAMAGE_STEP: float = 0.1         # legacy
    IM_TARGET_RUL: float = 0.8          # legacy (unused in geometric-baseline IM model)
    IM_MULT: float = 1.2               # legacy (unused in geometric-baseline IM model)
    IM_GAIN: float = 0.35              # legacy
    IM_GAIN_DECAY: float = 2.5         # legacy
    IM_LONGTERM_W: float = 6.0         # legacy
    OPPORTUNITY_W: float = 3.0         # legacy
    RISK_TAU: float = 0.08
    BREAKDOWN_ENABLE: bool = False
    BREAKDOWN_W: float = 1.0
    URGENCY_REF: float = 0.0
    URGENCY_SCALE: float = 120.0
    URGENCY_CAP: float = 1.0
    URGENCY_BIAS_THRESH: float = 0.8
    URGENCY_BIAS_SCALE: float = 0.6
    MAT_COST_DN: float = 0.0
    MAT_COST_IM: float = 1.0
    MAT_COST_CM: float = 2.0
    SCRAP_PART_COST: float = 20.0
    W_TIME: float = 1.0
    W_MAT: float = 1.0
    W_RISK: float = 1.0
    W_WINDOW_VIOLATION: float = 5.0
    WINDOW_E_BASE: float = 40.0
    WINDOW_L_BASE: float = 120.0
    WINDOW_L_MIN: float = 20.0
    WINDOW_RISK_K: float = 4.0
    SCRAP_K: float = 5.0
    SCRAP_TH: float = 0.6
    SCRAP_COST: float = 12.0
    SCRAP_DOWNTIME: float = 160.0
    FAIL_K: float = 10.0
    FAIL_IM_BETA: float = 0.25
    FAIL_COST: float = 8.0              # legacy / unused
    FAIL_EXTRA_DUR: float = 120.0
    FAIL_PENALTY: float = 12.0
    FAIL_STOCHASTIC: bool = True
    FAIL_COST_MULT: float = 9.0
    RISK_BASE: float = 0.2
    RISK_SCALE: float = 5.0
    RISK_HIGH_PENALTY: float = 20.0
    SLACK_PRESSURE_BETA: float = 1.0
    PRESSURE_W: float = 1.0
    FREQ_W: float = 1.0
    FREQ_TAU: float = 200.0
    RISK_W: float = 1.0
    TREND_W: float = 2.0
    IM_DAMAGE_W: float = 2.0
    SLACK_W: float = 1.0
    DOWNTIME_PENALTY: float = 0.03
    OVERDUE_MAINT_PENALTY: float = 2.0
    ETA_TIGHT: float = 8.0
    DH_TREND: float = 0.02
    TREND_COUNT_K: int = 2
    TREND_PENALTY: float = 3.0
    TREND_IM_PENALTY: float = 1.5
    IM_UNDER_HY_PENALTY: float = 4.0
    IM_COST: float = 1.0
    IM_TIME_WEIGHT: float = 0.02
    CM_COST: float = 5.0
    CM_TIME_WEIGHT: float = 0.02
    ETA_EPS: float = 1e-6
    ETA_CAP: float = 100.0
    MAINT_BIAS_ENABLED: bool = True
    MAINT_BIAS_DN: float = 0.2
    MAINT_BIAS_IM: float = 0.6
    MAINT_BIAS_CM: float = 0.2
    MAINT_MODE: str = "POMCP"
    POMCP_NUM_SIMS: int = 80
    POMCP_HORIZON: int = 6
    POMCP_UCB_C: float = 1.2
    POMCP_PARTICLES: int = 64
    POMCP_OBS_NOISE: float = 0.02
    POMCP_H_DECAY: float = 0.02            # legacy (no longer the primary generative decay)
    # failure probability feature (generative rollout)
    PFAIL_HORIZON: int = 6
    PFAIL_NUM_SIMS: int = 64

    # --- RL training ---
    GAMMA: float = 0.95
    LR: float = 1e-3
    BATCH: int = 256
    REPLAY_SIZE: int = 50_000
    TARGET_UPDATE: int = 500
    EPS_START: float = 1.0
    EPS_END: float = 0.10
    EPS_DECAY_STEPS: int = 20_000

    TRAIN_EPISODES: int = 100
    EVAL_EPISODES: int = 1
    EVAL_EVERY: int = 20
    SAVE_EVERY: int = 20
    EARLY_STOP_ENABLED: bool = True
    EARLY_STOP_WINDOW: int = 10
    EARLY_STOP_PATIENCE: int = 5
    EARLY_STOP_REL_TOL: float = 0.01
    EARLY_STOP_EPS: float = 1e-6
    CURVE_SMOOTH_WINDOW: int = 10
    DIAG_EVERY: int = 1
    SLACK_PRESSURE_BINS = [0.0, 0.33, 0.66, 1.01]
    ETA_BINS = [0.0, 5.0, 15.0, 1000.0]
    IM_COUNT_BINS = [0, 1, 2, 3, 10]
    H_BINS = [0.0, 0.3, 0.7, 1.01]

    MAX_EVENTS: int = 20000
    TARGET_MAINT_DECISIONS_PER_MACHINE: int = 6

    # paired experiment runner
    EXPERIMENT_SEEDS: tuple = (42,)
    TRAIN_MAINT_MODES: tuple = ("DQN", "POMCP")
    TRAIN_POLICY_ROUTES: tuple = ("region_on", "region_off")
    SCENARIO_LOCK_SCOPE: str = "full"
    ENABLE_OOD_DIAGNOSTIC_EVAL: bool = False

    # --- checkpointing ---
    CKPT_DIR: str = "checkpoints"
    SAVE_BEST: bool = True
    BEST_METRIC: str = "total_cost"
    EXPORT_CONFIG_SNAPSHOT: bool = True

    # logging / plots
    SEED: int = 42
