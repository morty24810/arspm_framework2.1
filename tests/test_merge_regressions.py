import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from config import SimConfig, resolve_machine_set
from checkpointing import CheckpointManager, load_checkpoint
from infer_demo import (
    build_infer_route_result,
    build_infer_summary_row,
    infer_thdqn_low_state_mode,
    should_use_formal_final_eval_scenario,
)
from run_experiment import (
    _sync_idle_after_maintenance,
    _build_run_record,
    build_episode_combos,
    build_maintenance_state,
    effective_base_degradation_rate,
    effective_degradation_bounds,
    filter_non_improving_im,
    maintenance_agent_arch_for_context,
    maintenance_agent_store_transition,
    maintenance_cm_preference_penalty,
    maintenance_hier_metrics,
    maintenance_hier_rewards,
    maintenance_im_history_features,
    maintenance_low_gain_penalty,
    maintenance_prior_penalty,
    maintenance_reward,
    maintenance_state_dim_for_context,
    normalize_jobs_target_for_combo_mode,
    select_maintenance_action,
)
from src.agents import HierMaintenanceAgentDDQN, MaintenanceAgentDDQN, PPOSchedulerAgent, THDQNAgent
from src.compare import combo_dominant_maps, compare_mode_results, summarize_combo_conditioned_behavior, summarize_scheduling_strategy
from src.env import EventDrivenShopEnv
from src.viz import plot_gantt, plot_rul_curves, plot_rule_vs_features


class _RecoveryEnvStub:
    @staticmethod
    def peek_rul_true(mid: int) -> float:
        return 1.0


class _RolloutEnvStub:
    def __init__(self):
        self.machines = {0: SimpleNamespace(maint_rul_baseline=0.9)}
        self.expected_breakdown_args = None

    @staticmethod
    def get_region_b_elapsed(mid: int, h: float) -> float:
        return 0.0

    @staticmethod
    def operating_index_from_rul(mid: int, h: float) -> float:
        return 40.0 + float(h)

    def expected_breakdown_loss(self, mid: int, local_urgency: float, *,
                                pt=None, stress=None, h_true=None, idx_before=None):
        self.expected_breakdown_args = {
            "mid": mid,
            "local_urgency": local_urgency,
            "pt": pt,
            "stress": stress,
            "h_true": h_true,
            "idx_before": idx_before,
        }
        return {
            "p_fail_exec": 0.25,
            "expected_breakdown_loss": 7.5,
            "breakdown_penalty_cost": 3.0,
            "expected_redispatch_pt": 1.0,
            "recovery_dur": 2.0,
            "hard_breakdown_flag": 0.0,
            "h_end_true": 0.2,
        }

    def im_target_rul(self, mid: int, baseline_rul=None) -> float:
        baseline = self.machines[mid].maint_rul_baseline if baseline_rul is None else float(baseline_rul)
        return float(min(1.0, max(0.0, 0.8 * baseline)))

    @staticmethod
    def generative_step(mid: int, particle, action: int, current_stress: float, rng: random.Random):
        return dict(particle), float(particle.get("h_true", 1.0)), {"dur": 0.0}


class _SingleStepPOMCP:
    def __init__(self):
        self.last_reward = None

    def plan(self, belief, model, num_sims: int, horizon: int, **kwargs) -> int:
        _, _, reward = model(belief[0], 0)
        self.last_reward = reward
        return 0


class _BestRewardPOMCP:
    def __init__(self):
        self.last_rewards = {}

    def plan(self, belief, model, num_sims: int, horizon: int, **kwargs) -> int:
        best_action = None
        best_reward = None
        for action in (1, 2):
            _, _, reward = model(belief[0], action)
            self.last_rewards[action] = reward
            if best_reward is None or reward > best_reward:
                best_reward = reward
                best_action = action
        return int(best_action)


class _CompareEnvStub:
    def __init__(self, breakdown_count: int, breakdown_cost: float,
                 requeued_op_count: int, interrupted_proc_time: float, makespan: float):
        self.breakdown_count = breakdown_count
        self.breakdown_cost_total = breakdown_cost
        self.requeued_op_count = requeued_op_count
        self.interrupted_proc_time = interrupted_proc_time
        self.timeline_ops = [(0, 0.0, makespan - 1.0, 0, 0, "DONE")]
        self.timeline_maint = [(0, makespan - 1.0, makespan, "CM")]
        self.last_decision_log = []


class _ObserverStub:
    def __init__(self):
        self.arrivals = []
        self.op_times = []

    def reset(self):
        self.arrivals.clear()
        self.op_times.clear()


class _ImGainEnvStub(_RolloutEnvStub):
    def __init__(self, allow_im: bool):
        super().__init__()
        self.allow_im = bool(allow_im)
        self.machines = {0: SimpleNamespace(maint_rul_baseline=0.9, im_since_cm=0, im_damage=0.0)}

    def im_has_positive_gain(self, mid: int, h: float, baseline_rul=None) -> bool:
        return self.allow_im


class _LatePreferenceEnvStub(_ImGainEnvStub):
    def __init__(self):
        super().__init__(True)

    @staticmethod
    def get_region_b_elapsed(mid: int, h: float) -> float:
        return 40.0

    @staticmethod
    def generative_step(mid: int, particle, action: int, current_stress: float, rng: random.Random):
        dur = 11.6 if int(action) == 1 else 20.0
        return dict(particle), float(particle.get("h_true", 1.0)), {"dur": dur}


class _HardBreakdownVetoEnvStub(_ImGainEnvStub):
    def __init__(self):
        super().__init__(False)

    def expected_breakdown_loss(self, mid: int, local_urgency: float, *,
                                pt=None, stress=None, h_true=None, idx_before=None):
        return {
            "p_fail_exec": 1.0,
            "expected_breakdown_loss": 99.0,
            "breakdown_penalty_cost": 99.0,
            "expected_redispatch_pt": 1.0,
            "recovery_dur": 5.0,
            "hard_breakdown_flag": 1.0,
            "h_end_true": 0.02,
        }


class _CostEnvStub:
    def __init__(self, cfg):
        self.cfg = cfg
        self.jobs = {}
        self.timeline_maint = [
            (0, 0.0, 10.0, "IM"),
            (0, 10.0, 30.0, "CM"),
            (0, 30.0, 40.0, "FAIL_CM"),
            (0, 40.0, 50.0, "SCRAP"),
            (0, 50.0, 70.0, "BREAKDOWN"),
        ]

    @staticmethod
    def breakdown_penalty_cost(recovery_dur=None):
        return 77.0


class _RULCacheStub:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.cache = {0: self.values}

    def get_h(self, mid: int, idx: int) -> float:
        i = min(max(int(idx), 0), int(self.values.size - 1))
        return float(self.values[i])

    def get_h_obs(self, mid: int, idx: int) -> float:
        return self.get_h(mid, idx)


class _PredictorStub:
    def __init__(self, values):
        self.values = list(values)

    def predict(self, data_no: int, window_array, t_idx: int, lifespan: int) -> float:
        i = min(max(int(t_idx), 0), len(self.values) - 1)
        return float(self.values[i])


class _BankStub:
    def __init__(self, life: int):
        self.machine_curve = {0: 4}
        self._life = int(life)

    def lifespan(self, mid: int) -> int:
        return self._life

    def data_no(self, mid: int) -> int:
        return 4

    def window(self, mid: int, end_idx: int, W: int):
        return np.zeros((W, 1), dtype=np.float32)


class MergeRegressionTests(unittest.TestCase):
    def test_test_dataset_is_right_censored_with_linear_rul_labels(self):
        df = pd.read_csv("Test_Data_CSV.csv")

        terminal_rul = []
        for _, group in df.groupby("Data_No"):
            group = group.sort_values("Time").reset_index(drop=True)
            total_life = group["Time"] + group["RUL"]
            self.assertLess(float((total_life.max() - total_life.min())), 1e-10)
            diffs = group["RUL"].diff().dropna().round(10).unique().tolist()
            self.assertEqual(diffs, [-0.1])
            terminal_rul.append(float(group["RUL"].iloc[-1]))

        self.assertTrue(all(rul > 0.0 for rul in terminal_rul))

    def test_unrestricted_maintenance_prior_penalty_matches_health_range(self):
        cfg = SimConfig()

        self.assertEqual(maintenance_prior_penalty(1, 0.2, cfg, enforce_region=False), 0.0)
        self.assertEqual(maintenance_prior_penalty(2, 0.3, cfg, enforce_region=False), 0.0)
        self.assertGreater(maintenance_prior_penalty(1, 0.8, cfg, enforce_region=False), 0.0)
        self.assertGreater(maintenance_prior_penalty(2, 0.05, cfg, enforce_region=False), 0.0)
        self.assertGreater(
            maintenance_prior_penalty(1, 0.05, cfg, enforce_region=False),
            maintenance_prior_penalty(2, 0.05, cfg, enforce_region=False),
        )
        self.assertGreater(
            maintenance_prior_penalty(2, 0.8, cfg, enforce_region=False),
            maintenance_prior_penalty(1, 0.8, cfg, enforce_region=False),
        )
        self.assertEqual(maintenance_prior_penalty(1, 0.8, cfg, enforce_region=True), 0.0)
        self.assertEqual(maintenance_prior_penalty(0, 0.8, cfg, enforce_region=False), 0.0)

    def test_thdqn_dqn_unrestricted_maintenance_state_expands_to_20_dims(self):
        cfg = SimConfig()
        cfg.SCHEDULER_MODE = "THDQN"
        cfg.MAINT_MODE = "DQN"
        self.assertEqual(maintenance_state_dim_for_context(cfg, "DQN", False), 20)
        self.assertEqual(maintenance_state_dim_for_context(cfg, "DQN", True), 18)
        cfg.SCHEDULER_MODE = "PPO"
        self.assertEqual(maintenance_state_dim_for_context(cfg, "DQN", False), 18)

        state18 = build_maintenance_state(
            0.8, 0.0, 10.0, 0.1, 0.0, 10.0,
            40.0, 1.5, 0.2, 0.0, 0.0,
            0.8, 0.01, 5.0, 0.0,
            0.1, 0.1, 0.1, 0.2,
            state_dim=18,
        )
        state20 = build_maintenance_state(
            0.8, 0.0, 10.0, 0.1, 0.0, 10.0,
            40.0, 1.5, 0.2, 0.0, 0.0,
            0.8, 0.01, 5.0, 0.0,
            0.1, 0.1, 0.1, 0.2,
            im_since_cm_norm=0.33,
            im_damage_norm=0.66,
            state_dim=20,
        )
        self.assertEqual(state18.shape[0], 18)
        self.assertEqual(state20.shape[0], 20)
        self.assertAlmostEqual(float(state20[-2]), 0.33, places=6)
        self.assertAlmostEqual(float(state20[-1]), 0.66, places=6)

    def test_maintenance_im_history_features_are_normalized(self):
        cfg = SimConfig()
        env = _ImGainEnvStub(True)
        env.machines[0].im_since_cm = 2
        env.machines[0].im_damage = 0.5
        count_norm, damage_norm = maintenance_im_history_features(env, 0, cfg)
        self.assertAlmostEqual(count_norm, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(damage_norm, 0.5 / float(cfg.IM_DAMAGE_CAP), places=6)

    def test_cm_preference_penalty_only_hits_thdqn_dqn_unrestricted(self):
        cfg = SimConfig()
        cfg.SCHEDULER_MODE = "THDQN"
        cfg.THDQN_DQN_HIER_MAINT = False
        env = _ImGainEnvStub(True)

        penalty, readiness, active = maintenance_cm_preference_penalty(
            2, 0.95, cfg,
            maint_mode="DQN",
            enforce_region=False,
            env=env,
            mid=0,
            baseline_rul=0.9,
            im_since_cm_norm=0.0,
            im_damage_norm=0.0,
            dn_imminent_breakdown=False,
        )
        self.assertGreater(penalty, 0.0)
        self.assertEqual(readiness, 0.0)
        self.assertTrue(active)

        penalty_safe, _, active_safe = maintenance_cm_preference_penalty(
            2, 0.95, cfg,
            maint_mode="DQN",
            enforce_region=False,
            env=env,
            mid=0,
            baseline_rul=0.9,
            im_since_cm_norm=0.0,
            im_damage_norm=0.0,
            dn_imminent_breakdown=True,
        )
        self.assertEqual(penalty_safe, 0.0)
        self.assertFalse(active_safe)

    def test_im_late_penalty_rises_with_low_health_or_im_history(self):
        cfg = SimConfig()
        cfg.SCHEDULER_MODE = "THDQN"
        cfg.THDQN_DQN_HIER_MAINT = False
        env = _ImGainEnvStub(True)
        penalty, readiness, active = maintenance_cm_preference_penalty(
            1, 0.12, cfg,
            maint_mode="DQN",
            enforce_region=False,
            env=env,
            mid=0,
            baseline_rul=0.9,
            im_since_cm_norm=0.8,
            im_damage_norm=0.3,
            dn_imminent_breakdown=False,
        )
        self.assertGreater(readiness, 0.7)
        self.assertGreater(penalty, 0.0)
        self.assertTrue(active)

    def test_plot_gantt_supports_footer_cards_without_overlap_errors(self):
        jobs = {
            0: SimpleNamespace(completed=True, completion_time=10.0, due=20.0),
        }
        timeline_ops = [(0, 0.0, 10.0, 0, 0, "DONE")]
        timeline_maint = [(0, 10.0, 14.0, "IM"), (0, 14.0, 20.0, "CM")]
        schedule = [
            (0.0, 9.0, 20.0, 1.0),
            (9.0, 18.0, 40.0, 1.5),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "gantt.png"
            plot_gantt(timeline_ops, timeline_maint, jobs, str(out_path), schedule=schedule, policy_label="Policy: Unrestricted")
            self.assertTrue(out_path.exists())

    def test_maintenance_reward_uses_fixed_action_cost_and_unrestricted_prior(self):
        cfg = SimConfig()
        reward_inside = maintenance_reward(
            1,
            12.0,
            0.5,
            0.0,
            False,
            cfg,
            h_for_prior=0.2,
            enforce_region=False,
        )
        reward_early = maintenance_reward(
            1,
            12.0,
            0.5,
            0.0,
            False,
            cfg,
            h_for_prior=0.8,
            enforce_region=False,
        )
        reward_restricted = maintenance_reward(
            1,
            12.0,
            0.5,
            0.0,
            False,
            cfg,
            h_for_prior=0.8,
            enforce_region=True,
        )
        reward_late_im = maintenance_reward(
            1,
            12.0,
            0.0,
            0.0,
            False,
            cfg,
            h_for_prior=0.05,
            enforce_region=False,
        )
        reward_late_cm = maintenance_reward(
            2,
            20.0,
            0.0,
            0.0,
            False,
            cfg,
            h_for_prior=0.05,
            enforce_region=False,
        )

        self.assertAlmostEqual(reward_inside, -(12.0 * 0.5 + cfg.IM_COST))
        self.assertLess(reward_early, reward_inside)
        self.assertAlmostEqual(reward_restricted, -(12.0 * 0.5 + cfg.IM_COST))
        self.assertLess(reward_late_im, reward_late_cm)

    def test_compute_costs_uses_fixed_im_cm_action_costs(self):
        cfg = SimConfig()
        env = _CostEnvStub(cfg)

        tard, maint = EventDrivenShopEnv.compute_costs(env)

        self.assertEqual(tard, 0.0)
        self.assertAlmostEqual(
            maint,
            cfg.IM_COST + cfg.CM_COST + cfg.FAIL_COST_MULT * 10.0 + cfg.SCRAP_COST * 10.0 + 77.0,
        )

    def test_im_duration_growth_is_softened_and_capped(self):
        cfg = SimConfig()
        env = EventDrivenShopEnv.__new__(EventDrivenShopEnv)
        env.cfg = cfg

        self.assertLess(cfg.MT_IM_LINEAR, 0.10)
        short_dur = EventDrivenShopEnv._maintenance_duration(env, 0, 1, region_b_elapsed=50.0)
        long_dur = EventDrivenShopEnv._maintenance_duration(env, 0, 1, region_b_elapsed=5000.0)

        self.assertGreater(long_dur, short_dur)
        self.assertAlmostEqual(long_dur, cfg.MT_IM_MAX)

    def test_stress_enabled_full_matrix_defaults(self):
        cfg = SimConfig()

        self.assertEqual(cfg.EXPERIMENT_SEEDS, (42,))
        self.assertFalse(cfg.ENABLE_MAINT_ONLY_COMPARE)
        self.assertFalse(cfg.FAIL_STOCHASTIC)
        self.assertTrue(cfg.RUL_LINEAR_TAIL_ENABLE)
        self.assertIsNone(cfg.RUL_LINEAR_TAIL_STEP)
        self.assertEqual(cfg.TRAIN_SCHEDULER_MODES, ("THDQN",))
        self.assertEqual(cfg.TRAIN_MAINT_MODES, ("DQN",))
        self.assertAlmostEqual(cfg.DEGRADATION_RATE_SCALE, 1.30)
        self.assertEqual(cfg.SCHEDULER_STATE_DIM, 15)
        self.assertEqual(cfg.MAINTENANCE_STATE_DIM, 18)

    def test_hier_maintenance_arch_only_enables_for_thdqn_dqn_unrestricted(self):
        cfg = SimConfig()
        cfg.SCHEDULER_MODE = "THDQN"
        cfg.MAINT_MODE = "DQN"
        self.assertEqual(maintenance_agent_arch_for_context(cfg, "DQN", False, "THDQN"), "hier_ddqn")
        self.assertEqual(maintenance_agent_arch_for_context(cfg, "DQN", True, "THDQN"), "flat_ddqn")
        self.assertEqual(maintenance_agent_arch_for_context(cfg, "DQN", False, "PPO"), "flat_ddqn")
        self.assertEqual(maintenance_agent_arch_for_context(cfg, "POMCP", False, "THDQN"), "flat_ddqn")

    def test_hier_maintenance_rewards_separate_gate_and_type_penalties(self):
        cfg = SimConfig()
        action_meta = {"maint_gate_action": "MAINT", "maint_type_action": "CM"}
        metrics = maintenance_hier_metrics(
            action_meta,
            0.9,
            0.05,
            False,
            cfg,
            maint_mode="DQN",
            enforce_region=False,
            im_since_cm_norm=0.0,
            im_damage_norm=0.0,
        )
        gate_reward, type_reward = maintenance_hier_rewards(-10.0, action_meta, metrics)
        self.assertLess(gate_reward, -10.0)
        self.assertLess(type_reward, -10.0)
        self.assertTrue(metrics["gate_penalty_active"])
        self.assertTrue(metrics["type_penalty_active"])

    def test_hier_maintenance_transition_storage_splits_gate_and_type_buffers(self):
        cfg = SimConfig()
        cfg.SCHEDULER_MODE = "THDQN"
        cfg.MAINT_MODE = "DQN"
        cfg.ENFORCE_REGION_POLICY = False
        agent = HierMaintenanceAgentDDQN(state_dim=20, cfg=cfg, rng=random.Random(0), device=torch.device("cpu"))
        s = np.zeros(20, dtype=np.float32)
        sp = np.ones(20, dtype=np.float32)
        maintenance_agent_store_transition(
            agent,
            s,
            2,
            -5.0,
            sp,
            0.0,
            action_meta={"maint_gate_action": "MAINT", "maint_type_action": "CM"},
            gate_reward=-6.0,
            type_reward=-7.0,
        )
        self.assertEqual(len(agent.buf_gate), 1)
        self.assertEqual(len(agent.buf_type), 1)

    def test_default_machine_set_and_paper_machine_set_can_be_resolved(self):
        cfg = SimConfig()
        mode, ids, num_machines = resolve_machine_set(cfg)
        self.assertEqual(mode, "current6")
        self.assertEqual(ids, (4, 8, 11, 17, 18, 23))
        self.assertEqual(num_machines, 6)

        cfg.MACHINE_SET_MODE = "paper8"
        mode, ids, num_machines = resolve_machine_set(cfg)
        self.assertEqual(mode, "paper8")
        self.assertEqual(ids, (4, 8, 11, 17, 18, 23, 28, 49))
        self.assertEqual(num_machines, 8)

    def test_paper_gru_defaults_match_paper_profile(self):
        cfg = SimConfig()
        self.assertEqual(cfg.RUL_TRAIN_DATA_NO, 18)
        self.assertEqual(cfg.RUL_VAL_RATIO, 0.2)
        self.assertEqual(cfg.RUL_WINDOW, 30)
        self.assertEqual(cfg.RUL_GRU_HIDDEN_DIM, 40)
        self.assertEqual(cfg.RUL_GRU_BATCH_SIZE, 1024)
        self.assertEqual(cfg.RUL_GRU_EPOCHS, 250)
        self.assertAlmostEqual(cfg.RUL_GRU_DROPOUT, 0.25)
        self.assertAlmostEqual(cfg.RUL_GRU_LR, 1e-3)

    def test_effective_degradation_scaling_applies_to_base_and_bounds(self):
        cfg = SimConfig()

        self.assertAlmostEqual(
            effective_base_degradation_rate(cfg),
            cfg.BASE_DEGRADATION_RATE * cfg.DEGRADATION_RATE_SCALE,
        )
        low_eff, high_eff = effective_degradation_bounds(cfg)
        self.assertAlmostEqual(low_eff, cfg.DEGRAD_LOW * cfg.DEGRADATION_RATE_SCALE)
        self.assertAlmostEqual(high_eff, cfg.DEGRAD_HIGH * cfg.DEGRADATION_RATE_SCALE)

    def test_scheduler_features_include_current_stress_as_15th_dimension(self):
        cfg = SimConfig()
        env = EventDrivenShopEnv.__new__(EventDrivenShopEnv)
        env.cfg = cfg
        env.machines = [
            SimpleNamespace(mid=0, status="IDLE"),
            SimpleNamespace(mid=1, status="PROC"),
        ]
        env.jobs = {}
        env.time = 0.0
        env.arrival_times = []
        env.combo_levels = [(20.0, 1.0), (60.0, 2.0)]
        env.combo_levels_seq = [1]
        env.current_combo_seg = 0
        env.compute_slack_stats = lambda: (10.0, 5.0, 0.0, 0.4)
        env.get_obs_estimates = lambda avg_slack, slack_pressure: (3.0, 20.0, 0.0, 0.0, 1.5, 0.0)
        env._ready_ops = lambda: [1, 2]
        env._peek_rul = lambda mid: 0.6
        env.failure_prob = lambda h: 1.0 - float(h)

        features = EventDrivenShopEnv.get_global_features(env)

        self.assertEqual(features.shape[0], 15)
        self.assertAlmostEqual(float(features[-1]), EventDrivenShopEnv.get_current_stress(env, 0.4))

    def test_scheduler_oracle_mode_replaces_observer_lambda_and_ddt(self):
        cfg = SimConfig()
        cfg.SCHED_REGIME_FEATURE_MODE = "oracle"
        env = EventDrivenShopEnv.__new__(EventDrivenShopEnv)
        env.cfg = cfg
        env.machines = [SimpleNamespace(mid=0, status="IDLE")]
        env.jobs = {}
        env.time = 0.0
        env.arrival_times = []
        env.combo_levels = [(20.0, 1.0), (60.0, 2.0)]
        env.combo_levels_seq = [1]
        env.current_combo_seg = 0
        env.compute_slack_stats = lambda: (10.0, 5.0, 0.0, 0.4)
        env.get_obs_estimates = lambda avg_slack, slack_pressure: (3.0, 20.0, 0.0, 0.0, 1.5, 0.0)
        env._ready_ops = lambda: []
        env._peek_rul = lambda mid: 0.8
        env.failure_prob = lambda h: 1.0 - float(h)

        features = EventDrivenShopEnv.get_global_features(env)

        self.assertEqual(features.shape[0], 15)
        self.assertAlmostEqual(float(features[4]), 60.0)
        self.assertAlmostEqual(float(features[5]), 2.0)

    def test_episode_fixed_training_combo_uses_single_uniform_combo(self):
        cfg = SimConfig()
        cfg.ARRIVAL_LAM_VALUES = (20.0, 40.0)
        cfg.DDT_VALUES = (1.0, 1.5)
        combos, seq = build_episode_combos(cfg, random.Random(7), jobs_target=40, combo_mode="episode_fixed")

        self.assertEqual(len(combos), 1)
        self.assertTrue(all(idx == 0 for idx in seq))
        self.assertIn(tuple(combos[0]), {(20.0, 1.0), (20.0, 1.5), (40.0, 1.0), (40.0, 1.5)})

    def test_grid_full_eval_combo_covers_full_lambda_ddt_grid(self):
        cfg = SimConfig()
        cfg.ARRIVAL_LAM_VALUES = (20.0, 40.0, 60.0)
        cfg.DDT_VALUES = (1.0, 1.5, 2.0)

        combos, seq = build_episode_combos(cfg, random.Random(7), jobs_target=81, combo_mode="grid_full")

        self.assertEqual(len(combos), 9)
        self.assertEqual(seq, list(range(9)))
        self.assertEqual(
            combos,
            [
                (20.0, 1.0), (20.0, 1.5), (20.0, 2.0),
                (40.0, 1.0), (40.0, 1.5), (40.0, 2.0),
                (60.0, 1.0), (60.0, 1.5), (60.0, 2.0),
            ],
        )

    def test_grid_full_normalizes_jobs_target_to_full_grid(self):
        cfg = SimConfig()
        cfg.ARRIVAL_LAM_VALUES = (20.0, 40.0, 60.0)
        cfg.DDT_VALUES = (1.0, 1.5, 2.0)
        cfg.COMBO_SEGMENT_JOBS = 7

        self.assertEqual(normalize_jobs_target_for_combo_mode(cfg, 5, "grid_full"), 63)
        self.assertEqual(normalize_jobs_target_for_combo_mode(cfg, 999, "grid_full"), 63)
        self.assertEqual(normalize_jobs_target_for_combo_mode(cfg, 42, "variable"), 42)

    def test_thdqn_pruned_low_state_excludes_regime_features(self):
        cfg = SimConfig()
        agent = THDQNAgent(state_dim=15, low_state_dim=11, cfg=cfg, rng=random.Random(0), device=torch.device("cpu"))
        full_state = np.arange(15, dtype=np.float32)

        low_state = agent.build_low_state(full_state)

        self.assertEqual(low_state.shape[0], 11)
        np.testing.assert_allclose(low_state, np.array([0, 1, 2, 6, 7, 8, 9, 10, 12, 13, 14], dtype=np.float32))

    def test_infer_legacy_thdqn_low_state_mode_from_checkpoint_shape(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "legacy_thdqn.pt"
            torch.save(
                {
                    "meta": {"config": {"SCHEDULER_MODE": "THDQN"}},
                    "models": {
                        "high_q": {"net.0.weight": torch.zeros((8, 15))},
                        "low_q": {"net.0.weight": torch.zeros((8, 19))},
                    },
                },
                ckpt_path,
            )
            self.assertEqual(infer_thdqn_low_state_mode(ckpt_path, torch.device("cpu")), "full")

    def test_infer_pruned_thdqn_low_state_mode_from_checkpoint_shape(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "pruned_thdqn.pt"
            torch.save(
                {
                    "meta": {"config": {"SCHEDULER_MODE": "THDQN", "THDQN_LOW_STATE_DIM": 11}},
                    "models": {
                        "high_q": {"net.0.weight": torch.zeros((8, 15))},
                        "low_q": {"net.0.weight": torch.zeros((8, 15))},
                    },
                },
                ckpt_path,
            )
            self.assertEqual(infer_thdqn_low_state_mode(ckpt_path, torch.device("cpu")), "pruned")

    def test_maintenance_state_includes_current_stress_as_18th_dimension(self):
        state = build_maintenance_state(
            0.4, -0.02, 12.0,
            0.3, 0.1, 15.0,
            20.0, 1.5, 0.2, 4.0, 8.0,
            0.4, 0.01, 100.0, 80.0,
            0.1, 0.2, 0.3, 0.55,
        )

        self.assertEqual(state.shape[0], 18)
        self.assertAlmostEqual(float(state[-1]), 0.55)

    def test_breakdown_recovery_clears_pending_maintenance(self):
        pending_maint = {2: {"action": 1, "t_e": 10.0, "t_l": 20.0}}
        last_h = {2: 0.22}

        _sync_idle_after_maintenance(
            {"mid": 2, "from_maint": True, "from_breakdown": True},
            pending_maint,
            last_h,
            _RecoveryEnvStub(),
        )

        self.assertNotIn(2, pending_maint)
        self.assertEqual(last_h[2], 1.0)

    def test_pomcp_rollout_uses_particle_state_for_expected_breakdown_loss(self):
        cfg = SimConfig()
        cfg.ENFORCE_REGION_POLICY = False
        env = _RolloutEnvStub()
        planner = _SingleStepPOMCP()
        particle = {
            "h_true": 0.25,
            "stress": 0.9,
            "baseline_rul": 0.9,
            "region_b_elapsed": 0.0,
        }

        action, meta = select_maintenance_action(
            "POMCP",
            None,
            planner,
            {0: [particle]},
            env,
            0,
            0.25,
            np.zeros(18, dtype=np.float32),
            0.1,
            0.8,
            0.2,
            cfg,
            random.Random(0),
            explore=False,
        )

        self.assertEqual(action, 0)
        self.assertFalse(meta["dn_imminent_breakdown_veto"])
        self.assertIsNotNone(env.expected_breakdown_args)
        self.assertAlmostEqual(env.expected_breakdown_args["h_true"], 0.25)
        self.assertAlmostEqual(
            env.expected_breakdown_args["idx_before"],
            env.operating_index_from_rul(0, 0.25),
        )
        self.assertLess(planner.last_reward, 0.0)

    def test_non_improving_im_is_removed_from_allowed_actions(self):
        cfg = SimConfig()
        allowed = filter_non_improving_im(_ImGainEnvStub(False), 0, 0.9, [0, 1, 2])
        self.assertEqual(allowed, [0, 2])

    def test_low_gain_im_penalty_is_applied_near_target(self):
        cfg = SimConfig()
        env = _RolloutEnvStub()

        self.assertEqual(
            maintenance_low_gain_penalty(1, 0.5, cfg, env=env, mid=0, baseline_rul=1.0),
            0.0,
        )
        self.assertGreater(
            maintenance_low_gain_penalty(1, 0.79, cfg, env=env, mid=0, baseline_rul=1.0),
            0.0,
        )
        self.assertEqual(
            maintenance_low_gain_penalty(0, 0.79, cfg, env=env, mid=0, baseline_rul=1.0),
            0.0,
        )

    def test_pomcp_downgrades_non_improving_im_to_dn(self):
        cfg = SimConfig()
        cfg.ENFORCE_REGION_POLICY = False
        env = _ImGainEnvStub(False)
        planner = _SingleStepPOMCP()
        particle = {
            "h_true": 0.9,
            "stress": 0.5,
            "baseline_rul": 1.0,
            "region_b_elapsed": 0.0,
        }

        action, meta = select_maintenance_action(
            "POMCP",
            None,
            planner,
            {0: [particle]},
            env,
            0,
            0.9,
            np.zeros(18, dtype=np.float32),
            0.1,
            0.8,
            0.2,
            cfg,
            random.Random(0),
            explore=False,
        )

        self.assertEqual(action, 0)
        self.assertTrue(meta["im_invalid_flag"])

    def test_pomcp_prefers_cm_over_im_below_hy_in_unrestricted_mode(self):
        cfg = SimConfig()
        cfg.ENFORCE_REGION_POLICY = False
        env = _LatePreferenceEnvStub()
        planner = _BestRewardPOMCP()
        particle = {
            "h_true": 0.05,
            "stress": 0.2,
            "baseline_rul": 1.0,
            "region_b_elapsed": 40.0,
        }

        action, meta = select_maintenance_action(
            "POMCP",
            None,
            planner,
            {0: [particle]},
            env,
            0,
            0.05,
            np.zeros(18, dtype=np.float32),
            0.1,
            0.2,
            0.0,
            cfg,
            random.Random(0),
            explore=False,
        )

        self.assertEqual(action, 2)
        self.assertFalse(meta["dn_imminent_breakdown_veto"])
        self.assertGreater(planner.last_rewards[2], planner.last_rewards[1])

    def test_pomcp_unrestricted_safety_filter_vetoes_imminent_breakdown_dn(self):
        cfg = SimConfig()
        cfg.ENFORCE_REGION_POLICY = False
        cfg.POMCP_UNRESTRICTED_SAFETY_FILTER = True
        env = _HardBreakdownVetoEnvStub()
        planner = _SingleStepPOMCP()
        particle = {
            "h_true": 0.08,
            "stress": 0.4,
            "baseline_rul": 1.0,
            "region_b_elapsed": 0.0,
        }

        action, meta = select_maintenance_action(
            "POMCP",
            None,
            planner,
            {0: [particle]},
            env,
            0,
            0.08,
            np.zeros(18, dtype=np.float32),
            0.1,
            0.4,
            0.2,
            cfg,
            random.Random(0),
            explore=False,
        )

        self.assertEqual(action, 2)
        self.assertTrue(meta["im_invalid_flag"])
        self.assertTrue(meta["dn_imminent_breakdown_veto"])
        self.assertFalse(meta["cm_emergency_override"])
        self.assertIn("DN", meta["safety_filtered_actions"])

    def test_pomcp_unrestricted_high_health_gate_blocks_only_cm(self):
        cfg = SimConfig()
        cfg.ENFORCE_REGION_POLICY = False
        cfg.POMCP_UNRESTRICTED_SAFETY_FILTER = True
        cfg.POMCP_MAINT_ACTION_MAX_H = 0.40
        env = _ImGainEnvStub(True)
        planner = _BestRewardPOMCP()
        particle = {
            "h_true": 0.90,
            "stress": 0.1,
            "baseline_rul": 1.0,
            "region_b_elapsed": 0.0,
        }

        action, meta = select_maintenance_action(
            "POMCP",
            None,
            planner,
            {0: [particle]},
            env,
            0,
            0.90,
            np.zeros(18, dtype=np.float32),
            0.1,
            0.1,
            0.0,
            cfg,
            random.Random(0),
            explore=False,
        )

        self.assertEqual(action, 1)
        self.assertEqual(sorted(meta["allowed_actions"]), [0, 1])
        self.assertIn("CM", meta["safety_filtered_actions"])
        self.assertNotIn("IM", meta["safety_filtered_actions"])
        self.assertFalse(meta["cm_emergency_override"])

    def test_pomcp_unrestricted_high_health_cm_emergency_override(self):
        cfg = SimConfig()
        cfg.ENFORCE_REGION_POLICY = False
        cfg.POMCP_UNRESTRICTED_SAFETY_FILTER = True
        cfg.POMCP_MAINT_ACTION_MAX_H = 0.40
        env = _HardBreakdownVetoEnvStub()
        planner = _SingleStepPOMCP()
        particle = {
            "h_true": 0.90,
            "stress": 0.4,
            "baseline_rul": 1.0,
            "region_b_elapsed": 0.0,
        }

        action, meta = select_maintenance_action(
            "POMCP",
            None,
            planner,
            {0: [particle]},
            env,
            0,
            0.90,
            np.zeros(18, dtype=np.float32),
            0.1,
            0.4,
            0.2,
            cfg,
            random.Random(0),
            explore=False,
        )

        self.assertEqual(action, 2)
        self.assertEqual(meta["allowed_actions"], [2])
        self.assertTrue(meta["im_invalid_flag"])
        self.assertTrue(meta["dn_imminent_breakdown_veto"])
        self.assertTrue(meta["cm_emergency_override"])
        self.assertIn("CM", meta["safety_filtered_actions"])

    def test_pomcp_cm_emergency_override_is_not_set_when_cm_was_not_blocked(self):
        cfg = SimConfig()
        cfg.ENFORCE_REGION_POLICY = False
        cfg.POMCP_UNRESTRICTED_SAFETY_FILTER = True
        cfg.POMCP_MAINT_ACTION_MAX_H = 0.40
        env = _LatePreferenceEnvStub()
        planner = _BestRewardPOMCP()
        particle = {
            "h_true": 0.05,
            "stress": 0.2,
            "baseline_rul": 1.0,
            "region_b_elapsed": 40.0,
        }

        action, meta = select_maintenance_action(
            "POMCP",
            None,
            planner,
            {0: [particle]},
            env,
            0,
            0.05,
            np.zeros(18, dtype=np.float32),
            0.1,
            0.2,
            0.0,
            cfg,
            random.Random(0),
            explore=False,
        )

        self.assertEqual(action, 2)
        self.assertFalse(meta["cm_emergency_override"])

    def test_combo_conditioned_behavior_groups_rule_goal_and_maintenance_counts(self):
        decision_log = [
            {"event": "scheduling", "lambda_true_segment": 20.0, "ddt_true_segment": 1.0, "goal": 0, "rule": 2, "dispatched": True},
            {"event": "scheduling", "lambda_true_segment": 20.0, "ddt_true_segment": 1.0, "goal": 0, "rule": 2, "dispatched": False},
            {"event": "maintenance", "lambda_true_segment": 20.0, "ddt_true_segment": 1.0, "kind": "IM"},
            {"event": "scheduling", "lambda_true_segment": 60.0, "ddt_true_segment": 2.0, "goal": 3, "rule": 5, "dispatched": True},
            {"event": "maintenance", "lambda_true_segment": 60.0, "ddt_true_segment": 2.0, "kind": "CM"},
        ]

        summary = summarize_combo_conditioned_behavior(decision_log)
        dominant_rule_by_combo, dominant_goal_by_combo = combo_dominant_maps(summary)

        self.assertIn("lam=20.0|ddt=1.00", summary)
        self.assertEqual(summary["lam=20.0|ddt=1.00"]["rule_counts"]["2"], 2)
        self.assertEqual(summary["lam=20.0|ddt=1.00"]["goal_counts"]["0"], 2)
        self.assertEqual(summary["lam=20.0|ddt=1.00"]["maint_counts"]["IM"], 1)
        self.assertEqual(summary["lam=60.0|ddt=2.00"]["dominant_rule"]["label"], "5")
        self.assertEqual(dominant_rule_by_combo["lam=60.0|ddt=2.00"]["label"], "5")
        self.assertEqual(dominant_goal_by_combo["lam=20.0|ddt=1.00"]["label"], "0")

    def test_top_level_run_record_includes_combo_modes_and_filter_counts(self):
        env = _CompareEnvStub(0, 0.0, 0, 0.0, 12.0)
        env.cfg = SimConfig()
        env.machine_replay_to_label_scale = {0: 1.0}
        env.last_decision_log = [
            {
                "event": "maintenance",
                "mid": 0,
                "kind": "IM",
                "lambda_true_segment": 20.0,
                "ddt_true_segment": 1.0,
                "im_invalid_flag": True,
                "dn_imminent_breakdown_veto": False,
                "cm_emergency_override": True,
            },
            {
                "event": "scheduling",
                "goal": 1,
                "rule": 2,
                "dispatched": True,
                "current_stress": 0.2,
                "lambda_true_segment": 20.0,
                "ddt_true_segment": 1.0,
            },
        ]
        env.episode_scenario = SimpleNamespace(combos=[(20.0, 1.0)], combo_seq=[0])
        result = {
            "metrics": {"tard": 1.0, "maint": 2.0, "total": 3.0},
            "policy_label": "label",
            "decision_log": list(env.last_decision_log),
            "scheduler_mode": "PPO",
            "sched_regime_feature_mode": "oracle",
            "env": env,
            "overdue": {"ratio_ops": 0.0, "ratio_time": 0.0},
        }

        row = _build_run_record(42, "DQN", "region_off_unrestricted", "region_off_unrestricted", result)

        self.assertEqual(row["train_combo_mode"], "episode_fixed")
        self.assertEqual(row["eval_combo_mode"], "grid_full")
        self.assertEqual(row["lam_ddt_mode"], "variable")
        self.assertEqual(row["im_invalid_filtered_count"], 1)
        self.assertEqual(row["dn_veto_count"], 0)
        self.assertEqual(row["cm_emergency_override_count"], 1)
        self.assertIn("lam=20.0|ddt=1.00", row["combo_behavior"])
        self.assertIn("lam=20.0|ddt=1.00", row["dominant_rule_by_combo"])

    def test_infer_route_results_keep_env_for_compare_metrics(self):
        primary_env = _CompareEnvStub(2, 18.0, 3, 4.5, 11.0)
        compare_env = _CompareEnvStub(5, 30.0, 7, 9.0, 13.0)
        primary_result = build_infer_route_result(
            {"tard": 1.0, "maint": 2.0, "total": 3.0},
            primary_env,
            {"ratio_ops": 0.1, "ratio_time": 0.2},
            "primary",
            "maint_dqn",
        )
        compare_result = build_infer_route_result(
            {"tard": 1.5, "maint": 2.5, "total": 4.0},
            compare_env,
            {"ratio_ops": 0.2, "ratio_time": 0.3},
            "compare",
            "maint_pomcp",
        )

        summary, _ = compare_mode_results(primary_result, compare_result, "DQN", "POMCP")

        self.assertIs(primary_result["env"], primary_env)
        self.assertEqual(summary["primary_metrics"]["breakdown_count"], 2.0)
        self.assertEqual(summary["compare_metrics"]["breakdown_cost"], 30.0)
        self.assertEqual(summary["primary_schedule_summary"]["makespan"], 11.0)
        self.assertEqual(summary["compare_schedule_summary"]["makespan"], 13.0)

    def test_single_rul_uses_observed_segment_plus_linear_tail(self):
        cfg = SimConfig()
        cfg.RUL_LINEAR_TAIL_ENABLE = True
        env = EventDrivenShopEnv.__new__(EventDrivenShopEnv)
        env.cfg = cfg
        env.machine_lifespan = {0: 5}
        env.machine_observed_life = {0: 5}
        env.machine_curve = {0: 4}
        env.machine_label_meta = {
            0: {
                "time_rel": np.asarray([0.0, 8.0], dtype=np.float64),
                "rul_norm": np.asarray([1.0, 0.2], dtype=np.float64),
                "time_rel_end": 8.0,
                "replay_to_label_scale": 2.0,
                "tail_anchor_norm": 0.2,
                "full_index_span": 5.0,
            }
        }
        env.machine_replay_to_label_scale = {0: 2.0}
        env.rul_cache = _RULCacheStub([1.0, 0.9, 0.8, 0.7, 0.6])
        env.degr = None
        env.rul = None
        env.machine_operating_idx = {0: 6}
        env.machine_operating_frac = {0: 0.0}

        self.assertAlmostEqual(env.rul_from_operating_index(0, 3.5), 0.3)
        self.assertAlmostEqual(env.rul_from_operating_index(0, 4.0), 0.2)
        self.assertAlmostEqual(env.rul_from_operating_index(0, 4.5), 0.1)
        self.assertAlmostEqual(env.operating_index_from_rul(0, 0.05), 4.75)
        self.assertAlmostEqual(env.peek_rul_true(0), 0.0)
        self.assertAlmostEqual(env._peek_rul(0), 0.0)

    def test_canonical_rul_is_renormalized_to_one_after_reset(self):
        cfg = SimConfig()
        cfg.RUL_LINEAR_TAIL_ENABLE = True
        env = EventDrivenShopEnv.__new__(EventDrivenShopEnv)
        env.cfg = cfg
        env.machine_lifespan = {0: 5}
        env.machine_observed_life = {0: 5}
        env.machine_curve = {0: 4}
        env.machine_label_meta = {
            0: {
                "time_rel": np.asarray([0.0, 8.0], dtype=np.float64),
                "rul_norm": np.asarray([1.0, 0.2], dtype=np.float64),
                "time_rel_end": 8.0,
                "replay_to_label_scale": 2.0,
                "tail_anchor_norm": 0.2,
                "full_index_span": 5.0,
            }
        }
        env.machine_replay_to_label_scale = {0: 2.0}
        env.rul_cache = _RULCacheStub([0.97, 0.90, 0.80, 0.70, 0.60])
        env.degr = None
        env.rul = None
        env.machine_operating_idx = {0: 0}
        env.machine_operating_frac = {0: 0.0}

        self.assertAlmostEqual(env.rul_from_operating_index(0, 0.0), 1.0)
        self.assertAlmostEqual(env._peek_rul(0), 1.0)

    def test_true_rul_tail_can_be_disabled_for_plateau_ablation(self):
        cfg = SimConfig()
        cfg.RUL_LINEAR_TAIL_ENABLE = False
        env = EventDrivenShopEnv.__new__(EventDrivenShopEnv)
        env.cfg = cfg
        env.machine_lifespan = {0: 5}
        env.machine_observed_life = {0: 5}
        env.machine_curve = {0: 4}
        env.machine_label_meta = {
            0: {
                "time_rel": np.asarray([0.0, 8.0], dtype=np.float64),
                "rul_norm": np.asarray([1.0, 0.2], dtype=np.float64),
                "time_rel_end": 8.0,
                "replay_to_label_scale": 2.0,
                "tail_anchor_norm": 0.2,
                "full_index_span": 5.0,
            }
        }
        env.machine_replay_to_label_scale = {0: 2.0}
        env.rul_cache = _RULCacheStub([1.0, 0.9, 0.8, 0.7, 0.6])
        env.degr = None
        env.rul = None

        self.assertAlmostEqual(env.rul_from_operating_index(0, 6.0), 0.2)
        self.assertAlmostEqual(env.rul_from_operating_index(0, 20.0), 0.2)
        self.assertAlmostEqual(env.operating_index_from_rul(0, 0.05), 4.0)

    def test_label_driven_tail_does_not_freeze_when_observed_life_exceeds_label_span(self):
        cfg = SimConfig()
        cfg.RUL_LINEAR_TAIL_ENABLE = True
        env = EventDrivenShopEnv.__new__(EventDrivenShopEnv)
        env.cfg = cfg
        env.machine_observed_life = {0: 10}
        env.machine_lifespan = {0: 13}
        env.machine_curve = {0: 4}
        env.machine_label_meta = {
            0: {
                "time_rel": np.asarray([0.0, 6.0], dtype=np.float64),
                "rul_norm": np.asarray([1.0, 0.25], dtype=np.float64),
                "time_rel_end": 6.0,
                "replay_to_label_scale": 6.0 / 9.0,
                "tail_anchor_norm": 0.25,
                "full_index_span": 12.0,
            }
        }
        env.machine_replay_to_label_scale = {0: 6.0 / 9.0}
        env.rul_cache = _RULCacheStub([1.0] * 10)
        env.degr = None
        env.rul = None

        anchor = env.rul_from_operating_index(0, 9.0)
        after_tail = env.rul_from_operating_index(0, 10.5)
        self.assertAlmostEqual(anchor, 0.25)
        self.assertLess(after_tail, anchor)
        self.assertAlmostEqual(after_tail, 0.125)

    def test_gru_cache_is_monotone_non_increasing(self):
        from src.sensor_bank import GRUCache

        cache = GRUCache(_PredictorStub([0.9, 0.95, 0.7, 0.72, 0.4]), _BankStub(5), 3)
        cache.build()

        vals = cache.cache[0]
        self.assertTrue(np.all(np.diff(vals) <= 0.0))
        self.assertTrue(np.allclose(vals, np.array([0.9, 0.9, 0.7, 0.7, 0.4], dtype=np.float32), atol=1e-6))

    def test_plot_rul_curves_accepts_segment_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = f"{tmpdir}/rul_segments.png"
            plot_rul_curves(
                {0: [(0.0, 1.0), (10.0, 0.8), (20.0, 0.6)]},
                [(0, 20.0, 25.0, "CM")],
                0.3,
                0.1,
                out_path,
                policy_label="Policy: Unrestricted (no Hx/Hy enforcement) | Scheduler: THDQN | Maintenance: DQN",
                threshold_enforced=False,
                hard_threshold=0.05,
                rul_segments={0: [(0.0, 10.0, 1.0, 0.8, "PROC"), (12.0, 20.0, 0.95, 0.6, "PROC")]},
            )
            self.assertTrue(Path(out_path).exists())

    def test_rule_plot_uses_lambda_ddt_heatmap_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "rule_vs_features_test.png"
            rule_log = [
                (0.0, np.array([0, 0, 0, 1, 20.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), 0, 2),
                (1.0, np.array([0, 0, 0, 1, 20.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), 0, 2),
                (2.0, np.array([0, 0, 0, 1, 60.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), 3, 5),
            ]
            plot_rule_vs_features(rule_log, str(out_path), policy_label="Policy: Unrestricted | Scheduler: THDQN")
            self.assertTrue(out_path.exists())
            self.assertTrue((Path(tmpdir) / "goal_vs_features_test.png").exists())

    def test_infer_summary_row_includes_breakdown_and_stress_fields(self):
        cfg = SimConfig()
        env = _CompareEnvStub(
            breakdown_count=2,
            breakdown_cost=77.0,
            requeued_op_count=3,
            interrupted_proc_time=12.5,
            makespan=50.0,
        )
        env.hard_breakdown_count = 1
        env.stochastic_breakdown_count = 1
        env.last_decision_log = [
            {"event": "scheduling", "goal": None, "rule": 3, "dispatched": True, "current_stress": 0.2},
            {"event": "scheduling", "goal": None, "rule": 1, "dispatched": True, "current_stress": 0.5},
        ]
        summary = build_infer_summary_row(
            "20260402_000000",
            1,
            cfg,
            42,
            80,
            "region_off_unrestricted",
            "Policy",
            "DQN",
            "maint_dqn",
            {"tard": 1.0, "maint": 2.0, "total": 3.0},
            env,
            {"ratio_ops": 0.1, "ratio_time": 0.2, "overdue_ops": 4, "total_ops": 10},
            degradation_rate=72.8,
            degradation_rate_scale=1.3,
        )
        self.assertEqual(summary["breakdown_count"], 2)
        self.assertEqual(summary["breakdown_cost"], 77.0)
        self.assertEqual(summary["requeued_op_count"], 3)
        self.assertEqual(summary["interrupted_proc_time"], 12.5)
        self.assertEqual(summary["hard_breakdown_count"], 1)
        self.assertEqual(summary["stochastic_breakdown_count"], 1)
        self.assertAlmostEqual(summary["current_stress_mean"], 0.35)
        self.assertAlmostEqual(summary["current_stress_max"], 0.5)
        self.assertEqual(summary["degradation_rate"], 72.8)
        self.assertEqual(summary["degradation_rate_scale"], 1.3)
        self.assertEqual(summary["machine_set_mode"], "current6")
        self.assertEqual(summary["machine_curve_ids"], [4, 8, 11, 17, 18, 23])
        self.assertEqual(summary["rul_life_clock_mode"], "label_driven_scaled")

    def test_infer_demo_defaults_to_formal_final_eval_scenario(self):
        args = SimpleNamespace(
            jobs_target=None,
            segment_jobs=None,
            randomize_combos=0,
            combo_plan="",
        )
        self.assertTrue(should_use_formal_final_eval_scenario(args))

        args.randomize_combos = 1
        self.assertFalse(should_use_formal_final_eval_scenario(args))

        args.randomize_combos = 0
        args.jobs_target = 80
        self.assertFalse(should_use_formal_final_eval_scenario(args))

    def test_ppo_scheduler_checkpoint_roundtrip(self):
        cfg = SimConfig()
        cfg.SCHEDULER_MODE = "PPO"
        agent = PPOSchedulerAgent(state_dim=15, cfg=cfg, rng=random.Random(0), device=torch.device("cpu"))
        with torch.no_grad():
            for p in agent.actor.parameters():
                p.add_(0.123)
            for p in agent.critic.parameters():
                p.add_(0.321)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_mgr = CheckpointManager(tmpdir, cfg, torch.device("cpu"))
            ckpt_mgr.save_latest(agent, None, _ObserverStub(), None, {"tard": 1.0, "maint": 2.0, "total": 3.0})
            loaded = PPOSchedulerAgent(state_dim=15, cfg=cfg, rng=random.Random(1), device=torch.device("cpu"))
            load_checkpoint(str(ckpt_mgr.latest_path), sched_agent=loaded, maint_agent=None, observer=None, map_location="cpu")

            actor_key = next(iter(agent.actor.state_dict().keys()))
            critic_key = next(iter(agent.critic.state_dict().keys()))
            self.assertTrue(torch.allclose(agent.actor.state_dict()[actor_key], loaded.actor.state_dict()[actor_key]))
            self.assertTrue(torch.allclose(agent.critic.state_dict()[critic_key], loaded.critic.state_dict()[critic_key]))

    def test_hier_maintenance_checkpoint_roundtrip(self):
        cfg = SimConfig()
        cfg.SCHEDULER_MODE = "THDQN"
        cfg.MAINT_MODE = "DQN"
        cfg.ENFORCE_REGION_POLICY = False
        cfg.MAINT_AGENT_ARCH = "hier_ddqn"
        cfg.MAINTENANCE_STATE_DIM = 20
        sched = THDQNAgent(state_dim=15, cfg=cfg, rng=random.Random(0), device=torch.device("cpu"), low_state_dim=11)
        agent = HierMaintenanceAgentDDQN(state_dim=20, cfg=cfg, rng=random.Random(0), device=torch.device("cpu"))
        with torch.no_grad():
            for p in agent.q_gate.parameters():
                p.add_(0.123)
            for p in agent.q_type.parameters():
                p.add_(0.456)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_mgr = CheckpointManager(tmpdir, cfg, torch.device("cpu"))
            ckpt_mgr.save_latest(sched, agent, _ObserverStub(), None, {"tard": 1.0, "maint": 2.0, "total": 3.0})
            loaded_sched = THDQNAgent(state_dim=15, cfg=cfg, rng=random.Random(1), device=torch.device("cpu"), low_state_dim=11)
            loaded_agent = HierMaintenanceAgentDDQN(state_dim=20, cfg=cfg, rng=random.Random(1), device=torch.device("cpu"))
            load_checkpoint(str(ckpt_mgr.latest_path), sched_agent=loaded_sched, maint_agent=loaded_agent, observer=None, map_location="cpu")

            gate_key = next(iter(agent.q_gate.state_dict().keys()))
            type_key = next(iter(agent.q_type.state_dict().keys()))
            self.assertTrue(torch.allclose(agent.q_gate.state_dict()[gate_key], loaded_agent.q_gate.state_dict()[gate_key]))
            self.assertTrue(torch.allclose(agent.q_type.state_dict()[type_key], loaded_agent.q_type.state_dict()[type_key]))

    def test_scheduler_summary_tolerates_goal_less_ppo_logs(self):
        decision_log = [
            {"event": "scheduling", "goal": None, "rule": 3, "dispatched": True},
            {"event": "scheduling", "goal": None, "rule": 3, "dispatched": False},
            {"event": "scheduling", "goal": None, "rule": 1, "dispatched": True},
        ]
        summary = summarize_scheduling_strategy(decision_log, env=None)

        self.assertEqual(summary["goal_counts"], {"0": 0, "1": 0, "2": 0, "3": 0})
        self.assertEqual(summary["rule_counts"]["3"], 2)
        self.assertEqual(summary["dispatch_count"], 2)

    def test_scheduler_summary_includes_current_stress_stats(self):
        decision_log = [
            {"event": "scheduling", "goal": None, "rule": 3, "dispatched": True, "current_stress": 0.2},
            {"event": "scheduling", "goal": None, "rule": 1, "dispatched": True, "current_stress": 0.5},
            {"event": "scheduling", "goal": None, "rule": 1, "dispatched": False, "current_stress": 0.4},
        ]

        summary = summarize_scheduling_strategy(decision_log, env=None)

        self.assertAlmostEqual(summary["current_stress_mean"], (0.2 + 0.5 + 0.4) / 3.0)
        self.assertAlmostEqual(summary["current_stress_max"], 0.5)


if __name__ == "__main__":
    unittest.main()
