import random
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from config import SimConfig
from infer_demo import build_infer_route_result
from run_experiment import (
    _sync_idle_after_maintenance,
    maintenance_prior_penalty,
    maintenance_reward,
    select_maintenance_action,
)
from src.compare import compare_mode_results
from src.env import EventDrivenShopEnv


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

    @staticmethod
    def generative_step(mid: int, particle, action: int, slack_pressure: float, rng: random.Random):
        return dict(particle), float(particle.get("h_true", 1.0)), {"dur": 0.0}


class _SingleStepPOMCP:
    def __init__(self):
        self.last_reward = None

    def plan(self, belief, model, num_sims: int, horizon: int) -> int:
        _, _, reward = model(belief[0], 0)
        self.last_reward = reward
        return 0


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
        self.assertEqual(maintenance_prior_penalty(1, 0.8, cfg, enforce_region=True), 0.0)
        self.assertEqual(maintenance_prior_penalty(0, 0.8, cfg, enforce_region=False), 0.0)

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

        self.assertAlmostEqual(reward_inside, -(12.0 * 0.5 + cfg.IM_COST))
        self.assertLess(reward_early, reward_inside)
        self.assertAlmostEqual(reward_restricted, -(12.0 * 0.5 + cfg.IM_COST))

    def test_compute_costs_uses_fixed_im_cm_action_costs(self):
        cfg = SimConfig()
        env = _CostEnvStub(cfg)

        tard, maint = EventDrivenShopEnv.compute_costs(env)

        self.assertEqual(tard, 0.0)
        self.assertAlmostEqual(
            maint,
            cfg.IM_COST + cfg.CM_COST + cfg.FAIL_COST_MULT * 10.0 + cfg.SCRAP_COST * 10.0 + 77.0,
        )

    def test_fast_iteration_defaults_disable_maint_only_and_use_single_seed(self):
        cfg = SimConfig()

        self.assertEqual(cfg.EXPERIMENT_SEEDS, (42,))
        self.assertFalse(cfg.ENABLE_MAINT_ONLY_COMPARE)
        self.assertFalse(cfg.FAIL_STOCHASTIC)
        self.assertTrue(cfg.RUL_LINEAR_TAIL_ENABLE)
        self.assertIsNone(cfg.RUL_LINEAR_TAIL_STEP)

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

        action = select_maintenance_action(
            "POMCP",
            None,
            planner,
            {0: [particle]},
            env,
            0,
            0.25,
            np.zeros(17, dtype=np.float32),
            0.1,
            0.8,
            cfg,
            random.Random(0),
            explore=False,
        )

        self.assertEqual(action, 0)
        self.assertIsNotNone(env.expected_breakdown_args)
        self.assertAlmostEqual(env.expected_breakdown_args["h_true"], 0.25)
        self.assertAlmostEqual(
            env.expected_breakdown_args["idx_before"],
            env.operating_index_from_rul(0, 0.25),
        )
        self.assertLess(planner.last_reward, 0.0)

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
        env.machine_lifespan = {0: 10}
        env.machine_observed_life = {0: 5}
        env.machine_curve = {0: 4}
        env.rul_cache = _RULCacheStub([1.0, 0.9, 0.8, 0.7, 0.6])
        env.degr = None
        env.rul = None
        env.machine_operating_idx = {0: 6}
        env.machine_operating_frac = {0: 0.0}

        self.assertAlmostEqual(env.rul_from_operating_index(0, 3.5), 0.65)
        self.assertAlmostEqual(env.rul_from_operating_index(0, 6.0), 0.4)
        self.assertAlmostEqual(env.rul_from_operating_index(0, 9.5), 0.05)
        self.assertAlmostEqual(env.operating_index_from_rul(0, 0.05), 9.5)
        self.assertAlmostEqual(env.peek_rul_true(0), 0.4)
        self.assertAlmostEqual(env._peek_rul(0), 0.4)

    def test_true_rul_tail_can_be_disabled_for_plateau_ablation(self):
        cfg = SimConfig()
        cfg.RUL_LINEAR_TAIL_ENABLE = False
        env = EventDrivenShopEnv.__new__(EventDrivenShopEnv)
        env.cfg = cfg
        env.machine_lifespan = {0: 10}
        env.machine_observed_life = {0: 5}
        env.machine_curve = {0: 4}
        env.rul_cache = _RULCacheStub([1.0, 0.9, 0.8, 0.7, 0.6])
        env.degr = None
        env.rul = None

        self.assertAlmostEqual(env.rul_from_operating_index(0, 6.0), 0.6)
        self.assertAlmostEqual(env.rul_from_operating_index(0, 20.0), 0.6)
        self.assertAlmostEqual(env.operating_index_from_rul(0, 0.05), 4.0)

    def test_gru_cache_is_monotone_non_increasing(self):
        from src.sensor_bank import GRUCache

        cache = GRUCache(_PredictorStub([0.9, 0.95, 0.7, 0.72, 0.4]), _BankStub(5), 3)
        cache.build()

        vals = cache.cache[0]
        self.assertTrue(np.allclose(vals, np.array([0.9, 0.9, 0.7, 0.7, 0.4], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
