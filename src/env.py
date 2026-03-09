from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import heapq, math, random
import numpy as np

from .observer import ObserverAgent
from .sensor_bank import SensorReplayBank, GRUCache

@dataclass
class Operation:
    job_id: int
    op_id: int
    feasible_machines: List[int]
    proc_times: Dict[int, float]  # machine->time

@dataclass
class Job:
    job_id: int
    arrival: float
    due: float
    urgency: float
    ops: List[Operation]
    next_op: int = 0
    completed: bool = False
    completion_time: Optional[float] = None

@dataclass
class Machine:
    mid: int
    busy_until: float = 0.0
    status: str = "IDLE"   # IDLE, PROC, MAINT
    current: Optional[Tuple[int,int]] = None  # (job_id, op_id)
    maint_count_cm: int = 0
    maint_count_im: int = 0
    im_since_cm: int = 0
    total_im_count: int = 0
    im_damage: float = 0.0
    last_maint_end: float = 0.0
    crossed_Hx_time: Optional[float] = None
    maint_rul_baseline: float = 1.0
    im_grace_until: float = 0.0

class EventDrivenShopEnv:
    """
    Event-driven simulator with two decision points:
    - JOB_ARRIVAL
    - MACHINE_IDLE (after operation or maintenance)
    Maintenance decisions are evaluated *before* scheduling decisions to avoid conflicts.
    """
    def __init__(self, cfg, rng: random.Random, degr, rul_predictor):
        self.cfg = cfg
        self.rng = rng
        self.degr = degr
        self.rul = rul_predictor
        self.observer = ObserverAgent(cfg)

        self.time = 0.0
        self.event_q: List[Tuple[float, int, int, str, Any]] = []
        self._event_seq = 0
        self.machines: List[Machine] = []
        self.jobs: Dict[int, Job] = {}
        self.waiting_ops: List[Operation] = []

        # logs for visualization
        self.timeline_ops = []     # (mid, t0, t1, job_id, op_id)
        self.timeline_maint = []   # (mid, t0, t1, kind)
        self.rul_log = { }         # mid -> list[(t, h)]
        self.im_damage_log = { }   # mid -> list[(t, im_damage)]
        self.rule_log = []         # (t, state_vec, goal, rule)
        self.arrival_times: List[float] = []
        self.rush_schedule: List[Tuple[float, float, float, float]] = []
        self.combo_levels: List[Tuple[float, float]] = []
        self.combo_levels_seq: List[int] = []
        self.combo_segment_jobs: int = 1
        self.episode_combos: Optional[List[Tuple[float, float]]] = None
        self.episode_combo_seq: Optional[List[int]] = None
        self.combo_time_log: List[Tuple[float, float, float, float, int]] = []
        self.current_combo_seg: Optional[int] = None
        self.combo_seg_start_time: float = 0.0
        self.slack_scale: Optional[float] = None
        self.last_slack_pressure: float = 0.0
        self.event_count: int = 0
        self.sensor_bank: Optional[SensorReplayBank] = None
        self.rul_cache: Optional[GRUCache] = None
        self.last_breakdown: Optional[Dict[str, float]] = None
        self.material_cost: float = 0.0
        self.scrap_part_cost: float = 0.0

        # per-machine mapping to degradation curve id
        self.machine_curve: Dict[int, int] = {}
        self.machine_operating_idx: Dict[int, int] = {}  # increments when processing (not idle/maint)
        self.machine_operating_frac: Dict[int, float] = {}
        self.machine_lifespan: Dict[int, int] = {}
        self.machine_time_scale: Dict[int, float] = {}
        self.machine_pt_sum: Dict[int, float] = {}
        self.machine_pt_count: Dict[int, int] = {}
        self.machine_pt_base: Dict[int, float] = {}

    # ------------------- scenario sampling -------------------
    def _build_combo_plan(self):
        if self.episode_combos and self.episode_combo_seq:
            self.combo_levels = [(float(lam), float(ddt)) for lam, ddt in self.episode_combos]
            self.combo_levels_seq = list(self.episode_combo_seq)
            segment_count = max(1, len(self.combo_levels_seq))
            self.combo_segment_jobs = max(1, int(math.ceil(self.cfg.JOBS_TARGET / segment_count)))
            self.combo_time_log = []
            self.current_combo_seg = None
            self.combo_seg_start_time = 0.0
            return
        segment_jobs = max(1, int(getattr(self.cfg, "COMBO_SEGMENT_JOBS", 1)))
        segment_count = max(1, int(math.ceil(self.cfg.JOBS_TARGET / segment_jobs)))

        if not bool(getattr(self.cfg, "COMBO_RANDOMIZE", True)):
            combos = list(getattr(self.cfg, "DEFAULT_COMBOS", []))
            if not combos:
                lam_values = list(getattr(self.cfg, "ARRIVAL_LAM_VALUES", [100.0]))
                ddt_values = list(getattr(self.cfg, "DDT_VALUES", (1.0,)))
                if not ddt_values:
                    ddt_values = [1.0]
                combos = [(float(lam_values[0]), float(ddt_values[0]))]
            seq = [i % len(combos) for i in range(segment_count)]
            self.combo_levels = combos
            self.combo_levels_seq = seq
            self.combo_segment_jobs = segment_jobs
            self.combo_time_log = []
            self.current_combo_seg = None
            self.combo_seg_start_time = 0.0
            return

        lam_values = list(getattr(self.cfg, "ARRIVAL_LAM_VALUES", [100.0]))
        if not lam_values:
            lam_values = [100.0]
        ddt_values = list(getattr(self.cfg, "DDT_VALUES", (1.0,)))
        if not ddt_values:
            ddt_values = [1.0]

        combos = []
        seq = []
        lam_cycle = list(lam_values)
        self.rng.shuffle(lam_cycle)
        for seg in range(segment_count):
            if not lam_cycle:
                lam_cycle = list(lam_values)
                self.rng.shuffle(lam_cycle)
            lam = float(lam_cycle.pop())
            ddt = float(self.rng.choice(ddt_values))
            combos.append((lam, ddt))
            seq.append(seg)

        self.combo_levels = combos
        self.combo_levels_seq = seq
        self.combo_segment_jobs = segment_jobs
        self.combo_time_log = []
        self.current_combo_seg = None
        self.combo_seg_start_time = 0.0

    def _event_priority(self, etype: str) -> int:
        return 0 if etype == "MACHINE_IDLE" else 1

    def _push_event(self, t: float, etype: str, payload: Any):
        heapq.heappush(self.event_q, (t, self._event_priority(etype), self._event_seq, etype, payload))
        self._event_seq += 1

    def _combo_for_job(self, job_id: int) -> Tuple[float, float, int, int]:
        if not self.combo_levels_seq:
            self._build_combo_plan()
        seg = min(job_id // self.combo_segment_jobs, len(self.combo_levels_seq) - 1)
        level_idx = self.combo_levels_seq[seg]
        lam, ddt = self.combo_levels[level_idx]
        return lam, ddt, level_idx, seg

    def _update_combo_on_job(self, job_id: int, arrival: float) -> Tuple[float, float]:
        lam, ddt, level_idx, seg = self._combo_for_job(job_id)
        if self.current_combo_seg is None:
            self.current_combo_seg = seg
            self.combo_seg_start_time = arrival
        elif seg != self.current_combo_seg:
            prev_idx = self.combo_levels_seq[self.current_combo_seg]
            prev_lam, prev_ddt = self.combo_levels[prev_idx]
            self.combo_time_log.append((self.combo_seg_start_time, arrival, prev_lam, prev_ddt, prev_idx))
            self.current_combo_seg = seg
            self.combo_seg_start_time = arrival
        return lam, ddt

    def get_combo_time_log(self, t_end: float) -> List[Tuple[float, float, float, float, int]]:
        log = list(self.combo_time_log)
        if self.current_combo_seg is not None:
            idx = self.combo_levels_seq[self.current_combo_seg]
            lam, ddt = self.combo_levels[idx]
            log.append((self.combo_seg_start_time, t_end, lam, ddt, idx))
        return log

    def _sample_next_arrival(self, t: float, job_id: int) -> float:
        lam, _, _, _ = self._combo_for_job(job_id)
        if lam <= 0:
            return math.inf
        # exponential inter-arrival (mean = lambda)
        u = self.rng.random()
        return t + (-math.log(max(u, 1e-12)) * lam)

    def _make_job(self, job_id: int, arrival: float) -> Job:
        # operations and flexibility
        num_ops = self.rng.randint(self.cfg.OPS_PER_JOB_MIN, self.cfg.OPS_PER_JOB_MAX)
        ops: List[Operation] = []

        # simple job route: op j requires a "type", mapped to subset of machines by feasibility
        for j in range(num_ops):
            k = self.rng.randint(self.cfg.FEASIBLE_M_MIN, min(self.cfg.FEASIBLE_M_MAX, self.cfg.NUM_MACHINES))
            feasible = self.rng.sample(list(range(self.cfg.NUM_MACHINES)), k=k)
            proc = {
                m: self.rng.uniform(self.cfg.PT_MIN, self.cfg.PT_MAX) * self.machine_time_scale.get(m, 1.0)
                for m in feasible
            }
            ops.append(Operation(job_id=job_id, op_id=j, feasible_machines=feasible, proc_times=proc))

        # due date: arrival + sum(avg_pt) * DDT
        _, ddt = self._update_combo_on_job(job_id, arrival)
        avg_sum = 0.0
        for op in ops:
            avg_sum += float(np.mean(list(op.proc_times.values())))
        due = arrival + avg_sum * ddt

        urgency = self.rng.uniform(0.8, 1.2)
        return Job(job_id=job_id, arrival=arrival, due=due, urgency=urgency, ops=ops)

    # ------------------- reset / step -------------------
    def reset(self, machine_curve_ids: List[int]):
        if not machine_curve_ids:
            default_ids = list(getattr(self.cfg, "MACHINE_CURVE_IDS", []))
            if not default_ids:
                raise ValueError("machine_curve_ids must be non-empty (or set cfg.MACHINE_CURVE_IDS).")
            machine_curve_ids = default_ids
        self.time = 0.0
        self.event_q.clear()
        self._event_seq = 0
        self.jobs.clear()
        self.waiting_ops.clear()
        self.arrival_times.clear()
        self.observer.reset()
        self._build_combo_plan()
        self.slack_scale = None
        self.last_slack_pressure = 0.0
        self.event_count = 0
        self.last_breakdown = None
        self.material_cost = 0.0
        self.scrap_part_cost = 0.0

        self.machines = [Machine(mid=i) for i in range(self.cfg.NUM_MACHINES)]
        self.machine_curve = {i: int(machine_curve_ids[i % len(machine_curve_ids)]) for i in range(self.cfg.NUM_MACHINES)}
        self.machine_operating_idx = {i: 0 for i in range(self.cfg.NUM_MACHINES)}
        self.machine_operating_frac = {i: 0.0 for i in range(self.cfg.NUM_MACHINES)}
        lifespans = [self.degr.lifespan(self.machine_curve[i]) for i in range(self.cfg.NUM_MACHINES)]
        avg_life = float(np.mean(lifespans)) if lifespans else 1.0
        self.machine_lifespan = {i: int(lifespans[i]) for i in range(self.cfg.NUM_MACHINES)}
        self.machine_time_scale = {}
        self.machine_pt_sum = {i: 0.0 for i in range(self.cfg.NUM_MACHINES)}
        self.machine_pt_count = {i: 0 for i in range(self.cfg.NUM_MACHINES)}
        self.machine_pt_base = {}
        for i in range(self.cfg.NUM_MACHINES):
            scale = lifespans[i] / max(avg_life, 1e-6)
            scale = min(max(scale, self.cfg.MACHINE_PT_SCALE_MIN), self.cfg.MACHINE_PT_SCALE_MAX)
            self.machine_time_scale[i] = float(scale)
            base_mean = 0.5 * (self.cfg.PT_MIN + self.cfg.PT_MAX) * scale
            self.machine_pt_base[i] = float(base_mean)
        self.sensor_bank = SensorReplayBank(self.degr, self.machine_curve)
        self.rul_cache = GRUCache(self.rul, self.sensor_bank, self.cfg.RUL_WINDOW,
                                  noise_std=self.cfg.RUL_OBS_NOISE, rng=self.rng)
        self.rul_cache.build()

        self.timeline_ops.clear()
        self.timeline_maint.clear()
        self.rul_log = {i: [] for i in range(self.cfg.NUM_MACHINES)}
        self.im_damage_log = {i: [] for i in range(self.cfg.NUM_MACHINES)}
        self.rule_log.clear()

        # schedule first arrival
        t1 = self._sample_next_arrival(0.0, job_id=0)
        if math.isfinite(t1):
            self._push_event(t1, "JOB_ARRIVAL", {"job_id": 0})

        return self._get_global_state()

    def get_obs_estimates(self, avg_slack: float, slack_pressure: float):
        return self.observer.get_features(self.time, avg_slack, slack_pressure)

    def get_global_features(self):
        # scheduling features (observable only)
        idle = sum(1 for m in self.machines if m.status == "IDLE")
        wip = sum(1 for j in self.jobs.values() if (not j.completed and j.arrival <= self.time))
        ready_len = len(self._ready_ops())
        avg_slack, slack_q10, overdue_rate, slack_pressure = self.compute_slack_stats()
        arrivals, lam_hat, _, _, ddt_hat, rush = self.get_obs_estimates(avg_slack, slack_pressure)
        u_ave = self._utilization()
        return np.array([
            idle, wip, ready_len,
            arrivals, lam_hat, ddt_hat,
            avg_slack, slack_q10, slack_pressure,
            u_ave, overdue_rate, rush
        ], dtype=np.float32)

    def _get_global_state(self):
        return self.get_global_features()

    def _ready_ops(self) -> List[Operation]:
        # arrived and previous ops done (regardless of machine idleness)
        ready = []
        in_proc = {m.current for m in self.machines if m.current is not None}
        for j in self.jobs.values():
            if j.completed or j.arrival > self.time:
                continue
            if j.next_op >= len(j.ops):
                continue
            if (j.job_id, j.next_op) in in_proc:
                continue
            ready.append(j.ops[j.next_op])
        return ready

    def _available_ops(self) -> List[Operation]:
        # ready ops that can run on at least one idle feasible machine
        idle_set = {m.mid for m in self.machines if m.status == "IDLE"}
        avail = []
        for op in self._ready_ops():
            if any(mid in idle_set for mid in op.feasible_machines):
                avail.append(op)
        return avail

    def has_idle_machine(self) -> bool:
        return any(m.status == "IDLE" for m in self.machines)

    def has_ready_ops(self) -> bool:
        # "ready" here means dispatchable with at least one idle feasible machine.
        return bool(self._available_ops())

    def _slack_pressure(self, avg_slack: float, scale: float) -> float:
        slack_pressure = 1.0 - math.exp(-max(0.0, -avg_slack) / scale)
        gamma = float(getattr(self.cfg, "SLACK_PRESSURE_GAMMA", 1.0))
        if gamma != 1.0:
            slack_pressure = 1.0 - (1.0 - slack_pressure) ** gamma
        return float(slack_pressure)

    def compute_slack_stats(self) -> Tuple[float, float, float, float]:
        slacks = []
        for j in self.jobs.values():
            if j.completed or j.arrival > self.time:
                continue
            rem = 0.0
            for op in j.ops[j.next_op:]:
                rem += float(np.mean(list(op.proc_times.values())))
            est_completion = self.time + rem
            slacks.append(j.due - est_completion)
        if not slacks:
            avg_slack = float(self.cfg.SLACK_REF)
            slack_q10 = float(self.cfg.SLACK_REF)
            overdue_rate = 0.0
        else:
            avg_slack = float(np.mean(slacks))
            slack_q10 = float(np.quantile(slacks, 0.1))
            overdue_rate = float(np.mean([1.0 if s < 0.0 else 0.0 for s in slacks]))

        abs_slack = abs(avg_slack)
        if self.slack_scale is None:
            scale = max(abs_slack, float(self.cfg.SLACK_SCALE_MIN))
        else:
            scale = max(self.slack_scale, float(self.cfg.SLACK_SCALE_MIN))
        slack_pressure = self._slack_pressure(avg_slack, scale)
        return avg_slack, slack_q10, overdue_rate, slack_pressure

    def update_slack_state(self, avg_slack: float) -> float:
        abs_slack = abs(avg_slack)
        if self.slack_scale is None:
            self.slack_scale = max(abs_slack, float(self.cfg.SLACK_SCALE_MIN))
        else:
            alpha = float(self.cfg.SLACK_SCALE_ALPHA)
            self.slack_scale = (1.0 - alpha) * self.slack_scale + alpha * abs_slack
        scale = max(self.slack_scale, float(self.cfg.SLACK_SCALE_MIN))
        slack_pressure = self._slack_pressure(avg_slack, scale)
        self.last_slack_pressure = slack_pressure
        return float(slack_pressure)

    def _slack_stats(self) -> Tuple[float, float, float, float]:
        # Legacy name kept for compatibility; this is now pure.
        return self.compute_slack_stats()

    def _arrivals_in_window(self) -> float:
        window = max(self.cfg.ARRIVAL_WINDOW, 1e-6)
        t0 = self.time - window
        return float(sum(1 for t in self.arrival_times if t >= t0))

    def _utilization(self) -> float:
        busy = sum(1 for m in self.machines if m.status == "PROC")
        return busy / max(len(self.machines), 1)

    def _mean_proc_time(self, mid: int) -> float:
        count = self.machine_pt_count.get(mid, 0)
        if count > 0:
            return self.machine_pt_sum.get(mid, 0.0) / count
        return self.machine_pt_base.get(mid, 0.5 * (self.cfg.PT_MIN + self.cfg.PT_MAX))

    def _region_b_elapsed(self, mid: int, h: Optional[float] = None) -> float:
        if h is not None and h > self.cfg.Hx:
            return 0.0
        crossed_hx_time = self.machines[mid].crossed_Hx_time
        if crossed_hx_time is None:
            return 0.0
        return max(0.0, self.time - crossed_hx_time)

    def _maintenance_duration(self, mid: int, action: int,
                              h: Optional[float] = None,
                              region_b_elapsed: Optional[float] = None) -> float:
        if action == 2:
            return float(self.cfg.MT_CM)
        if action != 1:
            return 0.0
        if region_b_elapsed is None:
            region_b_elapsed = self._region_b_elapsed(mid, h=h)
        # legacy behavior (commented): ratio-based duration by mean processing time.
        # mean_pt = self._mean_proc_time(mid)
        # return self.cfg.MAINT_IM_RATIO * mean_pt
        return float(self.cfg.MT_IM_BASE + self.cfg.MT_IM_LINEAR * max(0.0, float(region_b_elapsed)))

    def get_region_b_elapsed(self, mid: int, h: Optional[float] = None) -> float:
        return self._region_b_elapsed(mid, h=h)

    # ------------------- maintenance routing -------------------
    def _query_rul(self, mid: int) -> float:
        idx = self.machine_operating_idx[mid]
        if self.rul_cache is not None:
            h = self.rul_cache.get_h_obs(mid, idx)
        else:
            curve = self.machine_curve[mid]
            lifespan = self.degr.lifespan(curve)
            Xw = self.degr.window(curve, end_idx=idx, W=self.cfg.RUL_WINDOW)
            h = self.rul.predict(curve, Xw, t_idx=idx, lifespan=lifespan)
        self.rul_log[mid].append((self.time, float(h)))
        return float(h)

    def _log_rul(self, mid: int, t: float, h: float):
        self.rul_log[mid].append((float(t), float(h)))

    def _peek_rul(self, mid: int) -> float:
        idx = self.machine_operating_idx[mid]
        if self.rul_cache is not None:
            return float(self.rul_cache.get_h_obs(mid, idx))
        curve = self.machine_curve[mid]
        lifespan = self.degr.lifespan(curve)
        Xw = self.degr.window(curve, end_idx=idx, W=self.cfg.RUL_WINDOW)
        return float(self.rul.predict(curve, Xw, t_idx=idx, lifespan=lifespan))

    def maintenance_decision_point(self, mid: int):
        # called when machine becomes IDLE after completing an operation
        h = self._query_rul(mid)
        m = self.machines[mid]
        if h <= self.cfg.Hx and m.crossed_Hx_time is None:
            m.crossed_Hx_time = self.time
        self.im_damage_log[mid].append((self.time, float(m.im_damage)))
        return h

    def failure_prob(self, h: float) -> float:
        z = self.cfg.FAIL_K * (self.cfg.Hy - h)
        base = 1.0 / (1.0 + math.exp(-z))
        return max(0.0, min(1.0, base))

    def get_local_urgency(self, mid: int) -> float:
        ops = [op for op in self._ready_ops() if mid in op.feasible_machines]
        if not ops:
            return 0.0
        slacks = []
        for op in ops:
            job = self.jobs[op.job_id]
            rem = 0.0
            for op2 in job.ops[job.next_op:]:
                rem += float(np.mean(list(op2.proc_times.values())))
            est_completion = self.time + rem
            slacks.append(job.due - est_completion)
        min_slack = min(slacks)
        scale = max(self.cfg.URGENCY_SCALE, 1e-6)
        urgency = (self.cfg.URGENCY_REF - min_slack) / scale
        urgency = max(0.0, urgency)
        return float(min(urgency, self.cfg.URGENCY_CAP))

    def _enforce_region_action(self, action: int, h: Optional[float]) -> int:
        if not bool(getattr(self.cfg, "ENFORCE_REGION_POLICY", True)):
            return int(action)
        if h is None:
            return int(action)
        if h > self.cfg.Hx:
            return 0
        if h < self.cfg.Hy:
            return 2
        return int(action)

    def apply_maintenance(self, mid: int, action: int, h: Optional[float] = None):
        # action: 0 DN, 1 IM, 2 CM
        m = self.machines[mid]
        h_now = h
        if bool(getattr(self.cfg, "ENFORCE_REGION_POLICY", True)):
            if h_now is None:
                h_now = self._query_rul(mid)
            action = self._enforce_region_action(action, h_now)

        def schedule_block(dur: float, kind: str):
            t0 = self.time
            t1 = self.time + dur
            m.status = "MAINT"
            m.busy_until = t1
            self.timeline_maint.append((mid, t0, t1, kind))
            self._push_event(t1, "MACHINE_IDLE", {"mid": mid, "from_maint": True})

        def log_damage():
            self.im_damage_log[mid].append((self.time, float(m.im_damage)))

        def reset_operating_state():
            m.crossed_Hx_time = None
            self.machine_operating_idx[mid] = 0
            self.machine_operating_frac[mid] = 0.0
            m.im_grace_until = 0.0

        if action == 0:
            log_damage()
            return 0.0, "DN", None

        if action == 2:
            # CM: fixed duration, reset health by resetting operating idx (replay restart)
            dur = self._maintenance_duration(mid, action=2)
            m.maint_count_cm += 1
            m.maint_count_im = 0
            m.im_since_cm = 0
            m.im_damage = 0.0
            m.maint_rul_baseline = 1.0
            reset_operating_state()
            kind = "CM"
            post_rul = 1.0
            schedule_block(dur, kind)
            log_damage()
            self.material_cost += self.cfg.MAT_COST_CM
            return dur, kind, post_rul

        h_pre = h_now if h_now is not None else self._query_rul(mid)
        dur = self._maintenance_duration(mid, action=1, h=h_pre)
        if bool(getattr(self.cfg, "ENFORCE_REGION_POLICY", True)) and h_pre < self.cfg.Hy:
            # IM invalid below Hy; fall back to CM to preserve feasibility.
            dur = self._maintenance_duration(mid, action=2)
            m.maint_count_cm += 1
            m.maint_count_im = 0
            m.im_since_cm = 0
            m.im_damage = 0.0
            m.maint_rul_baseline = 1.0
            reset_operating_state()
            kind = "CM"
            post_rul = 1.0
            schedule_block(dur, kind)
            log_damage()
            self.material_cost += self.cfg.MAT_COST_CM
            return dur, kind, post_rul

        m.maint_count_im += 1
        m.im_since_cm += 1
        m.total_im_count += 1

        curve = self.machine_curve[mid]
        lifespan = self.degr.lifespan(curve)
        # legacy behavior (commented): fixed-target repair.
        # target_rul = max(0.0, min(1.0, float(self.cfg.IM_TARGET_RUL)))
        # new behavior: geometric maintenance baseline, L_k = 0.8 * L_{k-1}.
        target_rul = max(0.0, min(1.0, 0.8 * float(m.maint_rul_baseline)))
        m.maint_rul_baseline = target_rul
        self.machine_operating_idx[mid] = int((1.0 - target_rul) * (lifespan - 1))
        self.machine_operating_frac[mid] = 0.0
        m.crossed_Hx_time = None
        kind = "IM"
        post_rul = target_rul
        schedule_block(dur, kind)
        log_damage()
        self.material_cost += self.cfg.MAT_COST_IM
        return dur, kind, post_rul

    def generative_step(self, mid: int, particle: Dict[str, float], action: int,
                        slack_pressure: float, rng: random.Random):
        h = float(particle.get("h_true", 1.0))
        stress = float(particle.get("stress", slack_pressure))
        baseline_rul = float(particle.get("baseline_rul", 1.0))
        region_b_elapsed = max(0.0, float(particle.get("region_b_elapsed", 0.0)))

        if bool(getattr(self.cfg, "ENFORCE_REGION_POLICY", True)):
            action = self._enforce_region_action(action, h)

        kind = "DN"
        dur = 0.0
        if action == 2:
            kind = "CM"
            dur = self._maintenance_duration(mid, action=2)
            baseline_rul = 1.0
            region_b_elapsed = 0.0
            h = 1.0
        elif action == 1:
            kind = "IM"
            dur = self._maintenance_duration(mid, action=1, region_b_elapsed=region_b_elapsed)
            # legacy behavior (commented): fixed-target repair.
            # h = max(0.0, min(1.0, float(self.cfg.IM_TARGET_RUL)))
            # new behavior: geometric maintenance baseline, L_k = 0.8 * L_{k-1}.
            baseline_rul = max(0.0, min(1.0, 0.8 * baseline_rul))
            region_b_elapsed = 0.0
            h = baseline_rul

        h_before_decay = h
        degr = self.cfg.POMCP_H_DECAY * (1.0 + self.cfg.DEGRAD_ALPHA * stress)
        h = max(0.0, h - degr)
        if action == 0:
            if h > self.cfg.Hx:
                region_b_elapsed = 0.0
            elif h_before_decay > self.cfg.Hx:
                region_b_elapsed = 0.0
            else:
                region_b_elapsed += float(self.cfg.OBS_PROC_DEFAULT)

        p_fail = self.failure_prob(h)
        obs = h + rng.normalvariate(0.0, self.cfg.POMCP_OBS_NOISE)
        obs = max(0.0, min(1.0, obs))
        next_particle = {
            "h_true": h,
            "stress": stress,
            "baseline_rul": baseline_rul,
            "region_b_elapsed": region_b_elapsed,
        }
        info = {"kind": kind, "dur": dur, "p_fail": p_fail}
        return next_particle, obs, info

    # ------------------- scheduling -------------------
    def dispatch(self, rule_id: int):
        # deterministic composite dispatching rules (MVP)
        self.last_breakdown = None
        avail = self._available_ops()
        if not avail:
            return False

        # helper: job objects
        def job_of(op): return self.jobs[op.job_id]

        # machine selection helpers
        def earliest_idle_machine(op: Operation) -> int:
            idle = [m.mid for m in self.machines if m.status == "IDLE" and m.mid in op.feasible_machines]
            return min(idle)  # tie-break: smallest id

        def shortest_pt_machine(op: Operation) -> int:
            idle = [m.mid for m in self.machines if m.status == "IDLE" and m.mid in op.feasible_machines]
            return min(idle, key=lambda mid: op.proc_times[mid])

        # job-level metrics
        def rpt(job: Job) -> float:
            s = 0.0
            for op in job.ops[job.next_op:]:
                s += float(np.mean(list(op.proc_times.values())))
            return s

        def est_completion(job: Job) -> float:
            # crude estimate: now + remaining mean proc times
            return self.time + rpt(job)

        def tard(job: Job) -> float:
            return max(est_completion(job) - job.due, 0.0)

        def slack(job: Job) -> float:
            return job.due - est_completion(job)

        # --- operation selection (6 rules) ---
        if rule_id == 0:  # EDD + earliest idle
            op = min(avail, key=lambda o: job_of(o).due)
            mid = earliest_idle_machine(op)
        elif rule_id == 1:  # EDD + SPT machine
            op = min(avail, key=lambda o: job_of(o).due)
            mid = shortest_pt_machine(op)
        elif rule_id == 2:  # max urgency*tard + SPT
            op = max(avail, key=lambda o: job_of(o).urgency * tard(job_of(o)))
            mid = shortest_pt_machine(op)
        elif rule_id == 3:  # SRPT + SPT
            op = min(avail, key=lambda o: rpt(job_of(o)))
            mid = shortest_pt_machine(op)
        elif rule_id == 4:  # LRPT + earliest idle
            op = max(avail, key=lambda o: rpt(job_of(o)))
            mid = earliest_idle_machine(op)
        else:  # slack min + earliest idle
            op = min(avail, key=lambda o: slack(job_of(o)))
            mid = earliest_idle_machine(op)

        # start processing
        m = self.machines[mid]
        pt = float(op.proc_times[mid])
        t0 = self.time
        t1 = self.time + pt
        h = self._peek_rul(mid)
        self._log_rul(mid, t0, h)
        if self.cfg.BREAKDOWN_ENABLE:
            p_break = self.failure_prob(h) * (pt / max(self.cfg.PT_REF, 1e-6)) * self.cfg.BREAKDOWN_W
            p_break = max(0.0, min(1.0, p_break))
            if self.rng.random() < p_break:
                dur = self._maintenance_duration(mid, action=2)
                t1 = self.time + dur
                m.status = "MAINT"
                m.busy_until = t1
                m.current = None
                m.maint_count_cm += 1
                m.maint_count_im = 0
                m.im_since_cm = 0
                m.im_damage = 0.0
                m.crossed_Hx_time = None
                m.maint_rul_baseline = 1.0
                m.im_grace_until = 0.0
                self.machine_operating_idx[mid] = 0
                self.machine_operating_frac[mid] = 0.0
                self.timeline_maint.append((mid, self.time, t1, "BREAKDOWN"))
                self._push_event(t1, "MACHINE_IDLE", {"mid": mid, "from_maint": True})
                self.material_cost += self.cfg.MAT_COST_CM + self.cfg.SCRAP_PART_COST
                self.scrap_part_cost += self.cfg.SCRAP_PART_COST
                self.last_breakdown = {
                    "mid": float(mid),
                    "time": float(self.time),
                    "dur": float(dur),
                    "cost": float(self.cfg.SCRAP_PART_COST),
                    "p_break": float(p_break),
                }
                return False
        self.machine_pt_sum[mid] += pt
        self.machine_pt_count[mid] += 1
        m.status = "PROC"
        m.busy_until = t1
        m.current = (op.job_id, op.op_id)
        self.timeline_ops.append((mid, t0, t1, op.job_id, op.op_id))
        self._push_event(t1, "MACHINE_IDLE", {"mid": mid, "job_id": op.job_id, "pt": pt, "t0": t0})
        return True

    # ------------------- event loop -------------------
    def step_until_decision(self):
        """
        Advances time to next event, processes it, and returns (done, event_type, payload).
        Decision logic:
        - On MACHINE_IDLE(mid): maintenance decision before any scheduling.
        - On JOB_ARRIVAL: schedule only if dispatchable ops exist.
        """
        if not self.event_q:
            return True, None, None

        t, _, _, etype, payload = heapq.heappop(self.event_q)
        self.time = t
        self.event_count += 1
        if self.event_count >= self.cfg.MAX_EVENTS:
            return True, "MAX_EVENTS", None

        if etype == "JOB_ARRIVAL":
            jid = payload["job_id"]
            job = self._make_job(jid, arrival=self.time)
            self.jobs[jid] = job
            self.arrival_times.append(self.time)
            self.observer.update_on_job_arrival(self.time)
            # schedule next arrival if not too many
            if jid + 1 < self.cfg.JOBS_TARGET:
                next_jid = jid + 1
                tnext = self._sample_next_arrival(self.time, job_id=next_jid)
                if math.isfinite(tnext):
                    self._push_event(tnext, "JOB_ARRIVAL", {"job_id": next_jid})
            avg_slack, _, _, _ = self.compute_slack_stats()
            self.update_slack_state(avg_slack)
            return False, "JOB_ARRIVAL", payload

        if etype == "MACHINE_IDLE":
            mid = payload["mid"]
            m = self.machines[mid]
            if payload.get("from_maint"):
                m.status = "IDLE"
                m.last_maint_end = self.time
                self._query_rul(mid)
                avg_slack, _, _, _ = self.compute_slack_stats()
                self.update_slack_state(avg_slack)
                return False, "MACHINE_IDLE", payload

            jid = payload["job_id"]
            pt = float(payload.get("pt", 0.0))
            t0 = float(payload.get("t0", self.time - pt))

            self.observer.update_on_op_complete(self.time, op_duration=pt)
            avg_slack, _, _, _ = self.compute_slack_stats()
            slack_pressure = self.update_slack_state(avg_slack)
            slack_stress = slack_pressure
            util = self._utilization()
            util_ref = max(self.cfg.UTIL_REF, 1e-6)
            util_stress = max(0.0, (util - util_ref) / util_ref)
            stress = self.cfg.STRESS_W_SLACK * slack_stress + self.cfg.STRESS_W_UTIL * util_stress
            effective_rate = self.cfg.BASE_DEGRADATION_RATE * (1.0 + self.cfg.DEGRAD_ALPHA * stress)
            delta = effective_rate * pt / self.cfg.PT_REF
            acc = self.machine_operating_frac.get(mid, 0.0) + delta
            inc = int(acc)
            if inc > 0:
                self.machine_operating_idx[mid] += inc
                acc -= inc
            self.machine_operating_frac[mid] = acc

            m.status = "IDLE"
            m.current = None

            # update job progress
            j = self.jobs[jid]
            j.next_op += 1
            if j.next_op >= len(j.ops):
                j.completed = True
                j.completion_time = self.time

            return False, "MACHINE_IDLE", payload

        return False, etype, payload

    def done(self):
        # end when all planned arrivals have happened AND all arrived jobs completed AND no events
        if self.event_count >= self.cfg.MAX_EVENTS:
            return True
        if not self.event_q:
            return True
        arrivals_done = (max(self.jobs.keys()) if self.jobs else -1) >= (self.cfg.JOBS_TARGET - 1)
        all_completed = all(j.completed for j in self.jobs.values()) if self.jobs else False
        return arrivals_done and all_completed

    # --- rewards (simple but shaped) ---
    def compute_costs(self):
        # total tardiness cost
        tard = 0.0
        for j in self.jobs.values():
            if j.completed and j.completion_time is not None:
                tard += max(j.completion_time - j.due, 0.0) * j.urgency
        # maintenance cost proxy: proportional to maintenance time blocks
        maint = 0.0
        for _, t0, t1, kind in self.timeline_maint:
            if kind == "CM":
                maint += self.cfg.CM_COST * (t1 - t0)
            elif kind == "IM":
                maint += self.cfg.IM_COST * (t1 - t0)
            elif kind == "SCRAP":
                maint += self.cfg.SCRAP_COST * (t1 - t0)
            elif kind == "FAIL_CM":
                maint += self.cfg.FAIL_COST_MULT * (t1 - t0)
            elif kind == "BREAKDOWN":
                maint += self.cfg.CM_COST * (t1 - t0)
        return tard, maint
