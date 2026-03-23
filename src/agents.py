from __future__ import annotations
import math, random
import numpy as np
import torch
import torch.nn as nn

from .rl import MLP, ReplayBuffer, ddqn_update

class EpsSchedule:
    def __init__(self, eps_start: float, eps_end: float, decay_steps: int):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = max(1, int(decay_steps))

    def __call__(self, step: int) -> float:
        frac = min(step / self.decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

class MaintenanceAgentDDQN:
    # actions: 0 DN, 1 IM, 2 CM
    def __init__(self, state_dim: int, cfg, rng: random.Random, device):
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.q = MLP(state_dim, 3).to(device)
        self.qt = MLP(state_dim, 3).to(device)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.LR)
        self.buf = ReplayBuffer(cfg.REPLAY_SIZE, rng)
        self.eps = EpsSchedule(cfg.EPS_START, cfg.EPS_END, cfg.EPS_DECAY_STEPS)
        self.steps = 0

    def act(self, s: np.ndarray, explore=True, allowed_actions=None) -> int:
        e = self.eps(self.steps)
        self.steps += 1
        if allowed_actions is None:
            allowed_actions = [0, 1, 2]
        if explore and self.rng.random() < e:
            if self.cfg.MAINT_BIAS_ENABLED:
                h = float(s[0])
                local_urgency = float(s[4])
                if h > self.cfg.Hy:
                    weights = [self.cfg.MAINT_BIAS_DN, self.cfg.MAINT_BIAS_IM, self.cfg.MAINT_BIAS_CM]
                    if local_urgency > self.cfg.URGENCY_BIAS_THRESH:
                        denom = max(1e-6, 1.0 - self.cfg.URGENCY_BIAS_THRESH)
                        frac = (local_urgency - self.cfg.URGENCY_BIAS_THRESH) / denom
                        damp = max(0.1, 1.0 - self.cfg.URGENCY_BIAS_SCALE * frac)
                        weights[1] *= damp
                        weights[2] *= damp
                    total = sum(max(weights[a], 0.0) for a in allowed_actions)
                    if total > 0.0:
                        pick = self.rng.random() * total
                        acc = 0.0
                        for i in allowed_actions:
                            acc += max(weights[i], 0.0)
                            if pick <= acc:
                                return i
            return int(self.rng.choice(allowed_actions))
        with torch.no_grad():
            x = torch.tensor(s[None], dtype=torch.float32, device=self.device)
            q = self.q(x).detach().cpu()[0]
            best_a = max(allowed_actions, key=lambda a: float(q[a].item()))
            return int(best_a)

    def learn(self):
        if len(self.buf) < self.cfg.BATCH:
            return None
        batch = self.buf.sample(self.cfg.BATCH)
        loss = ddqn_update(self.q, self.qt, self.opt, batch, self.cfg.GAMMA, self.device)
        if self.steps % self.cfg.TARGET_UPDATE == 0:
            self.qt.load_state_dict(self.q.state_dict())
        return loss

class THDQNAgent:
    """
    Minimal hierarchical DQN:
    - High: choose goal g in {0..3}
    - Low: choose rule r in {0..5} conditioned on g
    """
    def __init__(self, state_dim: int, cfg, rng: random.Random, device):
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.state_dim = int(state_dim)

        self.q_high = MLP(self.state_dim, 4).to(device)
        self.q_high_t = MLP(self.state_dim, 4).to(device)
        self.q_high_t.load_state_dict(self.q_high.state_dict())
        self.opt_h = torch.optim.Adam(self.q_high.parameters(), lr=cfg.LR)

        self.q_low = MLP(self.state_dim + 4, 6).to(device)
        self.q_low_t = MLP(self.state_dim + 4, 6).to(device)
        self.q_low_t.load_state_dict(self.q_low.state_dict())
        self.opt_l = torch.optim.Adam(self.q_low.parameters(), lr=cfg.LR)

        self.buf_h = ReplayBuffer(cfg.REPLAY_SIZE, rng)
        self.buf_l = ReplayBuffer(cfg.REPLAY_SIZE, rng)

        self.eps = EpsSchedule(cfg.EPS_START, cfg.EPS_END, cfg.EPS_DECAY_STEPS)
        self.steps = 0

    def _onehot_goal(self, g: int):
        v = np.zeros(4, dtype=np.float32)
        v[int(g)] = 1.0
        return v

    def _match_state_dim(self, s: np.ndarray) -> np.ndarray:
        arr = np.asarray(s, dtype=np.float32)
        if arr.shape[0] == self.state_dim:
            return arr
        if arr.shape[0] > self.state_dim:
            return arr[:self.state_dim]
        padded = np.zeros(self.state_dim, dtype=np.float32)
        padded[:arr.shape[0]] = arr
        return padded

    def act(self, s: np.ndarray, explore=True):
        s = self._match_state_dim(s)
        e = self.eps(self.steps)
        self.steps += 1

        # high
        if explore and self.rng.random() < e:
            g = self.rng.randint(0, 3)
        else:
            with torch.no_grad():
                x = torch.tensor(s[None], dtype=torch.float32, device=self.device)
                g = int(torch.argmax(self.q_high(x), dim=1).item())

        sg = np.concatenate([s, self._onehot_goal(g)], axis=0).astype(np.float32)

        # low
        if explore and self.rng.random() < e:
            r = self.rng.randint(0, 5)
        else:
            with torch.no_grad():
                x = torch.tensor(sg[None], dtype=torch.float32, device=self.device)
                r = int(torch.argmax(self.q_low(x), dim=1).item())
        return g, r

    def learn(self):
        losses = {}
        if len(self.buf_h) >= self.cfg.BATCH:
            batch = self.buf_h.sample(self.cfg.BATCH)
            losses["high"] = ddqn_update(self.q_high, self.q_high_t, self.opt_h, batch, self.cfg.GAMMA, self.device)
        if len(self.buf_l) >= self.cfg.BATCH:
            batch = self.buf_l.sample(self.cfg.BATCH)
            losses["low"] = ddqn_update(self.q_low, self.q_low_t, self.opt_l, batch, self.cfg.GAMMA, self.device)

        if self.steps % self.cfg.TARGET_UPDATE == 0:
            self.q_high_t.load_state_dict(self.q_high.state_dict())
            self.q_low_t.load_state_dict(self.q_low.state_dict())
        return losses if losses else None
