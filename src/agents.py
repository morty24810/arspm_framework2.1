from __future__ import annotations
import math, random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

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


class HierMaintenanceAgentDDQN:
    # gate actions: 0 DN, 1 MAINT
    # type actions: 0 IM, 1 CM
    def __init__(self, state_dim: int, cfg, rng: random.Random, device):
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.state_dim = int(state_dim)

        self.q_gate = MLP(self.state_dim, 2).to(device)
        self.q_gate_t = MLP(self.state_dim, 2).to(device)
        self.q_gate_t.load_state_dict(self.q_gate.state_dict())
        self.opt_gate = torch.optim.Adam(self.q_gate.parameters(), lr=cfg.LR)

        self.q_type = MLP(self.state_dim, 2).to(device)
        self.q_type_t = MLP(self.state_dim, 2).to(device)
        self.q_type_t.load_state_dict(self.q_type.state_dict())
        self.opt_type = torch.optim.Adam(self.q_type.parameters(), lr=cfg.LR)

        self.buf_gate = ReplayBuffer(cfg.REPLAY_SIZE, rng)
        self.buf_type = ReplayBuffer(cfg.REPLAY_SIZE, rng)

        self.eps = EpsSchedule(cfg.EPS_START, cfg.EPS_END, cfg.EPS_DECAY_STEPS)
        self.steps = 0

    def _match_state_dim(self, s: np.ndarray) -> np.ndarray:
        arr = np.asarray(s, dtype=np.float32)
        if arr.shape[0] == self.state_dim:
            return arr
        if arr.shape[0] > self.state_dim:
            return arr[:self.state_dim]
        padded = np.zeros(self.state_dim, dtype=np.float32)
        padded[:arr.shape[0]] = arr
        return padded

    def _sample_weighted_action(self, allowed_actions, weights):
        total = sum(max(float(weights[a]), 0.0) for a in allowed_actions)
        if total <= 0.0:
            return int(self.rng.choice(allowed_actions))
        pick = self.rng.random() * total
        acc = 0.0
        for action in allowed_actions:
            acc += max(float(weights[action]), 0.0)
            if pick <= acc:
                return int(action)
        return int(allowed_actions[-1])

    def _gate_weights(self, s: np.ndarray):
        h = float(s[0]) if s.size > 0 else 1.0
        local_urgency = float(s[4]) if s.size > 4 else 0.0
        dn_weight = float(getattr(self.cfg, "MAINT_BIAS_DN", 1.0))
        maint_weight = max(
            float(getattr(self.cfg, "MAINT_BIAS_IM", 1.0)),
            float(getattr(self.cfg, "MAINT_BIAS_CM", 1.0)),
        )
        if h > float(getattr(self.cfg, "Hy", 0.1)) and local_urgency > float(getattr(self.cfg, "URGENCY_BIAS_THRESH", 0.8)):
            denom = max(1e-6, 1.0 - float(getattr(self.cfg, "URGENCY_BIAS_THRESH", 0.8)))
            frac = (local_urgency - float(getattr(self.cfg, "URGENCY_BIAS_THRESH", 0.8))) / denom
            maint_weight *= max(0.1, 1.0 - float(getattr(self.cfg, "URGENCY_BIAS_SCALE", 0.6)) * frac)
        return {0: dn_weight, 1: maint_weight}

    def _type_weights(self, s: np.ndarray):
        h = float(s[0]) if s.size > 0 else 1.0
        local_urgency = float(s[4]) if s.size > 4 else 0.0
        weights = {
            0: float(getattr(self.cfg, "MAINT_BIAS_IM", 1.0)),
            1: float(getattr(self.cfg, "MAINT_BIAS_CM", 1.0)),
        }
        if h > float(getattr(self.cfg, "Hy", 0.1)) and local_urgency > float(getattr(self.cfg, "URGENCY_BIAS_THRESH", 0.8)):
            denom = max(1e-6, 1.0 - float(getattr(self.cfg, "URGENCY_BIAS_THRESH", 0.8)))
            frac = (local_urgency - float(getattr(self.cfg, "URGENCY_BIAS_THRESH", 0.8))) / denom
            damp = max(0.1, 1.0 - float(getattr(self.cfg, "URGENCY_BIAS_SCALE", 0.6)) * frac)
            weights[0] *= damp
            weights[1] *= damp
        return weights

    def act(self, s: np.ndarray, explore=True, gate_allowed_actions=None, type_allowed_actions=None):
        s = self._match_state_dim(s)
        gate_allowed_actions = list(gate_allowed_actions or [0, 1])
        type_allowed_actions = list(type_allowed_actions or [0, 1])
        e = self.eps(self.steps)
        self.steps += 1

        if explore and self.rng.random() < e:
            if bool(getattr(self.cfg, "MAINT_BIAS_ENABLED", False)):
                gate_action = self._sample_weighted_action(gate_allowed_actions, self._gate_weights(s))
            else:
                gate_action = int(self.rng.choice(gate_allowed_actions))
        else:
            with torch.no_grad():
                x = torch.tensor(s[None], dtype=torch.float32, device=self.device)
                q_gate = self.q_gate(x).detach().cpu()[0]
                gate_action = max(gate_allowed_actions, key=lambda a: float(q_gate[a].item()))

        type_action = None
        if int(gate_action) == 1 and type_allowed_actions:
            if explore and self.rng.random() < e:
                if bool(getattr(self.cfg, "MAINT_BIAS_ENABLED", False)):
                    type_action = self._sample_weighted_action(type_allowed_actions, self._type_weights(s))
                else:
                    type_action = int(self.rng.choice(type_allowed_actions))
            else:
                with torch.no_grad():
                    x = torch.tensor(s[None], dtype=torch.float32, device=self.device)
                    q_type = self.q_type(x).detach().cpu()[0]
                    type_action = max(type_allowed_actions, key=lambda a: float(q_type[a].item()))
        return int(gate_action), None if type_action is None else int(type_action)

    def learn(self):
        losses = {}
        if len(self.buf_gate) >= self.cfg.BATCH:
            batch = self.buf_gate.sample(self.cfg.BATCH)
            losses["gate"] = ddqn_update(self.q_gate, self.q_gate_t, self.opt_gate, batch, self.cfg.GAMMA, self.device)
        if len(self.buf_type) >= self.cfg.BATCH:
            batch = self.buf_type.sample(self.cfg.BATCH)
            losses["type"] = ddqn_update(self.q_type, self.q_type_t, self.opt_type, batch, self.cfg.GAMMA, self.device)

        if self.steps % self.cfg.TARGET_UPDATE == 0:
            self.q_gate_t.load_state_dict(self.q_gate.state_dict())
            self.q_type_t.load_state_dict(self.q_type.state_dict())
        return losses if losses else None


class FixedRuleSchedulerAgent:
    """
    Stateless scheduler baseline that always dispatches the same rule.
    Returned goal is None so downstream logging can distinguish fixed-rule
    baselines from learned hierarchical schedulers.
    """

    def __init__(self, rule_id: int):
        self.rule_id = int(rule_id)
        self.steps = 0

    def act(self, s: np.ndarray, explore=True):
        self.steps += 1
        return None, int(self.rule_id)

    def learn(self):
        return None

    def finish_episode(self):
        return None


class THDQNAgent:
    """
    Minimal hierarchical DQN:
    - High: choose goal g in {0..3}
    - Low: choose rule r in {0..5} conditioned on g
    """
    LOW_STATE_KEEP = (0, 1, 2, 6, 7, 8, 9, 10, 12, 13, 14)

    def __init__(self, state_dim: int, cfg, rng: random.Random, device, low_state_dim: int | None = None):
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.state_dim = int(state_dim)
        self.low_state_mode = str(getattr(cfg, "THDQN_LOW_STATE_MODE", "pruned")).strip().lower()
        self.low_state_indices = tuple(self.LOW_STATE_KEEP)
        self.low_state_dim = int(
            low_state_dim
            if low_state_dim is not None
            else getattr(cfg, "THDQN_LOW_STATE_DIM", len(self.low_state_indices))
        )

        self.q_high = MLP(self.state_dim, 4).to(device)
        self.q_high_t = MLP(self.state_dim, 4).to(device)
        self.q_high_t.load_state_dict(self.q_high.state_dict())
        self.opt_h = torch.optim.Adam(self.q_high.parameters(), lr=cfg.LR)

        self.q_low = MLP(self.low_state_dim + 4, 6).to(device)
        self.q_low_t = MLP(self.low_state_dim + 4, 6).to(device)
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

    def _match_low_state_dim(self, s: np.ndarray) -> np.ndarray:
        arr = np.asarray(s, dtype=np.float32)
        if arr.shape[0] == self.low_state_dim:
            return arr
        if arr.shape[0] > self.low_state_dim:
            return arr[:self.low_state_dim]
        padded = np.zeros(self.low_state_dim, dtype=np.float32)
        padded[:arr.shape[0]] = arr
        return padded

    def build_low_state(self, s: np.ndarray) -> np.ndarray:
        high_state = self._match_state_dim(s)
        if self.low_state_mode != "pruned":
            return self._match_low_state_dim(high_state)
        if high_state.shape[0] <= max(self.low_state_indices):
            return self._match_low_state_dim(high_state)
        low = np.asarray([high_state[i] for i in self.low_state_indices], dtype=np.float32)
        return self._match_low_state_dim(low)

    def act(self, s: np.ndarray, explore=True):
        s = self._match_state_dim(s)
        low_s = self.build_low_state(s)
        e = self.eps(self.steps)
        self.steps += 1

        # high
        if explore and self.rng.random() < e:
            g = self.rng.randint(0, 3)
        else:
            with torch.no_grad():
                x = torch.tensor(s[None], dtype=torch.float32, device=self.device)
                g = int(torch.argmax(self.q_high(x), dim=1).item())

        sg = np.concatenate([low_s, self._onehot_goal(g)], axis=0).astype(np.float32)

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


class PPOSchedulerAgent:
    """
    Flat PPO scheduler:
    - choose dispatch rule r in {0..5}
    - no explicit high-level goal head
    """
    def __init__(self, state_dim: int, cfg, rng: random.Random, device):
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.state_dim = int(state_dim)
        self.actor = MLP(self.state_dim, 6).to(device)
        self.critic = MLP(self.state_dim, 1).to(device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(getattr(cfg, "PPO_LR", 3e-4)),
        )
        self.steps = 0
        self._rollout = []

    def _match_state_dim(self, s: np.ndarray) -> np.ndarray:
        arr = np.asarray(s, dtype=np.float32)
        if arr.shape[0] == self.state_dim:
            return arr
        if arr.shape[0] > self.state_dim:
            return arr[:self.state_dim]
        padded = np.zeros(self.state_dim, dtype=np.float32)
        padded[:arr.shape[0]] = arr
        return padded

    def act_with_info(self, s: np.ndarray, explore: bool = True):
        s = self._match_state_dim(s)
        self.steps += 1
        with torch.no_grad():
            x = torch.tensor(s[None], dtype=torch.float32, device=self.device)
            logits = self.actor(x)
            value = float(self.critic(x).squeeze(1).item())
            dist = Categorical(logits=logits)
            if explore:
                action = int(dist.sample().item())
            else:
                action = int(torch.argmax(logits, dim=1).item())
            logprob = float(dist.log_prob(torch.tensor(action, device=self.device)).item())
        return None, action, logprob, value

    def act(self, s: np.ndarray, explore: bool = True):
        goal, action, _, _ = self.act_with_info(s, explore=explore)
        return goal, action

    def store(self, state: np.ndarray, action: int, logprob: float, reward: float, value: float, done: float):
        self._rollout.append({
            "state": self._match_state_dim(state),
            "action": int(action),
            "logprob": float(logprob),
            "reward": float(reward),
            "value": float(value),
            "done": float(done),
        })

    def finish_episode(self):
        if not self._rollout:
            return None
        gamma = float(getattr(self.cfg, "PPO_GAMMA", 0.99))
        gae_lambda = float(getattr(self.cfg, "PPO_GAE_LAMBDA", 0.95))
        clip_eps = float(getattr(self.cfg, "PPO_CLIP", 0.2))
        entropy_coef = float(getattr(self.cfg, "PPO_ENTROPY_COEF", 0.01))
        value_coef = float(getattr(self.cfg, "PPO_VALUE_COEF", 0.5))
        ppo_epochs = max(1, int(getattr(self.cfg, "PPO_EPOCHS", 4)))
        minibatch = max(1, int(getattr(self.cfg, "PPO_MINIBATCH", 64)))

        rewards = [step["reward"] for step in self._rollout]
        values = [step["value"] for step in self._rollout] + [0.0]
        dones = [step["done"] for step in self._rollout]
        advantages = [0.0] * len(self._rollout)
        gae = 0.0
        for t in reversed(range(len(self._rollout))):
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        states = torch.tensor(
            np.stack([step["state"] for step in self._rollout]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor([step["action"] for step in self._rollout], dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor([step["logprob"] for step in self._rollout], dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        losses = []
        indices = list(range(int(states.shape[0])))
        for _ in range(ppo_epochs):
            self.rng.shuffle(indices)
            for start in range(0, len(indices), minibatch):
                batch_idx = indices[start:start + minibatch]
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logprobs = old_logprobs[batch_idx]
                b_returns = returns_t[batch_idx]
                b_adv = adv_t[batch_idx]

                logits = self.actor(b_states)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                values_pred = self.critic(b_states).squeeze(1)

                ratio = torch.exp(new_logprobs - b_old_logprobs)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = torch.mean((b_returns - values_pred) ** 2)
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 1.0)
                self.opt.step()
                losses.append(float(loss.item()))

        self._rollout.clear()
        if not losses:
            return None
        return {"ppo_loss": float(np.mean(losses))}
