from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple, List, Optional
import math
import random


@dataclass
class Node:
    n: int = 0
    n_a: Dict[int, int] = field(default_factory=dict)
    q_a: Dict[int, float] = field(default_factory=dict)
    children: Dict[Tuple[int, Any], "Node"] = field(default_factory=dict)

    def ensure_action(self, a: int):
        if a not in self.n_a:
            self.n_a[a] = 0
            self.q_a[a] = 0.0


class POMCPPlanner:
    """
    Minimal POMCP planner for discrete actions and continuous observations (bucketed).
    """
    def __init__(self, num_actions: int, gamma: float, c_ucb: float, rng: random.Random):
        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.c_ucb = float(c_ucb)
        self.rng = rng

    def plan(
        self,
        belief,
        generative_model: Callable,
        num_sims: int,
        horizon: int,
        *,
        root_candidates: Optional[List[int]] = None,
        action_candidates_fn: Optional[Callable[[Any], List[int]]] = None,
    ) -> int:
        root = Node()
        for _ in range(int(num_sims)):
            state = self.rng.choice(belief)
            self._simulate(state, root, generative_model, int(horizon), action_candidates_fn=action_candidates_fn)

        candidates = list(root_candidates) if root_candidates else list(range(self.num_actions))
        if not candidates:
            candidates = list(range(self.num_actions))
        best_a = int(candidates[0])
        best_q = -1e9
        for a in candidates:
            root.ensure_action(a)
            if root.q_a[a] > best_q:
                best_q = root.q_a[a]
                best_a = a
        return best_a

    def _simulate(
        self,
        state,
        node: Node,
        model: Callable,
        depth: int,
        *,
        action_candidates_fn: Optional[Callable[[Any], List[int]]] = None,
    ) -> float:
        if depth <= 0:
            return 0.0

        candidates = action_candidates_fn(state) if action_candidates_fn is not None else None
        a = self._select_action(node, candidates=candidates)
        next_state, obs, reward = model(state, a)
        obs_key = self._obs_key(obs)
        key = (a, obs_key)
        if key not in node.children:
            node.children[key] = Node()
            # rollout for new node
            total = reward + self.gamma * self._rollout(
                next_state,
                model,
                depth - 1,
                action_candidates_fn=action_candidates_fn,
            )
        else:
            total = reward + self.gamma * self._simulate(
                next_state,
                node.children[key],
                model,
                depth - 1,
                action_candidates_fn=action_candidates_fn,
            )

        node.n += 1
        node.ensure_action(a)
        node.n_a[a] += 1
        node.q_a[a] += (total - node.q_a[a]) / max(node.n_a[a], 1)
        return total

    def _select_action(self, node: Node, *, candidates: Optional[List[int]] = None) -> int:
        # UCB1
        action_space = list(candidates) if candidates else list(range(self.num_actions))
        if not action_space:
            action_space = list(range(self.num_actions))
        best_a = int(action_space[0])
        best_u = -1e9
        for a in action_space:
            node.ensure_action(a)
            if node.n_a[a] == 0:
                return a
            u = node.q_a[a] + self.c_ucb * math.sqrt(math.log(max(node.n, 1)) / node.n_a[a])
            if u > best_u:
                best_u = u
                best_a = a
        return best_a

    def _rollout(
        self,
        state,
        model: Callable,
        depth: int,
        *,
        action_candidates_fn: Optional[Callable[[Any], List[int]]] = None,
    ) -> float:
        total = 0.0
        discount = 1.0
        for _ in range(depth):
            candidates = action_candidates_fn(state) if action_candidates_fn is not None else None
            action_space = list(candidates) if candidates else list(range(self.num_actions))
            if not action_space:
                action_space = list(range(self.num_actions))
            a = self.rng.choice(action_space)
            state, _, r = model(state, a)
            total += discount * r
            discount *= self.gamma
        return total

    def _obs_key(self, obs) -> Any:
        # bucket continuous observation to keep the tree size controlled
        try:
            return round(float(obs), 2)
        except Exception:
            return obs
