from __future__ import annotations

from typing import Dict, Tuple


ALL_RULE_LABELS: Dict[int, str] = {
    0: "EDD + earliest idle",
    1: "EDD + SPT machine",
    2: "max urgency*tard + SPT",
    3: "SRPT + SPT",
    4: "LRPT + earliest idle",
    5: "min slack + earliest idle",
}

# Thesis scheduler line keeps only the three rules that remained both
# behaviorally distinct and empirically competitive in the stable pre-coverage runs.
ACTIVE_RULE_IDS: Tuple[int, ...] = (1, 2, 3)
ACTIVE_RULE_LABELS: Dict[int, str] = {rule_id: ALL_RULE_LABELS[rule_id] for rule_id in ACTIVE_RULE_IDS}
RULE_ID_TO_ACTION_INDEX: Dict[int, int] = {rule_id: idx for idx, rule_id in enumerate(ACTIVE_RULE_IDS)}
ACTION_INDEX_TO_RULE_ID: Dict[int, int] = {idx: rule_id for idx, rule_id in enumerate(ACTIVE_RULE_IDS)}


def active_rule_action_dim() -> int:
    return int(len(ACTIVE_RULE_IDS))


def scheduler_rule_id_from_action_index(action_index: int) -> int:
    idx = int(action_index)
    if idx not in ACTION_INDEX_TO_RULE_ID:
        raise ValueError(f"unsupported scheduler action index: {action_index}")
    return int(ACTION_INDEX_TO_RULE_ID[idx])


def scheduler_action_index_from_rule_id(rule_id: int) -> int:
    rid = int(rule_id)
    if rid not in RULE_ID_TO_ACTION_INDEX:
        raise ValueError(f"unsupported active scheduler rule id: {rule_id}")
    return int(RULE_ID_TO_ACTION_INDEX[rid])
