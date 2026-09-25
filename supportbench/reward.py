from dataclasses import dataclass
from typing import Mapping

from supportbench.cases import CaseSpec


SCORER_VERSION = "scorer-v1"


@dataclass(frozen=True, slots=True)
class RewardVector:
    verified_resolution: int
    satisfaction_change: float
    frustration: float
    repeated_effort: float
    elapsed_time: float
    cost: float
    handoff_quality: float


def verified_resolution(case: CaseSpec, record: Mapping[str, str]) -> bool:
    field, value = case.goal_update
    return record.get("case_id") == case.case_id and record.get(field) == value


def reward_vector(
    case: CaseSpec,
    record: Mapping[str, str],
    *,
    satisfaction: float,
    frustration: float,
    repeated_requests: int,
    elapsed_units: int,
    cost_units: int,
    handoffs: int,
    handoff_quality_total: int,
) -> RewardVector:
    contacts = len(case.contact_prompts)
    return RewardVector(
        verified_resolution=int(verified_resolution(case, record)),
        satisfaction_change=round(satisfaction - 0.5, 3),
        frustration=round(frustration, 3),
        repeated_effort=round(min(repeated_requests / contacts, 1.0), 3),
        elapsed_time=round(min(elapsed_units / (contacts * 8), 1.0), 3),
        cost=round(min(cost_units / (contacts * 10), 1.0), 3),
        handoff_quality=round(handoff_quality_total / handoffs, 3) if handoffs else 0.0,
    )


def terminal_reward(
    vector: RewardVector, *, policy_violation: bool, truncated: bool
) -> float:
    if policy_violation:
        return -1.0
    if truncated:
        return -0.5
    return round(
        0.40 * vector.verified_resolution
        + 0.15 * vector.satisfaction_change
        - 0.10 * vector.frustration
        - 0.10 * vector.repeated_effort
        - 0.10 * vector.elapsed_time
        - 0.10 * vector.cost
        + 0.05 * vector.handoff_quality,
        4,
    )
