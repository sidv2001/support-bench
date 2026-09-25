"""Original synthetic, multi-contact support environment."""

from supportbench.cases import GENERATOR_VERSION, case_ids
from supportbench.env import SupportEnv
from supportbench.models import (
    AskCustomer,
    Handoff,
    HandoffSummary,
    InspectCase,
    Observation,
    Respond,
    SearchGuide,
    UpdateCase,
)
from supportbench.replay import record_actions, record_script, replay_trace
from supportbench.reward import SCORER_VERSION, RewardVector

__all__ = [
    "AskCustomer",
    "GENERATOR_VERSION",
    "Handoff",
    "HandoffSummary",
    "InspectCase",
    "Observation",
    "Respond",
    "RewardVector",
    "SCORER_VERSION",
    "SearchGuide",
    "SupportEnv",
    "UpdateCase",
    "case_ids",
    "record_actions",
    "record_script",
    "replay_trace",
]
