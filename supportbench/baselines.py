from typing import Literal

from supportbench.cases import CaseSpec, get_case
from supportbench.models import (
    Action,
    AskCustomer,
    Handoff,
    HandoffSummary,
    InspectCase,
    Respond,
    SearchGuide,
    UpdateCase,
)


type Policy = Literal["context", "memoryless", "always_handoff"]
POLICIES: tuple[Policy, ...] = ("context", "memoryless", "always_handoff")
BASELINE_VERSION = "scripts-v1"


def _summary(case: CaseSpec) -> HandoffSummary:
    return HandoffSummary(
        case_id=case.case_id,
        reference=case.reference,
        issue=case.issue,
        previous_effort=case.previous_effort,
        next_step=case.human_next_step,
    )


def scripted_actions(policy: Policy, case_id: str) -> tuple[Action, ...]:
    """Authored sanity-check policies, not learned agents or benchmark results."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; choose from {POLICIES}")
    case = get_case(case_id)
    actions: list[Action] = []
    for contact in range(len(case.contact_prompts)):
        if policy == "always_handoff":
            actions.append(Handoff(_summary(case)))
            continue

        if policy == "memoryless":
            actions.append(AskCustomer("reference"))
        actions.append(InspectCase(case.case_id))
        if contact == 0:
            actions.append(SearchGuide(case.product))
            if case.agent_update is None:
                actions.append(Handoff(_summary(case)))
                continue
            field, value = case.agent_update
            actions.append(UpdateCase(case.case_id, field, value))
        if contact + 1 == len(case.contact_prompts):
            actions.append(Respond("resolved"))
        else:
            actions.append(Respond("working"))
    return tuple(actions)
