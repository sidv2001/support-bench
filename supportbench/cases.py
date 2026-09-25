from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


GENERATOR_VERSION = "cases-v1"


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case_id: str
    product: str
    issue: str
    reference: str
    previous_effort: str
    preference: str
    contact_prompts: tuple[tuple[str, str], ...]
    initial_record: tuple[tuple[str, str], ...]
    guide: str
    agent_update: tuple[str, str] | None
    goal_update: tuple[str, str]
    human_next_step: str
    requires_human: bool


CASES: Mapping[str, CaseSpec] = MappingProxyType(
    {
        "lamp-ring": CaseSpec(
            case_id="lamp-ring",
            product="GroveDesk lamp",
            issue="missing_diffuser",
            reference="DEMO-L-041",
            previous_effort="checked_all_packing",
            preference="brief_updates",
            contact_prompts=(
                (
                    "My GroveDesk lamp has no diffuser ring. Reference DEMO-L-041; "
                    "I checked all the packing.",
                    "I checked every box for the GroveDesk diffuser ring. "
                    "My reference is DEMO-L-041.",
                ),
                (
                    "I'm back about the ring. I already gave the reference last time; "
                    "is a spare queued?",
                    "Following up on the missing ring. Did the replacement request go through?",
                ),
            ),
            initial_record=(
                ("case_id", "lamp-ring"),
                ("status", "open"),
                ("fulfillment", "not_queued"),
            ),
            guide=(
                "For a confirmed missing diffuser ring, inspect the matching case "
                "and this guide before queueing one spare. A queued spare is not "
                "a delivery promise."
            ),
            agent_update=("fulfillment", "part_queued"),
            goal_update=("fulfillment", "part_queued"),
            human_next_step="queue_part",
            requires_human=False,
        ),
        "fan-setup": CaseSpec(
            case_id="fan-setup",
            product="PebbleFan",
            issue="setup_instructions",
            reference="DEMO-F-108",
            previous_effort="tried_quickstart_card",
            preference="step_by_step",
            contact_prompts=(
                (
                    "My PebbleFan setup card skipped a step. Reference DEMO-F-108; "
                    "I tried the card already.",
                    "I tried the PebbleFan quick-start card but need the full steps. "
                    "Reference DEMO-F-108.",
                ),
                (
                    "Checking back: were the complete setup steps sent?",
                    "I came back for the full setup steps; I don't want to repeat "
                    "the reference.",
                ),
                (
                    "One last follow-up: can you confirm the guide was sent?",
                    "I'm back again to check the guide delivery.",
                ),
            ),
            initial_record=(
                ("case_id", "fan-setup"),
                ("status", "open"),
                ("guide_delivery", "not_sent"),
            ),
            guide=(
                "After inspecting the matching case, dispatch the full PebbleFan "
                "setup guide through the simulated delivery record. Do not mark "
                "the case resolved just because an agent says the guide was sent."
            ),
            agent_update=("guide_delivery", "full_guide_sent"),
            goal_update=("guide_delivery", "full_guide_sent"),
            human_next_step="send_setup_guide",
            requires_human=False,
        ),
        "lamp-tracking": CaseSpec(
            case_id="lamp-tracking",
            product="GroveDesk lamp",
            issue="tracking_mismatch",
            reference="DEMO-L-309",
            previous_effort="compared_confirmation",
            preference="human_review",
            contact_prompts=(
                (
                    "The tracking row conflicts with my GroveDesk confirmation. "
                    "Reference DEMO-L-309; I compared both. I'd like a person to review it.",
                    "I compared my GroveDesk confirmation with the tracking row, "
                    "and they disagree. DEMO-L-309. Please involve a person.",
                ),
                (
                    "I'm following up on the tracking discrepancy and the human review.",
                    "Has someone reconciled the tracking row I asked about?",
                ),
            ),
            initial_record=(
                ("case_id", "lamp-tracking"),
                ("status", "open"),
                ("tracking", "mismatched"),
            ),
            guide=(
                "A tracking/confirmation conflict requires human review. The "
                "agent must not edit the tracking row; transfer the reference, "
                "issue, previous check, and proposed reconciliation step."
            ),
            agent_update=None,
            goal_update=("tracking", "reconciled"),
            human_next_step="reconcile_tracking",
            requires_human=True,
        ),
    }
)


def case_ids() -> tuple[str, ...]:
    return tuple(CASES)


def get_case(case_id: str) -> CaseSpec:
    try:
        return CASES[case_id]
    except KeyError as exc:
        raise ValueError(f"Unknown case {case_id!r}; choose from {case_ids()}") from exc
