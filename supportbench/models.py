from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class InspectCase:
    case_id: str


@dataclass(frozen=True, slots=True)
class SearchGuide:
    product: str


@dataclass(frozen=True, slots=True)
class AskCustomer:
    fact: str


@dataclass(frozen=True, slots=True)
class UpdateCase:
    case_id: str
    field: str
    value: str


@dataclass(frozen=True, slots=True)
class HandoffSummary:
    case_id: str
    reference: str
    issue: str
    previous_effort: str
    next_step: str


@dataclass(frozen=True, slots=True)
class Handoff:
    summary: HandoffSummary


@dataclass(frozen=True, slots=True)
class Respond:
    statement: Literal["working", "resolved"]


type Action = InspectCase | SearchGuide | AskCustomer | UpdateCase | Handoff | Respond


@dataclass(frozen=True, slots=True)
class Observation:
    case_id: str
    contact: int
    total_contacts: int
    source: Literal["customer", "tool", "system"]
    message: str
    tool_result: dict[str, object] | None = None
