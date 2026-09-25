import random
from dataclasses import asdict, dataclass, field
from typing import Literal, TypedDict

from supportbench.cases import GENERATOR_VERSION, CaseSpec, get_case
from supportbench.models import (
    Action,
    AskCustomer,
    Handoff,
    InspectCase,
    Observation,
    Respond,
    SearchGuide,
    UpdateCase,
)
from supportbench.reward import (
    SCORER_VERSION,
    reward_vector,
    terminal_reward,
    verified_resolution,
)


class StepInfo(TypedDict):
    generator_version: str
    scorer_version: str
    reward_vector: dict[str, int | float]
    metrics: dict[str, int]
    terminal_reason: str | None
    failure: str | None


@dataclass(slots=True)
class _Customer:
    known_facts: dict[str, str]
    patience: float
    questions: dict[str, int] = field(default_factory=dict)
    satisfaction: float = 0.5
    frustration: float = 0.0


@dataclass(slots=True)
class _Episode:
    case: CaseSpec
    record: dict[str, str]
    customer: _Customer
    prompt_variant: int
    contact_index: int = 0
    inspected: bool = False
    searched: bool = False
    turns: int = 0
    elapsed_units: int = 0
    cost_units: int = 0
    repeated_requests: int = 0
    handoffs: int = 0
    handoff_quality_total: int = 0
    history: list[str] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    failure: str | None = None


class SupportEnv:
    """One independent case per episode; contacts share state until reset."""

    def __init__(self) -> None:
        self._episode: _Episode | None = None

    def reset(self, *, case_id: str, seed: int) -> tuple[Observation, StepInfo]:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        case = get_case(case_id)
        rng = random.Random(seed)
        variant = rng.randrange(2)
        patience = (0.85, 1.0, 1.15)[rng.randrange(3)]
        self._episode = _Episode(
            case=case,
            record=dict(case.initial_record),
            customer=_Customer(
                known_facts={
                    "reference": case.reference,
                    "previous_effort": case.previous_effort,
                },
                patience=patience,
            ),
            prompt_variant=variant,
        )
        return self._observation(
            case.contact_prompts[0][variant], source="customer"
        ), self._info()

    def record_snapshot(self) -> dict[str, str]:
        return dict(self._current().record)

    def step(
        self, action: Action
    ) -> tuple[Observation, float, bool, bool, StepInfo]:
        episode = self._current()
        if episode.terminated or episode.truncated:
            raise RuntimeError("Episode finished; call reset before taking another action")
        if not isinstance(
            action, (InspectCase, SearchGuide, AskCustomer, UpdateCase, Handoff, Respond)
        ):
            raise TypeError(f"Unsupported action: {type(action).__name__}")

        episode.turns += 1
        episode.elapsed_units += 1
        case = episode.case

        if isinstance(action, InspectCase):
            if action.case_id != case.case_id:
                return self._result(self._deny("Inspection outside the active case"))
            episode.inspected = True
            episode.cost_units += 1
            observation = self._observation(
                "Case record retrieved.",
                {
                    "record": dict(episode.record),
                    "product": case.product,
                    "issue": case.issue,
                    "known_facts": dict(episode.customer.known_facts),
                    "history": list(episode.history),
                },
            )
        elif isinstance(action, SearchGuide):
            if action.product != case.product:
                return self._result(self._deny("Guide search outside the active product"))
            episode.searched = True
            episode.cost_units += 1
            observation = self._observation(
                "Fictional product guide retrieved.",
                {
                    "policy": case.guide,
                    "agent_update": (
                        {"field": case.agent_update[0], "value": case.agent_update[1]}
                        if case.agent_update
                        else None
                    ),
                    "human_next_step": case.human_next_step,
                },
            )
        elif isinstance(action, AskCustomer):
            facts = {
                "reference": case.reference,
                "previous_effort": case.previous_effort,
                "preference": case.preference,
            }
            if action.fact not in facts:
                return self._result(self._deny("Question outside the defined customer facts"))
            already_shared = action.fact in episode.customer.known_facts
            if already_shared:
                episode.repeated_requests += 1
                self._adjust(satisfaction=-0.08, frustration=0.18 / episode.customer.patience)
            answer = facts[action.fact]
            episode.customer.known_facts[action.fact] = answer
            episode.customer.questions[action.fact] = (
                episode.customer.questions.get(action.fact, 0) + 1
            )
            episode.cost_units += 1
            episode.history.append(
                f"Contact {episode.contact_index + 1}: asked {action.fact}"
                + (" again" if already_shared else "")
            )
            prefix = "I already shared" if already_shared else "My answer is"
            observation = self._observation(
                f"{prefix} {action.fact}: {answer}.",
                {"fact": action.fact, "answer": answer, "already_shared": already_shared},
                source="customer",
            )
        elif isinstance(action, UpdateCase):
            if action.case_id != case.case_id:
                return self._result(self._deny("Write outside the active case"))
            if not episode.inspected or not episode.searched:
                return self._result(self._deny("Inspect the case and guide before a write"))
            if (action.field, action.value) == case.agent_update:
                episode.record[action.field] = action.value
            elif action.field == "status" and action.value == "closed":
                if not verified_resolution(case, episode.record):
                    return self._result(self._deny("Cannot close a case without a verified outcome"))
                episode.record["status"] = "closed"
            elif action.field == "status" and action.value == "in_progress":
                episode.record["status"] = "in_progress"
            else:
                return self._result(self._deny("Unauthorized case change"))
            episode.cost_units += 2
            episode.history.append(
                f"Contact {episode.contact_index + 1}: changed {action.field} to {action.value}"
            )
            observation = self._observation(
                "Authorized record change saved.", {"record": dict(episode.record)}
            )
        elif isinstance(action, Handoff):
            summary = action.summary
            if summary.case_id != case.case_id:
                return self._result(self._deny("Handoff outside the active case"))
            complete = (
                summary.reference == case.reference
                and summary.issue == case.issue
                and summary.previous_effort == case.previous_effort
                and summary.next_step == case.human_next_step
            )
            already_resolved = verified_resolution(case, episode.record)
            appropriate = case.requires_human and not already_resolved
            episode.handoffs += 1
            episode.cost_units += 6
            episode.elapsed_units += 3
            if complete:
                if not already_resolved:
                    field_name, value = case.goal_update
                    episode.record[field_name] = value
                    outcome = "Human queue completed the verified service action."
                else:
                    outcome = "Human queue found the service action already complete."
                if appropriate:
                    episode.handoff_quality_total += 1
                    self._adjust(satisfaction=0.12)
                else:
                    episode.handoff_quality_total -= 1
                    self._adjust(satisfaction=-0.08, frustration=0.08)
            else:
                episode.handoff_quality_total -= 1
                episode.repeated_requests += 1
                self._adjust(satisfaction=-0.15, frustration=0.22 / episode.customer.patience)
                outcome = "Human queue returned the case for missing or incorrect context."
            episode.history.append(f"Contact {episode.contact_index + 1}: {outcome}")
            observation = self._end_contact(
                {"summary_complete": complete, "human_outcome": outcome}
            )
        else:
            if action.statement not in ("working", "resolved"):
                return self._result(self._deny("Unknown response statement"))
            if action.statement == "resolved":
                if not verified_resolution(case, episode.record):
                    return self._result(self._deny("Claimed resolution without a verified outcome"))
                self._adjust(satisfaction=0.15)
            else:
                self._adjust(satisfaction=0.02)
            episode.history.append(
                f"Contact {episode.contact_index + 1}: responded {action.statement}"
            )
            observation = self._end_contact({"response": action.statement})

        return self._result(observation)

    def _current(self) -> _Episode:
        if self._episode is None:
            raise RuntimeError("Call reset before using the environment")
        return self._episode

    def _observation(
        self,
        message: str,
        tool_result: dict[str, object] | None = None,
        *,
        source: Literal["customer", "tool", "system"] = "tool",
    ) -> Observation:
        episode = self._current()
        return Observation(
            case_id=episode.case.case_id,
            contact=episode.contact_index + 1,
            total_contacts=len(episode.case.contact_prompts),
            source=source,
            message=message,
            tool_result=tool_result,
        )

    def _adjust(self, *, satisfaction: float = 0.0, frustration: float = 0.0) -> None:
        customer = self._current().customer
        customer.satisfaction = round(
            max(0.0, min(1.0, customer.satisfaction + satisfaction)), 3
        )
        customer.frustration = round(
            max(0.0, min(1.0, customer.frustration + frustration)), 3
        )

    def _deny(self, reason: str) -> Observation:
        episode = self._current()
        episode.failure = reason
        episode.terminated = True
        self._adjust(satisfaction=-0.25, frustration=0.3)
        return self._observation(f"Action denied: {reason}", source="system")

    def _end_contact(self, result: dict[str, object]) -> Observation:
        episode = self._current()
        if episode.contact_index + 1 == len(episode.case.contact_prompts):
            episode.terminated = True
            return self._observation(
                "All scheduled contacts are complete.", result, source="system"
            )
        episode.contact_index += 1
        episode.elapsed_units += 2
        return self._observation(
            episode.case.contact_prompts[episode.contact_index][episode.prompt_variant],
            result,
            source="customer",
        )

    def _info(self) -> StepInfo:
        episode = self._current()
        vector = reward_vector(
            episode.case,
            episode.record,
            satisfaction=episode.customer.satisfaction,
            frustration=episode.customer.frustration,
            repeated_requests=episode.repeated_requests,
            elapsed_units=episode.elapsed_units,
            cost_units=episode.cost_units,
            handoffs=episode.handoffs,
            handoff_quality_total=episode.handoff_quality_total,
        )
        if episode.failure:
            reason = "policy_violation"
        elif episode.truncated:
            reason = "turn_limit"
        elif episode.terminated:
            reason = "contacts_complete"
        else:
            reason = None
        return {
            "generator_version": GENERATOR_VERSION,
            "scorer_version": SCORER_VERSION,
            "reward_vector": asdict(vector),
            "metrics": {
                "turns": episode.turns,
                "elapsed_units": episode.elapsed_units,
                "cost_units": episode.cost_units,
                "repeated_requests": episode.repeated_requests,
                "handoffs": episode.handoffs,
            },
            "terminal_reason": reason,
            "failure": episode.failure,
        }

    def _result(
        self, observation: Observation
    ) -> tuple[Observation, float, bool, bool, StepInfo]:
        episode = self._current()
        if not episode.terminated and episode.turns >= len(episode.case.contact_prompts) * 6:
            episode.truncated = True
        info = self._info()
        if episode.terminated or episode.truncated:
            vector = reward_vector(
                episode.case,
                episode.record,
                satisfaction=episode.customer.satisfaction,
                frustration=episode.customer.frustration,
                repeated_requests=episode.repeated_requests,
                elapsed_units=episode.elapsed_units,
                cost_units=episode.cost_units,
                handoffs=episode.handoffs,
                handoff_quality_total=episode.handoff_quality_total,
            )
            reward = terminal_reward(
                vector, policy_violation=episode.failure is not None, truncated=episode.truncated
            )
        else:
            reward = 0.0
        return observation, reward, episode.terminated, episode.truncated, info
