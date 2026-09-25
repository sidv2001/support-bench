import json
from dataclasses import asdict
from typing import Iterable, TypedDict

from supportbench.baselines import BASELINE_VERSION, Policy, scripted_actions
from supportbench.cases import GENERATOR_VERSION
from supportbench.env import StepInfo, SupportEnv
from supportbench.models import (
    Action,
    AskCustomer,
    Handoff,
    HandoffSummary,
    InspectCase,
    Observation,
    Respond,
    SearchGuide,
    UpdateCase,
)
from supportbench.reward import SCORER_VERSION


TRACE_VERSION = "trace-v1"


class TraceStep(TypedDict):
    action: dict[str, object]
    observation: dict[str, object]
    reward: float
    terminated: bool
    truncated: bool
    info: StepInfo


class ResetRecord(TypedDict):
    observation: dict[str, object]
    info: StepInfo


class EpisodeTrace(TypedDict):
    trace_version: str
    generator_version: str
    scorer_version: str
    case_id: str
    seed: int
    policy: str
    policy_version: str
    reset: ResetRecord
    steps: list[TraceStep]


_ACTION_NAMES = {
    InspectCase: "inspect_case",
    SearchGuide: "search_guide",
    AskCustomer: "ask_customer",
    UpdateCase: "update_case",
    Handoff: "handoff",
    Respond: "respond",
}


def action_to_dict(action: Action) -> dict[str, object]:
    try:
        kind = _ACTION_NAMES[type(action)]
    except KeyError as exc:
        raise TypeError(f"Unsupported action: {type(action).__name__}") from exc
    return {"kind": kind, **asdict(action)}


def _strings(data: dict[str, object], *names: str) -> dict[str, str]:
    if set(data) != {"kind", *names}:
        raise ValueError(f"Invalid action fields; expected kind and {names}")
    values = {name: data[name] for name in names}
    if not all(isinstance(value, str) for value in values.values()):
        raise ValueError("Action fields must be strings")
    return {name: str(value) for name, value in values.items()}


def action_from_dict(data: object) -> Action:
    if not isinstance(data, dict):
        raise ValueError("Action must be a JSON object")
    kind = data.get("kind")
    if kind == "inspect_case":
        fields = _strings(data, "case_id")
        return InspectCase(fields["case_id"])
    if kind == "search_guide":
        fields = _strings(data, "product")
        return SearchGuide(fields["product"])
    if kind == "ask_customer":
        fields = _strings(data, "fact")
        return AskCustomer(fields["fact"])
    if kind == "update_case":
        fields = _strings(data, "case_id", "field", "value")
        return UpdateCase(fields["case_id"], fields["field"], fields["value"])
    if kind == "respond":
        fields = _strings(data, "statement")
        if fields["statement"] == "working":
            return Respond("working")
        if fields["statement"] == "resolved":
            return Respond("resolved")
        raise ValueError("Unknown response statement")
    if kind == "handoff":
        if set(data) != {"kind", "summary"} or not isinstance(data["summary"], dict):
            raise ValueError("Handoff must contain a summary object")
        summary_fields = _strings(
            {"kind": "summary", **data["summary"]},
            "case_id",
            "reference",
            "issue",
            "previous_effort",
            "next_step",
        )
        return Handoff(
            HandoffSummary(
                case_id=summary_fields["case_id"],
                reference=summary_fields["reference"],
                issue=summary_fields["issue"],
                previous_effort=summary_fields["previous_effort"],
                next_step=summary_fields["next_step"],
            )
        )
    raise ValueError(f"Unknown action kind: {kind!r}")


def _step_record(
    action: Action,
    result: tuple[Observation, float, bool, bool, StepInfo],
) -> TraceStep:
    observation, reward, terminated, truncated, info = result
    return {
        "action": action_to_dict(action),
        "observation": asdict(observation),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


def record_actions(
    actions: Iterable[Action],
    *,
    case_id: str,
    seed: int,
    policy: str = "custom",
    policy_version: str = "custom-v1",
) -> EpisodeTrace:
    if not policy or not policy_version:
        raise ValueError("Policy label and version must be non-empty")
    env = SupportEnv()
    observation, info = env.reset(case_id=case_id, seed=seed)
    steps: list[TraceStep] = []
    finished = False
    for action in actions:
        if finished:
            raise ValueError("Action supplied after the episode finished")
        result = env.step(action)
        steps.append(_step_record(action, result))
        finished = result[2] or result[3]
    if not finished:
        raise ValueError("Action sequence ended before the episode finished")
    return {
        "trace_version": TRACE_VERSION,
        "generator_version": GENERATOR_VERSION,
        "scorer_version": SCORER_VERSION,
        "case_id": case_id,
        "seed": seed,
        "policy": policy,
        "policy_version": policy_version,
        "reset": {"observation": asdict(observation), "info": info},
        "steps": steps,
    }


def record_script(policy: Policy, *, case_id: str, seed: int) -> EpisodeTrace:
    return record_actions(
        scripted_actions(policy, case_id),
        case_id=case_id,
        seed=seed,
        policy=policy,
        policy_version=BASELINE_VERSION,
    )


def replay_trace(trace: object) -> None:
    if not isinstance(trace, dict):
        raise ValueError("Trace must be a JSON object")
    expected_keys = {
        "trace_version",
        "generator_version",
        "scorer_version",
        "case_id",
        "seed",
        "policy",
        "policy_version",
        "reset",
        "steps",
    }
    if set(trace) != expected_keys:
        raise ValueError("Trace has missing or unknown top-level fields")
    if (
        trace["trace_version"] != TRACE_VERSION
        or trace["generator_version"] != GENERATOR_VERSION
        or trace["scorer_version"] != SCORER_VERSION
    ):
        raise ValueError("Trace format, generator, or scorer version differs")
    if (
        not isinstance(trace["case_id"], str)
        or isinstance(trace["seed"], bool)
        or not isinstance(trace["seed"], int)
        or not isinstance(trace["policy"], str)
        or not isinstance(trace["policy_version"], str)
        or not isinstance(trace["steps"], list)
    ):
        raise ValueError("Invalid trace case, seed, policy, version, or steps")
    actions: list[Action] = []
    for index, step in enumerate(trace["steps"], 1):
        if not isinstance(step, dict) or "action" not in step:
            raise ValueError(f"Missing action at step {index}")
        actions.append(action_from_dict(step["action"]))
    actual = record_actions(
        actions,
        case_id=trace["case_id"],
        seed=trace["seed"],
        policy=trace["policy"],
        policy_version=trace["policy_version"],
    )
    if actual["reset"] != trace["reset"]:
        raise ValueError("Trace diverged at reset")
    for index, (expected_step, actual_step) in enumerate(
        zip(trace["steps"], actual["steps"], strict=True), 1
    ):
        if expected_step != actual_step:
            raise ValueError(f"Trace diverged at step {index}")


def trace_json(trace: EpisodeTrace) -> str:
    return json.dumps(trace, indent=2, ensure_ascii=False) + "\n"
