import unittest

from supportbench import (
    AskCustomer,
    Handoff,
    HandoffSummary,
    InspectCase,
    Respond,
    SearchGuide,
    SupportEnv,
    UpdateCase,
    case_ids,
    record_script,
)
from supportbench.cases import get_case


def ready_for_update(case_id: str) -> SupportEnv:
    env = SupportEnv()
    env.reset(case_id=case_id, seed=7)
    env.step(InspectCase(case_id))
    env.step(SearchGuide(get_case(case_id).product))
    return env


def summary_for(case_id: str, *, next_step: str | None = None) -> HandoffSummary:
    case = get_case(case_id)
    return HandoffSummary(
        case_id=case_id,
        reference=case.reference,
        issue=case.issue,
        previous_effort=case.previous_effort,
        next_step=next_step if next_step is not None else case.human_next_step,
    )


class EnvironmentTests(unittest.TestCase):
    def test_cases_have_two_products_and_two_or_three_contacts(self) -> None:
        cases = [get_case(case_id) for case_id in case_ids()]
        self.assertEqual({len(case.contact_prompts) for case in cases}, {2, 3})
        self.assertEqual(len({case.product for case in cases}), 2)

    def test_world_change_not_claim_or_status_verifies_resolution(self) -> None:
        env = ready_for_update("lamp-ring")
        _, _, terminated, _, info = env.step(
            UpdateCase("lamp-ring", "fulfillment", "part_queued")
        )
        self.assertFalse(terminated)
        self.assertEqual(info["reward_vector"]["verified_resolution"], 1)
        self.assertEqual(env.record_snapshot()["status"], "open")
        env.step(Respond("working"))
        observation, reward, terminated, truncated, info = env.step(Respond("resolved"))
        self.assertEqual(observation.contact, 2)
        self.assertEqual(observation.source, "system")
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertGreater(reward, 0)
        self.assertEqual(info["terminal_reason"], "contacts_complete")

    def test_accurate_closure_is_an_alternative_valid_path(self) -> None:
        env = ready_for_update("fan-setup")
        env.step(UpdateCase("fan-setup", "guide_delivery", "full_guide_sent"))
        env.step(UpdateCase("fan-setup", "status", "closed"))
        self.assertEqual(env.record_snapshot()["status"], "closed")
        env.step(Respond("working"))
        env.step(Respond("working"))
        _, _, terminated, _, info = env.step(Respond("resolved"))
        self.assertTrue(terminated)
        self.assertEqual(info["reward_vector"]["verified_resolution"], 1)

    def test_false_resolution_and_false_closure_fail_closed(self) -> None:
        env = SupportEnv()
        env.reset(case_id="lamp-ring", seed=7)
        _, reward, terminated, _, info = env.step(Respond("resolved"))
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)
        self.assertEqual(info["reward_vector"]["verified_resolution"], 0)
        self.assertEqual(env.record_snapshot()["fulfillment"], "not_queued")
        with self.assertRaises(RuntimeError):
            env.step(Respond("working"))

        env = ready_for_update("lamp-ring")
        _, reward, terminated, _, info = env.step(UpdateCase("lamp-ring", "status", "closed"))
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)
        self.assertEqual(info["terminal_reason"], "policy_violation")
        self.assertEqual(env.record_snapshot()["status"], "open")

    def test_policy_violation_overrides_an_earlier_verified_outcome(self) -> None:
        env = ready_for_update("lamp-ring")
        env.step(UpdateCase("lamp-ring", "fulfillment", "part_queued"))
        before = env.record_snapshot()
        _, reward, terminated, _, info = env.step(
            UpdateCase("fan-setup", "fulfillment", "part_queued")
        )
        self.assertTrue(terminated)
        self.assertEqual(info["reward_vector"]["verified_resolution"], 1)
        self.assertEqual(reward, -1.0)
        self.assertEqual(env.record_snapshot(), before)

    def test_unauthorized_writes_do_not_change_records(self) -> None:
        for case_id, field, value in (
            ("lamp-ring", "tracking", "reconciled"),
            ("lamp-tracking", "tracking", "reconciled"),
            ("fan-setup", "guide_delivery", "not_sent"),
        ):
            with self.subTest(case_id=case_id, field=field):
                env = ready_for_update(case_id)
                before = env.record_snapshot()
                _, reward, terminated, _, info = env.step(
                    UpdateCase(case_id, field, value)
                )
                self.assertTrue(terminated)
                self.assertEqual(reward, -1.0)
                self.assertEqual(info["reward_vector"]["verified_resolution"], 0)
                self.assertEqual(env.record_snapshot(), before)

        env = ready_for_update("lamp-ring")
        before = env.record_snapshot()
        _, reward, terminated, _, _ = env.step(
            UpdateCase("fan-setup", "fulfillment", "part_queued")
        )
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)
        self.assertEqual(env.record_snapshot(), before)

        env = SupportEnv()
        env.reset(case_id="lamp-ring", seed=7)
        _, _, terminated, _, _ = env.step(
            UpdateCase("lamp-ring", "fulfillment", "part_queued")
        )
        self.assertTrue(terminated)
        self.assertEqual(env.record_snapshot()["fulfillment"], "not_queued")

    def test_read_question_and_handoff_scope_is_enforced(self) -> None:
        denied_actions = (
            InspectCase("fan-setup"),
            SearchGuide("PebbleFan"),
            AskCustomer("unlisted_fact"),
            Handoff(summary_for("fan-setup")),
        )
        for action in denied_actions:
            with self.subTest(action=action):
                env = SupportEnv()
                env.reset(case_id="lamp-ring", seed=7)
                _, reward, terminated, _, info = env.step(action)
                self.assertTrue(terminated)
                self.assertEqual(reward, -1.0)
                self.assertEqual(info["terminal_reason"], "policy_violation")
                self.assertEqual(env.record_snapshot()["fulfillment"], "not_queued")

    def test_customer_remembers_facts_across_contacts(self) -> None:
        env = SupportEnv()
        opener, _ = env.reset(case_id="lamp-ring", seed=7)
        self.assertEqual(opener.source, "customer")
        self.assertIn("DEMO-L-041", opener.message)
        _, _, _, _, info = env.step(AskCustomer("preference"))
        self.assertEqual(info["metrics"]["repeated_requests"], 0)
        env.step(Respond("working"))
        observation, _, _, _, _ = env.step(InspectCase("lamp-ring"))
        self.assertEqual(observation.contact, 2)
        self.assertEqual(observation.source, "tool")
        self.assertEqual(observation.tool_result["known_facts"]["preference"], "brief_updates")
        self.assertIn("asked preference", observation.tool_result["history"][0])
        answer, _, _, _, info = env.step(AskCustomer("preference"))
        self.assertTrue(answer.tool_result["already_shared"])
        self.assertEqual(info["metrics"]["repeated_requests"], 1)
        self.assertGreater(info["reward_vector"]["frustration"], 0)

    def test_reasking_initial_fact_across_contacts_adds_burden(self) -> None:
        env = SupportEnv()
        env.reset(case_id="lamp-ring", seed=7)
        first, _, _, _, info = env.step(AskCustomer("reference"))
        self.assertTrue(first.tool_result["already_shared"])
        self.assertEqual(info["metrics"]["repeated_requests"], 1)
        env.step(Respond("working"))
        _, _, _, _, info = env.step(AskCustomer("reference"))
        self.assertEqual(info["metrics"]["repeated_requests"], 2)
        self.assertEqual(info["reward_vector"]["repeated_effort"], 1.0)

    def test_handoff_requires_useful_context_and_human_changes_world(self) -> None:
        env = SupportEnv()
        env.reset(case_id="lamp-tracking", seed=7)
        observation, _, _, _, info = env.step(Handoff(summary_for("lamp-tracking")))
        self.assertEqual(observation.contact, 2)
        self.assertEqual(observation.source, "customer")
        self.assertTrue(observation.tool_result["summary_complete"])
        self.assertEqual(env.record_snapshot()["tracking"], "reconciled")
        self.assertEqual(info["reward_vector"]["verified_resolution"], 1)
        self.assertEqual(info["reward_vector"]["handoff_quality"], 1.0)
        _, reward, terminated, _, _ = env.step(Respond("resolved"))
        self.assertTrue(terminated)
        self.assertGreater(reward, 0)

        env = SupportEnv()
        env.reset(case_id="lamp-tracking", seed=7)
        observation, _, _, _, info = env.step(
            Handoff(summary_for("lamp-tracking", next_step="unknown"))
        )
        self.assertFalse(observation.tool_result["summary_complete"])
        self.assertEqual(env.record_snapshot()["tracking"], "mismatched")
        self.assertEqual(info["reward_vector"]["handoff_quality"], -1.0)
        self.assertEqual(info["metrics"]["repeated_requests"], 1)
        _, _, terminated, _, info = env.step(Respond("working"))
        self.assertTrue(terminated)
        self.assertEqual(info["reward_vector"]["verified_resolution"], 0)

    def test_unnecessary_handoff_can_fix_but_is_not_free(self) -> None:
        env = SupportEnv()
        env.reset(case_id="lamp-ring", seed=7)
        _, _, _, _, info = env.step(Handoff(summary_for("lamp-ring")))
        self.assertEqual(info["reward_vector"]["verified_resolution"], 1)
        self.assertEqual(info["reward_vector"]["handoff_quality"], -1.0)
        self.assertGreater(info["metrics"]["cost_units"], 0)
        repeat, _, _, _, _ = env.step(Handoff(summary_for("lamp-ring")))
        self.assertIn("already complete", repeat.tool_result["human_outcome"])

        for case_id in case_ids():
            with self.subTest(case_id=case_id):
                context = record_script("context", case_id=case_id, seed=7)["steps"][-1]
                handoff = record_script("always_handoff", case_id=case_id, seed=7)[
                    "steps"
                ][-1]
                self.assertGreater(context["reward"], handoff["reward"])
                self.assertEqual(handoff["info"]["reward_vector"]["verified_resolution"], 1)

    def test_memoryless_policy_has_higher_repeated_effort(self) -> None:
        for case_id in case_ids():
            with self.subTest(case_id=case_id):
                context = record_script("context", case_id=case_id, seed=7)["steps"][-1]
                memoryless = record_script("memoryless", case_id=case_id, seed=7)[
                    "steps"
                ][-1]
                self.assertEqual(context["info"]["metrics"]["repeated_requests"], 0)
                self.assertGreater(memoryless["info"]["metrics"]["repeated_requests"], 0)
                self.assertGreater(context["reward"], memoryless["reward"])

    def test_reset_and_parallel_episodes_are_isolated(self) -> None:
        first = SupportEnv()
        second = SupportEnv()
        first.reset(case_id="lamp-ring", seed=7)
        second.reset(case_id="lamp-ring", seed=7)
        first.step(AskCustomer("preference"))
        first.step(InspectCase("lamp-ring"))
        first.step(SearchGuide("GroveDesk lamp"))
        first.step(UpdateCase("lamp-ring", "fulfillment", "part_queued"))
        self.assertEqual(second.record_snapshot()["fulfillment"], "not_queued")
        second.step(InspectCase("lamp-ring"))
        self.assertNotIn(
            "preference", second.step(InspectCase("lamp-ring"))[0].tool_result["known_facts"]
        )

        first.reset(case_id="fan-setup", seed=7)
        self.assertEqual(first.record_snapshot()["case_id"], "fan-setup")
        observation, _ = first.reset(case_id="lamp-ring", seed=7)
        self.assertEqual(first.record_snapshot()["fulfillment"], "not_queued")
        self.assertEqual(observation.contact, 1)
        self.assertEqual(first.step(InspectCase("lamp-ring"))[0].tool_result["history"], [])

    def test_snapshots_are_copies(self) -> None:
        env = SupportEnv()
        env.reset(case_id="lamp-ring", seed=7)
        snapshot = env.record_snapshot()
        snapshot["fulfillment"] = "part_queued"
        self.assertEqual(env.record_snapshot()["fulfillment"], "not_queued")
        self.assertEqual(env.step(InspectCase("lamp-ring"))[4]["reward_vector"]["verified_resolution"], 0)

    def test_turn_budget_truncates_without_faking_completion(self) -> None:
        env = SupportEnv()
        env.reset(case_id="lamp-ring", seed=7)
        for _ in range(11):
            _, reward, terminated, truncated, _ = env.step(InspectCase("lamp-ring"))
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        _, reward, terminated, truncated, info = env.step(InspectCase("lamp-ring"))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, -0.5)
        self.assertEqual(info["terminal_reason"], "turn_limit")
        with self.assertRaises(RuntimeError):
            env.step(InspectCase("lamp-ring"))

    def test_reward_vector_has_documented_shape(self) -> None:
        env = SupportEnv()
        _, info = env.reset(case_id="lamp-ring", seed=7)
        self.assertEqual(
            set(info["reward_vector"]),
            {
                "verified_resolution",
                "satisfaction_change",
                "frustration",
                "repeated_effort",
                "elapsed_time",
                "cost",
                "handoff_quality",
            },
        )
        self.assertEqual(info["reward_vector"]["verified_resolution"], 0)
        self.assertEqual(info["reward_vector"]["satisfaction_change"], 0.0)

    def test_invalid_reset_does_not_destroy_current_episode(self) -> None:
        env = SupportEnv()
        env.reset(case_id="lamp-ring", seed=7)
        with self.assertRaises(ValueError):
            env.reset(case_id="not-a-case", seed=7)
        with self.assertRaises(ValueError):
            env.reset(case_id="lamp-ring", seed=True)
        self.assertEqual(env.record_snapshot()["case_id"], "lamp-ring")
        with self.assertRaises(TypeError):
            env.step("inspect_case")
        self.assertEqual(env.step(InspectCase("lamp-ring"))[4]["metrics"]["turns"], 1)


if __name__ == "__main__":
    unittest.main()
