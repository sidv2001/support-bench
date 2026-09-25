import json
import unittest

from supportbench import Respond, case_ids, record_actions, record_script, replay_trace
from supportbench.baselines import BASELINE_VERSION
from supportbench.replay import action_from_dict


class ReplayTests(unittest.TestCase):
    def test_all_scripts_replay_through_json(self) -> None:
        for case_id in case_ids():
            for policy in ("context", "memoryless", "always_handoff"):
                with self.subTest(case_id=case_id, policy=policy):
                    trace = record_script(policy, case_id=case_id, seed=7)
                    self.assertEqual(trace["policy_version"], BASELINE_VERSION)
                    replay_trace(json.loads(json.dumps(trace)))
                    self.assertTrue(trace["steps"][-1]["terminated"])
                    self.assertEqual(trace["steps"][-1]["info"]["terminal_reason"], "contacts_complete")

    def test_same_seed_and_actions_produce_identical_trajectory(self) -> None:
        first = record_script("context", case_id="fan-setup", seed=7)
        second = record_script("context", case_id="fan-setup", seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["reset"]["observation"], second["reset"]["observation"])
        variations = {
            record_script("context", case_id="fan-setup", seed=seed)["reset"][
                "observation"
            ]["message"]
            for seed in range(8)
        }
        self.assertGreater(len(variations), 1)

    def test_custom_failure_is_replayable_and_a_tampered_step_is_not(self) -> None:
        trace = record_actions((Respond("resolved"),), case_id="lamp-ring", seed=7)
        self.assertEqual(trace["steps"][0]["reward"], -1.0)
        replay_trace(json.loads(json.dumps(trace)))
        altered = json.loads(json.dumps(trace))
        altered["steps"][0]["info"]["reward_vector"]["verified_resolution"] = 1
        with self.assertRaisesRegex(ValueError, "step 1"):
            replay_trace(altered)

    def test_reset_and_versions_are_checked(self) -> None:
        trace = record_script("context", case_id="lamp-ring", seed=7)
        changed_seed = json.loads(json.dumps(trace))
        changed_seed["seed"] = 19
        with self.assertRaisesRegex(ValueError, "reset"):
            replay_trace(changed_seed)
        changed_version = json.loads(json.dumps(trace))
        changed_version["scorer_version"] = "different"
        with self.assertRaisesRegex(ValueError, "version"):
            replay_trace(changed_version)

    def test_invalid_or_incomplete_action_trace_is_rejected(self) -> None:
        trace = record_script("context", case_id="lamp-ring", seed=7)
        with self.assertRaises(ValueError):
            record_actions((Respond("working"),), case_id="lamp-ring", seed=7)
        with self.assertRaises(ValueError):
            action_from_dict({"kind": "update_case", "case_id": "lamp-ring"})
        with self.assertRaises(ValueError):
            action_from_dict({"kind": "shell", "command": "anything"})
        extra = json.loads(json.dumps(trace))
        extra["steps"].append(extra["steps"][-1])
        with self.assertRaisesRegex(ValueError, "after the episode"):
            replay_trace(extra)


if __name__ == "__main__":
    unittest.main()
