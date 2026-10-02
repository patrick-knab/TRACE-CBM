import unittest
from types import SimpleNamespace

import numpy as np
import torch

from paper_expos.llm_intervention_policy import (
    activity_actions,
    assert_blinded_steering_prompt,
    conflict_free_order,
    reverse_transition_predecessors,
    signed_recommended_state,
    signed_transition_heuristic_order,
)


class FeedbackModel:
    def _activity_feedback_enabled(self):
        return True


class TargetSteeringRepairTests(unittest.TestCase):
    def setUp(self):
        self.workspace = SimpleNamespace(
            activity_names=["a", "b", "target", "d", "e", "f"],
            model=FeedbackModel(),
            standardized_splits={
                "train": {
                    "activity_labels": [
                        np.asarray([0, 2, 0, 2, 1, 2, 3, 2, 4, 2, 5, 1]),
                        np.asarray([1, 2, 1, 2, 0, 2, 3, 2, 4, 2, 5, 2]),
                    ],
                    "lengths": [12, 12],
                }
            },
        )

    def test_reverse_transition_keeps_target_then_ranks_unique_predecessors(self):
        rows = reverse_transition_predecessors(self.workspace, target_class=2, remaining_horizon=1, maximum_classes=5)
        self.assertEqual(rows[0]["class_idx"], 2)
        self.assertEqual(len({row["class_idx"] for row in rows}), 5)
        tail = [row["reverse_transition_probability"] for row in rows[1:]]
        self.assertEqual(tail, sorted(tail, reverse=True))
        self.assertTrue(all(row["reverse_transition_remaining_horizon"] == 1 for row in rows))

    def test_signed_head_weight_recommends_intervention_direction(self):
        self.assertEqual(signed_recommended_state(0.25), "on")
        self.assertEqual(signed_recommended_state(0.0), "on")
        self.assertEqual(signed_recommended_state(-0.25), "off")

    def test_signed_transition_order_is_deterministic_and_conflict_free(self):
        actions = [
            {
                "action_id": "concept_on",
                "action_type": "concept",
                "conflict_key": "concept:x",
                "target_head_weight": 2.0,
                "matches_recommended_state": True,
            },
            {
                "action_id": "concept_off",
                "action_type": "concept",
                "conflict_key": "concept:x",
                "target_head_weight": 2.0,
                "matches_recommended_state": False,
            },
            {
                "action_id": "activity_0",
                "action_type": "activity",
                "conflict_key": "activity:0",
                "reverse_transition_probability": 0.9,
            },
            {
                "action_id": "activity_0_alt",
                "action_type": "activity",
                "conflict_key": "activity:0",
                "reverse_transition_probability": 0.8,
            },
            {
                "action_id": "activity_1",
                "action_type": "activity",
                "conflict_key": "activity:1",
                "reverse_transition_probability": 0.7,
            },
        ]
        first = signed_transition_heuristic_order(actions)
        second = signed_transition_heuristic_order(actions)
        self.assertEqual(first, second)
        self.assertEqual(first[0], "concept_on")
        self.assertEqual(len(first), len(set(actions_by_conflict(actions, first))))

    def test_legacy_activity_candidates_are_unchanged(self):
        baseline = {
            "effective_activity_probs_by_step": {
                step: torch.tensor([[[0.6, 0.1, 0.2, 0.05, 0.03, 0.02]]])
                for step in range(3)
            }
        }
        actions = activity_actions(
            self.workspace,
            baseline,
            "h3",
            maximum_classes=3,
            target_class=1,
            candidate_rule="legacy",
        )
        self.assertEqual(
            [action["action_id"] for action in actions],
            [
                "a0_c1", "a0_c0", "a0_c2",
                "a1_c1", "a1_c0", "a1_c2",
                "a2_c1", "a2_c0", "a2_c2",
            ],
        )
        self.assertTrue(all("reverse_transition_probability" not in action for action in actions))

    def test_conflict_free_order_and_prompt_leak_guard(self):
        actions = [
            {"action_id": "a", "conflict_key": "same"},
            {"action_id": "b", "conflict_key": "same"},
            {"action_id": "c", "conflict_key": "other"},
        ]
        self.assertEqual(conflict_free_order(actions, ["a", "b", "c"]), ["a", "c"])
        assert_blinded_steering_prompt("blinded prompt", actions)
        with self.assertRaises(RuntimeError):
            assert_blinded_steering_prompt("held_out_h3_label", actions)
        with self.assertRaises(RuntimeError):
            assert_blinded_steering_prompt(
                "blinded prompt",
                [{"action_id": "a", "conflict_key": "a", "oracle_single_action": {}}],
            )


def actions_by_conflict(actions, ordered):
    by_id = {action["action_id"]: action for action in actions}
    return [by_id[action_id]["conflict_key"] for action_id in ordered]


if __name__ == "__main__":
    unittest.main()
