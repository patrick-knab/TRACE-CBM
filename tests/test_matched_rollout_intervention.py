import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from paper_expos.matched_rollout_intervention import (
    ARMS,
    aggregate,
    fit_shared_external_probe,
    horizon_logits,
    method_family,
    parse_budgets,
    sequence_payload,
    shared_external_probe_oracle_order,
    sparse_forward,
)
from utils.model import LinearSparseDynamicsSharedHeadClassifier, _build_sequence_model


class MatchedRolloutInterventionTests(unittest.TestCase):
    def test_legacy_trace_method_is_the_graph_method_family(self):
        self.assertEqual(method_family("concept_forecast_cbm"), "graph_cbm")
        self.assertEqual(method_family("graph_cbm"), "graph_cbm")
        self.assertEqual(method_family("linear_sparse_dynamics_shared_head"), "linear_sparse_dynamics_shared_head")

    def test_capacity_dense_reuses_the_forecast_transition(self):
        common = {
            "st_graph_layers": 1,
            "st_task_graph_layers": 1,
            "st_enable_cross_temporal": True,
            "st_state_activation": "bounded_logit",
            "st_prediction_transform": "logit",
            "motif_z_attention_layers": 0,
            "st_activity_feedback_mode": "sparse_label_to_concept",
            "st_activity_feedback_history_steps": 1,
        }
        trace = _build_sequence_model(
            method="graph_cbm", num_concepts=8, num_activities=3, horizon=3,
            history_length=5, max_sequence_length=32, model_hparams=common,
        )
        dense = _build_sequence_model(
            method="graph_cbm", num_concepts=8, num_activities=3, horizon=3,
            history_length=5, max_sequence_length=32,
            model_hparams={
                **common,
                "st_observed_refiner_mode": "dense",
                "st_forecast_rollout_mode": "controlled_dense",
                "st_controlled_dense_reuse_forecast_layer": True,
            },
        )
        self.assertIs(dense.controlled_rollout_layers, dense.forecast_graph_layers)
        self.assertEqual(
            sum(parameter.numel() for parameter in trace.parameters()),
            sum(parameter.numel() for parameter in dense.parameters()),
        )

    def test_budget_validation_and_h1_payload(self):
        self.assertEqual(parse_budgets("1,2,3,4,5"), (1, 2, 3, 4, 5))
        with self.assertRaises(ValueError):
            parse_budgets("2,1")
        payload = sequence_payload([4, 2, 1], [0.1, 0.2, 0.3, 0.4, 0.5], 2)
        self.assertEqual([item["concept_idx"] for item in payload["items"]], [4, 2])
        self.assertTrue(all(item["rollout_step"] == 1 for item in payload["items"]))

    def test_shared_external_probe_order_uses_common_data_not_a_model(self):
        workspace = SimpleNamespace(
            activity_names=["a", "b"],
            standardized_splits={
                "train": {
                    "concepts_std": torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]).numpy(),
                    "lengths": torch.tensor([4]).numpy(),
                },
                "test": {
                    "concepts_std": torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]).numpy(),
                },
            },
            preprocessed_data={"train": {"activity_labels": torch.tensor([[0, 0, 0, 1]]).numpy()}},
        )
        coefficients, metadata = fit_shared_external_probe(workspace)
        self.assertEqual(metadata["n_train_examples"], 1)
        self.assertEqual(coefficients.shape, (3, 2))
        order = shared_external_probe_oracle_order(
            workspace,
            {"video_index": 0, "timestep": 0, "future_labels": {3: 0}},
            # Feature 0 supports class a over b and feature 1 opposes it.
            torch.tensor([[2.0, -1.0], [-1.0, 2.0], [0.0, 0.0]]).numpy(),
            seed=7,
        )
        self.assertEqual(order[0], 0)

    def test_sparse_clamp_is_applied_after_the_h1_transition(self):
        model = LinearSparseDynamicsSharedHeadClassifier(num_concepts=2, num_classes=2, horizon=3, top_k=2)
        with torch.no_grad():
            model.concept_dynamics.weight.copy_(torch.eye(2))
            model.concept_dynamics.bias.fill_(1.0)
            model.activity_head.weight.zero_()
            model.activity_head.bias.zero_()
        workspace = SimpleNamespace(model={"forecast_model": model}, device=torch.device("cpu"))
        instance = {"concepts": torch.zeros(1, 2).numpy()}
        outputs = sparse_forward(workspace, instance, selected=[0], target=[5.0, 0.0])
        self.assertEqual(outputs["predicted_concepts_by_step"][1][0].tolist(), [5.0, 1.0])
        self.assertEqual(outputs["predicted_concepts_by_step"][2][0].tolist(), [6.0, 2.0])
        self.assertEqual(tuple(horizon_logits(workspace, outputs, 2).shape), (2,))

    def test_aggregate_reports_threshold_and_non_attainment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = [
                "dataset", "seed", "arm", "arm_label", "budget", "horizon", "n_cases",
                "baseline_wrong_n", "baseline_correct_n", "true_probability_delta_mean",
                "future_concept_l1_mean", "future_concept_error_reduction_mean",
                "future_concept_nonselected_l1_mean", "future_concept_nontrivial_fraction_mean",
                "future_concept_selected_displacement_share_mean", "label_flip_rate", "wrong_to_correct_rate",
                "correct_to_wrong_rate", "wrong_to_correct_by_budget_rate", "correct_to_wrong_by_budget_rate",
            ]
            for dataset in ("barista", "breakfast_s1", "mpii_attr"):
                for seed in (42, 43, 44):
                    path = root / f"{dataset}_seed{seed}_summary.csv"
                    with path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        writer.writeheader()
                        for arm in ARMS:
                            for budget in range(1, 6):
                                for horizon in (1, 2, 3):
                                    gain = 0.06 if (arm == "trace" and budget >= 3 and horizon == 3) else 0.01
                                    writer.writerow({
                                        "dataset": dataset, "seed": seed, "arm": arm, "arm_label": arm,
                                        "budget": budget, "horizon": horizon, "n_cases": 60,
                                        "baseline_wrong_n": 30, "baseline_correct_n": 30,
                                        "true_probability_delta_mean": gain, "future_concept_l1_mean": 0.1,
                                        "future_concept_error_reduction_mean": 0.1,
                                        "future_concept_nonselected_l1_mean": 0.1,
                                        "future_concept_nontrivial_fraction_mean": 0.1,
                                        "future_concept_selected_displacement_share_mean": 0.1,
                                        "label_flip_rate": 0.0, "wrong_to_correct_rate": 0.0,
                                        "correct_to_wrong_rate": 0.0, "wrong_to_correct_by_budget_rate": 0.0,
                                        "correct_to_wrong_by_budget_rate": 0.0,
                                    })
            with patch("paper_expos.matched_rollout_intervention.plot_budget_curves"):
                aggregate(root)
            with (root / "efficiency_thresholds.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            trace = next(row for row in rows if row["dataset"] == "barista" and row["arm"] == "trace")
            dense = next(row for row in rows if row["dataset"] == "barista" and row["arm"] == "dense")
            self.assertEqual(trace["minimum_budget"], "3")
            self.assertEqual(trace["attainment_seeds_at_minimum"], "3")
            self.assertEqual(dense["minimum_budget"], "not_reached_by_5")


if __name__ == "__main__":
    unittest.main()
