#!/usr/bin/env python3
"""Build the five-model paper batch for main or additional datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44, 45, 46]
MAIN_PROTOCOLS = [
    ("barista_s1", "barista", "s1", "barista_iterative_qwen35_pe_l14_128_v1", "Barista/True_32_clip_pe-l14_state.pkl", "action"),
    ("breakfast_s1", "Breakfast", "official_s1", "breakfast_iterative_qwen35_pe_l14_128_v1", "Breakfast/True_32_clip_pe-l14.pkl", "action"),
    ("breakfast_s2", "Breakfast", "official_s2", "breakfast_iterative_qwen35_pe_l14_128_v1", "Breakfast/True_32_clip_pe-l14.pkl", "action"),
    ("breakfast_s3", "Breakfast", "official_s3", "breakfast_iterative_qwen35_pe_l14_128_v1", "Breakfast/True_32_clip_pe-l14.pkl", "action"),
    ("breakfast_s4", "Breakfast", "official_s4", "breakfast_iterative_qwen35_pe_l14_128_v1", "Breakfast/True_32_clip_pe-l14.pkl", "action"),
    ("mpii_attr", "mpii_cooking_2", "attr", "mpii_cooking_2_iterative_qwen35_pe_l14_128_v1", "MPII_Cooking_2/True_32_clip_pe-l14_state.pkl", "action"),
    ("mpii_dishes", "mpii_cooking_2", "dishes", "mpii_cooking_2_iterative_qwen35_pe_l14_128_v1", "MPII_Cooking_2/True_32_clip_pe-l14_state.pkl", "action"),
]
ADDITIONAL_PROTOCOLS = [
    ("gtea_s1", "gtea_gaze", "s1", "gtea_gaze_iterative_qwen35_pe_l14_128_v1", "GTEA_Gaze/True_32_clip_pe-l14_state.pkl", "verb"),
    ("epic_kitchens_s1", "epic_kitchens_100", "s1", "epic_kitchens_100_iterative_qwen35_pe_l14_128_v1", "EPIC-KITCHENS-100/True_32_clip_pe-l14_state.pkl", "action"),
]

TRACE_HPARAMS = {
    "future_concept_horizons": [1, 2, 3],
    "st_graph_layers": 1,
    "st_task_graph_layers": 1,
    "st_spatial_top_k": 20,
    "st_temporal_top_k": 20,
    "st_enable_cross_temporal": True,
    "st_cross_temporal_top_k": 20,
    "st_state_activation": "bounded_logit",
    "st_prediction_transform": "logit",
    "motif_z_attention_layers": 0,
    "edge_gate_init": 0,
    "st_residual_gate_init": 0,
    "st_topk_warmup_epochs": 20,
    "st_topk_ramp_epochs": 30,
    "st_topk_training_mode": "gradual",
    "st_activity_feedback_mode": "sparse_label_to_concept",
    "st_activity_feedback_top_k": 40,
    "st_activity_feedback_gate_init": 0.0,
    "st_activity_feedback_history_steps": 3,
    "st_activity_feedback_history_gate_init": 0.0,
}

MODELS = [
    ("linear", "linear_sparse_dynamics_shared_head", {"st_spatial_top_k": 20}),
    ("motif", "motif", {"transformer_layers": 1, "dropout": 0.1, "dimension": 1, "shared_activity_head": True}),
    ("trace", "trace", TRACE_HPARAMS),
    ("slowfast_tcn", "feature_slowfast_tcn", None),
    ("feature_transformer", "feature_transformer", None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("main", "additional"), default="main")
    parser.add_argument(
        "--method-group",
        choices=("all", "linear", "motif", "trace", "blackbox"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path(os.environ.get("TRACE_EMBEDDING_ROOT", "data/embeddings")),
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocols = MAIN_PROTOCOLS if args.scope == "main" else ADDITIONAL_PROTOCOLS
    backbone = "pe-l14"
    method_keys = {
        "all": {"linear", "motif", "trace", "slowfast_tcn", "feature_transformer"},
        "linear": {"linear"},
        "motif": {"motif"},
        "trace": {"trace"},
        "blackbox": {"slowfast_tcn", "feature_transformer"},
    }[args.method_group]
    output = args.output or PROJECT_ROOT / "configs/generated" / (
        "main_models.json" if args.scope == "main" else "additional_dataset_models.json"
    )
    defaults = {
        "window_size": 32,
        "data_root": str(args.embedding_root),
        "horizon": 3,
        "history_length": 5,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "batch_size": 256,
        "num_epochs": 1 if args.smoke else 100,
        "patience": 1 if args.smoke else 40,
        "early_stopping_metric": "val_forecast_top3_accuracy",
        "wandb_project": "TRACE-ICLR2027",
        "wandb_mode": "offline",
        "binary": False,
        "learn_concept_threshold": True,
        "teacher_forcing_start_ratio": 0.5,
        "teacher_forcing_end_ratio": 0.0,
        "concept_forecast_loss_weight": 0.5,
        "concept_forecast_loss_deadzone_std": 0.0,
        "forecast_transition_tolerance_radius": 1,
        "forecast_transition_tolerance_weight": 0.25,
        "graph_edge_regularization_weight": 0.0,
        "graph_intervention_loss_weight": 0.0,
        "graph_intervention_margin": 0.01,
        "graph_intervention_edges_per_batch": 2,
        "graph_disabled_eval": False,
        "graph_corruption_eval": False,
        "classifier_l1_weight": 0.001,
        "activity_sil_false_positive_penalty": 0.5,
        "activity_class_weighting": False,
        "activity_class_weight_cap": 5.0,
        "forecast_sil_false_positive_penalty": 0.5,
        "forecast_class_weighting": False,
        "forecast_class_weight_cap": 5.0,
        "train_sampling_strategy": "uniform",
        "transition_sampler_boundary_radius": 2,
        "transition_sampler_strength": 3.0,
        "transition_sampler_rare_alpha": 0.5,
        "generate_embeddings": False,
        "embedding_batch_size": 256,
        "embedding_num_gpus": 0,
        "enable_tf32": True,
        "random_windows": True,
    }
    if args.smoke:
        defaults.update({"no_wandb": True, "max_sequences_per_split": 4, "batch_size": 4})

    selected_models = [row for row in MODELS if row[0] in method_keys]
    if args.method_group != "all":
        defaults.update({
            "backbone": backbone,
            "activity_label_mode": "action",
            "activity_label_fill_mode": "sil",
        })
    if len(selected_models) == 1:
        _, base_method, model_hparams = selected_models[0]
        defaults["base_method"] = base_method
        if model_hparams is not None:
            defaults["model_hparams"] = model_hparams
        if base_method in {
            "linear_sparse_dynamics_shared_head",
            "feature_slowfast_tcn",
            "feature_transformer",
        }:
            defaults["forecast_transition_tolerance_radius"] = 0
            defaults["forecast_transition_tolerance_weight"] = 0.0
    elif args.method_group == "blackbox":
        defaults["forecast_transition_tolerance_radius"] = 0
        defaults["forecast_transition_tolerance_weight"] = 0.0

    runs = []
    for protocol_key, dataset, split, concept_set, embedding_relpath, activity_label_mode in protocols:
        for model_key, base_method, model_hparams in selected_models:
            run = {
                "name": f"{protocol_key}_{model_key}_seed{{seed}}",
                "dataset": dataset,
                "test_split": split,
                "concept_set": concept_set,
                "embedding_path": str(args.embedding_root / embedding_relpath),
            }
            if args.method_group == "all":
                run.update({
                    "backbone": backbone,
                    "base_method": base_method,
                    "activity_label_mode": activity_label_mode,
                    "activity_label_fill_mode": "sil",
                })
                if model_hparams is not None:
                    run["model_hparams"] = model_hparams
                if base_method in {
                    "linear_sparse_dynamics_shared_head",
                    "feature_slowfast_tcn",
                    "feature_transformer",
                }:
                    run["forecast_transition_tolerance_radius"] = 0
                    run["forecast_transition_tolerance_weight"] = 0.0
            else:
                if activity_label_mode != "action":
                    run["activity_label_mode"] = activity_label_mode
                if len(selected_models) > 1:
                    run["base_method"] = base_method
            runs.append(run)

    payload = {
        "batch_name": f"iclr2027_{args.scope}_dataset_models"
        + (f"_{args.method_group}" if args.method_group != "all" else "")
        + ("_smoke" if args.smoke else ""),
        "batch_output_dir": "runs/train_models",
        "gpus": [0, 1, 2, 3],
        "workers_per_gpu": 2,
        "max_parallel": 8,
        "seeds": [42] if args.smoke else SEEDS,
        "defaults": defaults,
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}: {len(runs)} templates, {len(runs) * len(payload['seeds'])} runs")


if __name__ == "__main__":
    main()
