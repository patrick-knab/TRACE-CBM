from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
UTILS_ROOT = PROJECT_ROOT / "utils"
for import_root in (PROJECT_ROOT, UTILS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from utils.dataset_handling import DATASET_MAP, process_dataset
from utils.model import canonical_method, compact_summary_metrics, train_model
from utils.prepare_datasets import prepare_data
from utils.paths import embedding_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train activity forecasting CBM models.")
    parser.add_argument("--dataset", default="Breakfast", help="Dataset name/key, e.g. Breakfast.")
    parser.add_argument("--test-split", default="s1", help="Dataset split name or split key.")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--concept-set", default="breakfast_simple_v1")
    parser.add_argument("--backbone", default="pe-l14")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=embedding_root(),
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "train_models")
    parser.add_argument("--run-name", default=None, help="Optional label appended to this run directory.")

    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-method",
        default="trace",
        type=canonical_method,
        choices=[
            "linear_sparse_dynamics_shared_head",
            "feature_slowfast_tcn",
            "feature_transformer",
            "motif",
            "trace",
        ],
    )

    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--wandb-project", default="TRACE-ICLR2027")
    parser.add_argument(
        "--wandb-mode",
        default=os.environ.get("WANDB_MODE", "offline"),
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging for this run.")
    parser.add_argument("--binary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--activity-label-mode",
        default="action",
        choices=["action", "verb", "noun", "coarse_noun", "raw_action", "coarse_action"],
        help="Activity target granularity for datasets that expose multiple labels, currently GTEA Gaze.",
    )
    parser.add_argument(
        "--activity-label-fill-mode",
        default="sil",
        choices=["sil", "nearest", "previous", "next", "nearest_internal", "annotated_only"],
        help="How to label windows with no annotated segment overlap.",
    )
    parser.add_argument("--learn-concept-threshold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-forcing-start-ratio", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-end-ratio", type=float, default=0.8)
    parser.add_argument("--concept-forecast-loss-weight", type=float, default=1.0)
    parser.add_argument("--gcssbm-state-transition-loss-weight", type=float, default=0.0)
    parser.add_argument("--concept-intervention-task-loss-weight", type=float, default=0.0)
    parser.add_argument("--concept-intervention-mask-ratio", type=float, default=1.0)
    parser.add_argument("--persistent-intervention-task-loss-weight", type=float, default=0.0)
    parser.add_argument("--persistent-intervention-sample-fraction", type=float, default=0.25)
    parser.add_argument(
        "--persistent-intervention-budgets",
        default="1,3,5",
        help="Comma-separated sparse class-prototype intervention budgets.",
    )
    parser.add_argument("--persistent-intervention-margin", type=float, default=0.05)
    parser.add_argument(
        "--persistent-intervention-target-mode",
        choices=("class_prototype", "instance_oracle"),
        default="class_prototype",
        help="Target sparse rollout edits with class means or instance-specific future concepts.",
    )
    parser.add_argument("--activity-intervention-task-loss-weight", type=float, default=0.0)
    parser.add_argument("--activity-intervention-sample-fraction", type=float, default=0.25)
    parser.add_argument("--activity-intervention-budgets", default="1,3")
    parser.add_argument("--activity-intervention-margin", type=float, default=0.05)
    parser.add_argument(
        "--forecast-horizon-loss-weights",
        default=None,
        help="JSON list or object of H1..Hn loss weights; normalized to sum to one.",
    )
    parser.add_argument(
        "--forecast-transition-tolerance-radius",
        type=int,
        default=0,
        help="Allow forecast labels within +/- this many steps in the transition-tolerant loss.",
    )
    parser.add_argument(
        "--forecast-transition-tolerance-weight",
        type=float,
        default=0.0,
        help="Mix weight in [0, 1] for transition-tolerant forecast loss; 0 keeps exact CE only.",
    )
    parser.add_argument(
        "--concept-forecast-loss-deadzone-std",
        type=float,
        default=0.0,
        help="No future-concept loss is charged inside this standardized absolute-error margin.",
    )
    parser.add_argument("--graph-intervention-loss-weight", type=float, default=0.0)
    parser.add_argument("--graph-task-intervention-loss-weight", type=float, default=0.0)
    parser.add_argument("--graph-task-intervention-margin", type=float, default=0.02)
    parser.add_argument("--graph-intervention-margin", type=float, default=0.01)
    parser.add_argument("--graph-intervention-edges-per-batch", type=int, default=2)
    parser.add_argument(
        "--graph-intervention-sample-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of each training batch used by the expensive high/low graph-intervention "
            "forwards. Normal activity, forecast, and concept losses still use the full batch."
        ),
    )
    parser.add_argument("--graph-edge-regularization-weight", type=float, default=1e-5)
    parser.add_argument("--graph-necessity-loss-weight", type=float, default=0.0)
    parser.add_argument("--graph-necessity-margin", type=float, default=0.02)
    parser.add_argument("--graph-necessity-edges-per-batch", type=int, default=2)
    parser.add_argument("--graph-necessity-sample-fraction", type=float, default=0.25)
    parser.add_argument("--edge-edit-response-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--edge-edit-response-mode",
        choices=("probability_l1", "oracle_margin"),
        default="probability_l1",
    )
    parser.add_argument("--edge-edit-response-margin", type=float, default=0.0005)
    parser.add_argument("--edge-edit-response-edges-per-batch", type=int, default=2)
    parser.add_argument("--edge-edit-response-sample-fraction", type=float, default=0.25)
    parser.add_argument("--graph-disabled-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--graph-corruption-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--graph-corruption-seed",
        type=int,
        default=None,
        help="Seed for deterministic post-training graph-edge shuffling; defaults to --seed.",
    )
    parser.add_argument("--classifier-l1-weight", type=float, default=0.0)
    parser.add_argument(
        "--early-stopping-metric",
        default="val_selection_loss",
        help="Epoch metric used for checkpoint selection, e.g. val_activity_forecast_accuracy.",
    )
    parser.add_argument(
        "--activity-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train/evaluate sequence models for current-window activity classification only.",
    )
    parser.add_argument("--activity-class-weighting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--activity-class-weight-cap", type=float, default=5.0)
    parser.add_argument("--activity-sil-false-positive-penalty", type=float, default=0.0)
    parser.add_argument("--forecast-class-weighting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--forecast-class-weight-cap", type=float, default=5.0)
    parser.add_argument("--forecast-sil-false-positive-penalty", type=float, default=0.0)
    parser.add_argument("--train-sampling-strategy", choices=["uniform", "transition_aware"], default="uniform")
    parser.add_argument("--transition-sampler-boundary-radius", type=int, default=2)
    parser.add_argument("--transition-sampler-strength", type=float, default=3.0)
    parser.add_argument("--transition-sampler-rare-alpha", type=float, default=0.5)
    parser.add_argument(
        "--model-hparams",
        default=None,
        help="JSON object or path to a JSON file with model-constructor hyperparameter overrides.",
    )
    parser.add_argument(
        "--dataset-hparams",
        default=None,
        help="JSON object or path to a JSON file with dataset/preparer hyperparameter overrides.",
    )

    parser.add_argument("--embedding-path", type=Path, default=None)
    parser.add_argument("--generate-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--embedding-num-gpus", type=int, default=0)
    parser.add_argument("--pe-video-batch-size", type=int, default=None)
    parser.add_argument("--pe-target-t", type=int, default=None)
    parser.add_argument("--enable-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-windows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-sequences-per-split",
        type=int,
        default=None,
        help="Optional smoke-test cap; keeps the shortest N sequences from each split after preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_hparams = load_json_arg(args.model_hparams, expected_type=dict, default={})
    args.dataset_hparams = load_json_arg(args.dataset_hparams, expected_type=dict, default={})
    args.forecast_horizon_loss_weights = load_json_arg(
        args.forecast_horizon_loss_weights,
        expected_type=(list, dict),
        default=None,
    )
    device = resolve_device(args.device)
    os.environ["WANDB_MODE"] = "disabled" if args.no_wandb else args.wandb_mode

    run_dir = make_run_dir(args)
    logging_name = run_dir.name
    reproducibility = reproducibility_metadata(run_dir)
    write_json(run_dir / "args.json", vars(args))

    embeddings = load_or_create_embeddings(args)
    preprocessed_data = prepare_data(
        embeddings,
        args.concept_set,
        args.test_split,
        args.backbone,
        device,
        dataset=args.dataset,
        binary=args.binary and not args.learn_concept_threshold,
        activity_label_mode=args.activity_label_mode,
        activity_label_fill_mode=args.activity_label_fill_mode,
        dataset_hparams=args.dataset_hparams,
    )
    if args.max_sequences_per_split is not None:
        preprocessed_data = cap_preprocessed_sequences(preprocessed_data, args.max_sequences_per_split)

    trained_model = train_model(
        preprocessed_data,
        args.horizon,
        args.history_length,
        args.learning_rate,
        args.weight_decay,
        args.batch_size,
        args.num_epochs,
        args.patience,
        None if args.no_wandb else args.wandb_project,
        args.seed,
        args.base_method,
        device=device,
        teacher_forcing_start_ratio=args.teacher_forcing_start_ratio,
        teacher_forcing_end_ratio=args.teacher_forcing_end_ratio,
        concept_forecast_loss_weight=args.concept_forecast_loss_weight,
        concept_forecast_loss_deadzone_std=args.concept_forecast_loss_deadzone_std,
        gcssbm_state_transition_loss_weight=args.gcssbm_state_transition_loss_weight,
        concept_intervention_task_loss_weight=args.concept_intervention_task_loss_weight,
        concept_intervention_mask_ratio=args.concept_intervention_mask_ratio,
        persistent_intervention_task_loss_weight=args.persistent_intervention_task_loss_weight,
        persistent_intervention_sample_fraction=args.persistent_intervention_sample_fraction,
        persistent_intervention_budgets=args.persistent_intervention_budgets,
        persistent_intervention_margin=args.persistent_intervention_margin,
        persistent_intervention_target_mode=args.persistent_intervention_target_mode,
        activity_intervention_task_loss_weight=args.activity_intervention_task_loss_weight,
        activity_intervention_sample_fraction=args.activity_intervention_sample_fraction,
        activity_intervention_budgets=args.activity_intervention_budgets,
        activity_intervention_margin=args.activity_intervention_margin,
        forecast_horizon_loss_weights=args.forecast_horizon_loss_weights,
        forecast_transition_tolerance_radius=args.forecast_transition_tolerance_radius,
        forecast_transition_tolerance_weight=args.forecast_transition_tolerance_weight,
        graph_intervention_loss_weight=args.graph_intervention_loss_weight,
        graph_task_intervention_loss_weight=args.graph_task_intervention_loss_weight,
        graph_task_intervention_margin=args.graph_task_intervention_margin,
        graph_intervention_margin=args.graph_intervention_margin,
        graph_intervention_edges_per_batch=args.graph_intervention_edges_per_batch,
        graph_intervention_sample_fraction=args.graph_intervention_sample_fraction,
        graph_edge_regularization_weight=args.graph_edge_regularization_weight,
        graph_necessity_loss_weight=args.graph_necessity_loss_weight,
        graph_necessity_margin=args.graph_necessity_margin,
        graph_necessity_edges_per_batch=args.graph_necessity_edges_per_batch,
        graph_necessity_sample_fraction=args.graph_necessity_sample_fraction,
        edge_edit_response_loss_weight=args.edge_edit_response_loss_weight,
        edge_edit_response_mode=args.edge_edit_response_mode,
        edge_edit_response_margin=args.edge_edit_response_margin,
        edge_edit_response_edges_per_batch=args.edge_edit_response_edges_per_batch,
        edge_edit_response_sample_fraction=args.edge_edit_response_sample_fraction,
        graph_disabled_eval=args.graph_disabled_eval,
        graph_corruption_eval=args.graph_corruption_eval,
        graph_corruption_seed=args.graph_corruption_seed,
        classifier_l1_weight=args.classifier_l1_weight,
        activity_only=args.activity_only,
        learn_concept_threshold=args.learn_concept_threshold,
        activity_class_weighting=args.activity_class_weighting,
        activity_class_weight_cap=args.activity_class_weight_cap,
        activity_sil_false_positive_penalty=args.activity_sil_false_positive_penalty,
        forecast_class_weighting=args.forecast_class_weighting,
        forecast_class_weight_cap=args.forecast_class_weight_cap,
        forecast_sil_false_positive_penalty=args.forecast_sil_false_positive_penalty,
        train_sampling_strategy=args.train_sampling_strategy,
        transition_sampler_boundary_radius=args.transition_sampler_boundary_radius,
        transition_sampler_strength=args.transition_sampler_strength,
        transition_sampler_rare_alpha=args.transition_sampler_rare_alpha,
        early_stopping_metric=args.early_stopping_metric,
        model_hparams=args.model_hparams,
        run_metadata={
            "dataset": args.dataset,
            "test_split": args.test_split,
            "window_size": int(args.window_size),
            "concept_set": args.concept_set,
            "backbone": args.backbone,
            "activity_label_mode": args.activity_label_mode,
            "activity_label_fill_mode": args.activity_label_fill_mode,
            "dataset_hparams": args.dataset_hparams,
            "graph_disabled_eval": bool(args.graph_disabled_eval),
            "graph_corruption_eval": bool(args.graph_corruption_eval),
            "graph_corruption_seed": args.graph_corruption_seed,
            "run_name": args.run_name,
            "logging_name": logging_name,
            "embedding_path": str(args.embedding_path or default_embedding_path(args)),
            "random_windows": bool(args.random_windows),
            **reproducibility,
        },
    )

    checkpoint_path = run_dir / "model.pt"
    torch.save(
        {
            "model": trained_model.model,
            "base_method": trained_model.base_method,
            "horizon": trained_model.horizon,
            "history_length": trained_model.history_length,
            "activity_names": trained_model.activity_names,
            "num_concepts": trained_model.num_concepts,
            "num_activities": trained_model.num_activities,
            "info": trained_model.info,
            "metadata": preprocessed_data.get("metadata", {}),
            "args": vars(args),
        },
        checkpoint_path,
    )
    write_json(run_dir / "metrics.json", trained_model.metrics)
    write_json(run_dir / "summary_metrics.json", compact_summary_metrics(trained_model.metrics))
    write_json(run_dir / "history.json", trained_model.history)

    print(f"[train_models] Saved checkpoint: {checkpoint_path}", flush=True)
    print(f"[train_models] Saved metrics: {run_dir / 'metrics.json'}", flush=True)
    print(f"[train_models] {trained_model}", flush=True)


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {value!r}, but CUDA is not available.")
    return value


def reproducibility_metadata(run_dir: Path) -> dict[str, Any]:
    commit = git_output(["git", "rev-parse", "HEAD"])
    dirty = git_output(["git", "status", "--porcelain"])
    return {
        "code_git_commit": commit,
        "code_git_dirty": bool(dirty),
        "resolved_command": " ".join([sys.executable, *sys.argv]),
        "python_executable": sys.executable,
        "output_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def git_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def make_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dataset = safe_name(args.dataset)
    split = safe_name(args.test_split)
    method = safe_name(args.base_method)
    backbone = safe_name(args.backbone)
    name_parts = [
        dataset,
        timestamp,
        split,
        backbone,
        method,
        f"h{args.horizon}",
        f"seed{args.seed}",
    ]
    if args.run_name:
        name_parts.append(safe_name(args.run_name))
    name = "_".join(name_parts)
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def safe_name(value: object) -> str:
    return str(value).replace("+", "_").replace("/", "_").replace(" ", "_")


def load_json_arg(value: str | None, *, expected_type: type | tuple[type, ...], default: Any) -> Any:
    if value in (None, ""):
        return default
    stripped = value.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        parsed = json.loads(stripped)
    else:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"JSON argument path does not exist: {path}")
        parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, expected_type):
        expected_name = (
            " or ".join(item.__name__ for item in expected_type)
            if isinstance(expected_type, tuple)
            else expected_type.__name__
        )
        raise TypeError(f"Expected {expected_name} JSON, got {type(parsed).__name__}.")
    return parsed


def load_or_create_embeddings(args: argparse.Namespace) -> Any:
    if str(args.dataset).strip().lower().replace("-", "_").startswith("synthetic_"):
        return {"dataset_name": args.dataset, "config": {"dataset_key": args.dataset}}
    embedding_path = args.embedding_path or default_embedding_path(args)
    if embedding_path.exists():
        print(f"[train_models] Loading embeddings: {embedding_path}", flush=True)
        return load_embedding_state(embedding_path)

    if not args.generate_embeddings:
        raise FileNotFoundError(
            f"Embedding file not found: {embedding_path}. "
            "Pass --generate-embeddings or provide --embedding-path."
        )

    print(f"[train_models] Embeddings not found; generating with process_dataset: {embedding_path}", flush=True)
    generated_path = Path(
        process_dataset(
            args.dataset,
            args.backbone,
            args.window_size,
            random=args.random_windows,
            batch_size=args.embedding_batch_size,
            pe_video_batch_size=args.pe_video_batch_size,
            pe_target_T=args.pe_target_t,
            enable_tf32=args.enable_tf32,
            seed=args.seed,
            num_gpus=args.embedding_num_gpus,
        )
    )
    # Parallel embedding generation returns the merged .npy state path, while also
    # writing the final pickle at the requested embedding path when enabled.
    load_path = embedding_path if embedding_path.exists() else generated_path
    return load_embedding_state(load_path)


def cap_preprocessed_sequences(preprocessed_data: dict[str, Any], max_sequences: int) -> dict[str, Any]:
    if max_sequences <= 0:
        raise ValueError("--max-sequences-per-split must be positive when provided.")

    capped = dict(preprocessed_data)
    split_sizes: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        split = dict(capped[split_name])
        lengths = split.get("lengths")
        if lengths is None:
            raise KeyError(f"preprocessed_data[{split_name!r}] is missing lengths.")
        num_sequences = int(len(lengths))
        keep_count = min(max_sequences, num_sequences)
        order = torch.as_tensor(lengths).argsort().cpu().numpy()[:keep_count]
        max_length = int(max(split["lengths"][order])) if keep_count else 0

        trimmed = {}
        for key, value in split.items():
            if hasattr(value, "shape") and len(value.shape) >= 1 and int(value.shape[0]) == num_sequences:
                selected = value[order]
                if key in {"concepts", "activity_labels", "mask"} and len(selected.shape) >= 2:
                    selected = selected[:, :max_length]
                trimmed[key] = selected
            else:
                trimmed[key] = value
        capped[split_name] = trimmed
        split_sizes[split_name] = keep_count

    metadata = dict(capped.get("metadata", {}))
    metadata["split_sizes"] = split_sizes
    metadata["max_sequences_per_split"] = int(max_sequences)
    capped["metadata"] = metadata
    print(
        "[train_models] Applied smoke sequence cap: "
        + ", ".join(
            f"{name}={capped[name]['concepts'].shape}" for name in ("train", "val", "test")
        ),
        flush=True,
    )
    return capped


def load_embedding_state(path: Path) -> Any:
    if path.suffix.lower() == ".npy":
        return np_load_dict(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def np_load_dict(path: Path) -> Any:
    import numpy as np

    loaded = np.load(path, allow_pickle=True)
    if hasattr(loaded, "item"):
        try:
            return loaded.item()
        except ValueError:
            pass
    return loaded


def default_embedding_path(args: argparse.Namespace) -> Path:
    dataset_dir = DATASET_MAP.get(str(args.dataset).lower(), args.dataset)
    return args.data_root / dataset_dir / f"{args.random_windows}_{args.window_size}_clip_{args.backbone}.pkl"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


if __name__ == "__main__":
    main()
