"""Training-free PE activity recognition from per-window label similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paper_expos.common import PROJECT_ROOT, write_csv, write_json
from train_models import load_embedding_state
from utils.prepare_datasets import _embed_pe_text_concepts, prepare_data
from utils.paths import embedding_root


PROTOCOLS = (
    {
        "key": "barista",
        "dataset": "barista",
        "test_split": "s1",
        "embedding_relpath": "Barista/True_32_clip_{backbone}_state.pkl",
        "concept_prefix": "barista_iterative_qwen35_pe_l14",
    },
    {
        "key": "breakfast_s1",
        "dataset": "Breakfast",
        "test_split": "official_s1",
        "embedding_relpath": "Breakfast/True_32_clip_{backbone}.pkl",
        "concept_prefix": "breakfast_iterative_qwen35_pe_l14",
    },
    {
        "key": "breakfast_s2",
        "dataset": "Breakfast",
        "test_split": "official_s2",
        "embedding_relpath": "Breakfast/True_32_clip_{backbone}.pkl",
        "concept_prefix": "breakfast_iterative_qwen35_pe_l14",
    },
    {
        "key": "breakfast_s3",
        "dataset": "Breakfast",
        "test_split": "official_s3",
        "embedding_relpath": "Breakfast/True_32_clip_{backbone}.pkl",
        "concept_prefix": "breakfast_iterative_qwen35_pe_l14",
    },
    {
        "key": "breakfast_s4",
        "dataset": "Breakfast",
        "test_split": "official_s4",
        "embedding_relpath": "Breakfast/True_32_clip_{backbone}.pkl",
        "concept_prefix": "breakfast_iterative_qwen35_pe_l14",
    },
    {
        "key": "mpii_attr",
        "dataset": "mpii_cooking_2",
        "test_split": "attr",
        "embedding_relpath": "MPII_Cooking_2/True_32_clip_{backbone}_state.pkl",
        "concept_prefix": "mpii_cooking_2_iterative_qwen35_pe_l14",
    },
    {
        "key": "mpii_dishes",
        "dataset": "mpii_cooking_2",
        "test_split": "dishes",
        "embedding_relpath": "MPII_Cooking_2/True_32_clip_{backbone}_state.pkl",
        "concept_prefix": "mpii_cooking_2_iterative_qwen35_pe_l14",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-index", type=int, required=True)
    parser.add_argument("--split", choices=("train", "dev", "val", "test"), required=True)
    parser.add_argument("--backbone", choices=("pe-l14", "pe-g14"), required=True)
    parser.add_argument("--num-concepts", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-template", default="a video of {label}")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)


def macro_f1(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    scores = []
    for class_index in range(num_classes):
        true_positive = int(np.sum((labels == class_index) & (predictions == class_index)))
        false_positive = int(np.sum((labels != class_index) & (predictions == class_index)))
        false_negative = int(np.sum((labels == class_index) & (predictions != class_index)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return float(np.mean(scores))


def main() -> None:
    args = parse_args()
    records_per_protocol = 3
    max_record_index = len(PROTOCOLS) * records_per_protocol - 1
    if args.record_index < 0 or args.record_index > max_record_index:
        raise IndexError(f"record-index must be in [0, {max_record_index}]")
    protocol = PROTOCOLS[args.record_index // records_per_protocol]
    evaluation_split = args.split
    data_split = "val" if evaluation_split == "dev" else evaluation_split
    embedding_path = (
        embedding_root() / str(protocol["embedding_relpath"]).format(backbone=args.backbone)
    ).resolve()
    concept_set = f"{protocol['concept_prefix']}_{args.num_concepts}_v1"
    concept_path = PROJECT_ROOT / "concepts" / f"{concept_set}.json"
    if not embedding_path.exists() or not concept_path.exists():
        raise FileNotFoundError(
            f"Missing matched inputs for backbone={args.backbone}, num_concepts={args.num_concepts}: "
            f"embedding={embedding_path}, concept_set={concept_path}"
        )
    if args.dry_run:
        print(json.dumps({"protocol": protocol, "evaluation_split": evaluation_split, "backbone": args.backbone, "num_concepts": args.num_concepts, "embedding_path": str(embedding_path), "concept_path": str(concept_path)}, indent=2))
        return

    embeddings = load_embedding_state(embedding_path)
    prepared = prepare_data(
        embeddings,
        concept_path,
        str(protocol["test_split"]),
        args.backbone,
        device=args.device,
        dataset=str(protocol["dataset"]),
        activity_label_mode="action",
        activity_label_fill_mode="sil",
        verbose=False,
    )
    split_data = prepared[data_split]
    activity_names = [str(name) for name in prepared["metadata"]["activity_names"]]
    prompts = [args.prompt_template.format(label=name.replace("_", " ")) for name in activity_names]
    label_embeddings = _embed_pe_text_concepts(prompts, args.backbone, device=args.device)

    valid = (np.asarray(split_data["mask"]) > 0) & (np.asarray(split_data["activity_labels"]) >= 0)
    video_indices, timesteps = np.nonzero(valid)
    features = np.asarray(split_data["raw_features"])[video_indices, timesteps]
    labels = np.asarray(split_data["activity_labels"])[video_indices, timesteps]
    scores = normalize_rows(features) @ normalize_rows(label_embeddings).T
    ranking = np.argsort(-scores, axis=1)
    predictions = ranking[:, 0]
    top3 = np.any(ranking[:, : min(3, len(activity_names))] == labels[:, None], axis=1)

    rows = [
        {
            "video_id": str(split_data["video_ids"][video_index]),
            "timestep": int(timestep),
            "true_label": activity_names[int(label)],
            "prediction": activity_names[int(prediction)],
            "top1_correct": int(prediction == label),
            "top3_correct": int(top3_correct),
        }
        for video_index, timestep, label, prediction, top3_correct in zip(
            video_indices, timesteps, labels, predictions, top3, strict=True
        )
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{protocol['key']}_{evaluation_split}"
    write_csv(args.output_dir / f"{stem}_per_window.csv", rows)
    write_json(
        args.output_dir / f"{stem}_summary.json",
        {
            "protocol": protocol,
            "evaluation_split": evaluation_split,
            "data_split": data_split,
            "method": f"zero_shot_{args.backbone}_label_similarity",
            "backbone": args.backbone,
            "num_concepts": args.num_concepts,
            "prompt_template": args.prompt_template,
            "prompts": prompts,
            "num_windows": int(len(labels)),
            "top1_accuracy": float(np.mean(predictions == labels)),
            "top3_accuracy": float(np.mean(top3)),
            "macro_f1": macro_f1(labels, predictions, len(activity_names)),
        },
    )


if __name__ == "__main__":
    main()
