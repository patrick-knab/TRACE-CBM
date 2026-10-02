#!/usr/bin/env python3
"""Check the main and optional additional-dataset paper inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.paths import dataset_root, embedding_root


DATASETS = {
    "Barista": "Barista/labels",
    "Breakfast": "Breakfast/breakfast_segmentation_coarse",
    "MPII Cooking 2": "MPII_Cooking_2/annotations",
}
EMBEDDINGS = {
    "Barista": "Barista/True_32_clip_pe-l14_state.pkl",
    "Breakfast": "Breakfast/True_32_clip_pe-l14.pkl",
    "MPII Cooking 2": "MPII_Cooking_2/True_32_clip_pe-l14_state.pkl",
}
ADDITIONAL_DATASETS = {
    "GTEA Gaze": "GTEA_Gaze/action_annotation/raw_annotations",
    "EPIC-KITCHENS": "EPIC-KITCHENS-100/annotations/epic-kitchens-100-annotations",
}
ADDITIONAL_EMBEDDINGS = {
    "GTEA Gaze": "GTEA_Gaze/True_32_clip_pe-l14_state.pkl",
    "EPIC-KITCHENS": "EPIC-KITCHENS-100/True_32_clip_pe-l14_state.pkl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=dataset_root())
    parser.add_argument("--embedding-root", type=Path, default=embedding_root())
    parser.add_argument("--include-additional", action="store_true")
    args = parser.parse_args()

    missing = []
    datasets = DATASETS | (ADDITIONAL_DATASETS if args.include_additional else {})
    embeddings = EMBEDDINGS | (ADDITIONAL_EMBEDDINGS if args.include_additional else {})
    for name, relative in datasets.items():
        path = args.dataset_root / relative
        print(f"dataset  {name:16} {'OK' if path.exists() else 'MISSING'}  {path}")
        if not path.exists():
            missing.append(path)
    for name, relative in embeddings.items():
        path = args.embedding_root / relative
        print(f"embedding {name:14} {'OK' if path.exists() else 'MISSING'}  {path}")
        if not path.exists():
            missing.append(path)
    for path in sorted((PROJECT_ROOT / "concepts").glob("*.json")):
        concepts = next(iter(json.loads(path.read_text(encoding="utf-8"))["concepts"].values()))
        print(f"concepts  {len(concepts):3d}             {path.name}")
        if len(concepts) != 128:
            missing.append(path)
    if missing:
        raise SystemExit(f"Preflight failed: {len(missing)} required inputs are missing or invalid")
    print("All paper inputs are present.")


if __name__ == "__main__":
    main()
