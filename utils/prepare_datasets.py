from __future__ import annotations

import json
import os
import re
import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from utils.paths import concept_root, dataset_root


DATASET_ROOT = dataset_root()
DEFAULT_BREAKFAST_ANNOTATION_ROOT = DATASET_ROOT / "Breakfast/breakfast_segmentation_coarse"
DEFAULT_GTEA_GAZE_ANNOTATION_ROOT = Path(
    DATASET_ROOT / "GTEA_Gaze/action_annotation/raw_annotations"
)
DEFAULT_MPII_COOKING_2_ANNOTATION_ROOT = Path(
    DATASET_ROOT / "MPII_Cooking_2/annotations"
)
DEFAULT_BARISTA_LABEL_ROOT = Path(
    DATASET_ROOT / "Barista/labels"
)
DEFAULT_EPIC_KITCHENS_100_ROOT = Path(
    DATASET_ROOT / "EPIC-KITCHENS-100"
)
_IGNORE_ACTIVITY_LABEL = "__IGNORE__"
BREAKFAST_S1_SPLIT = {
    "train": [
        "P03",
        "P04",
        "P06",
        "P07",
        "P09",
        "P12",
        "P13",
        "P15",
        "P17",
        "P18",
        "P21",
        "P23",
        "P24",
        "P25",
        "P26",
        "P28",
        "P29",
        "P30",
        "P31",
        "P32",
        "P33",
        "P34",
        "P37",
        "P39",
        "P40",
        "P41",
        "P43",
        "P44",
        "P45",
        "P47",
        "P48",
        "P49",
        "P50",
        "P51",
        "P52",
        "P54",
    ],
    "val": ["P10", "P11", "P14", "P20", "P22", "P35", "P36", "P42"],
    "test": ["P05", "P08", "P16", "P19", "P27", "P38", "P46", "P53"],
}
BREAKFAST_BENCHMARK_TEST_PARTICIPANTS = {
    "s1": [f"P{participant:02d}" for participant in range(3, 16)],
    "s2": [f"P{participant:02d}" for participant in range(16, 29)],
    "s3": [f"P{participant:02d}" for participant in range(29, 42)],
    "s4": [f"P{participant:02d}" for participant in range(42, 55)],
}

_RANGE_PATTERN = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s+(.+?)\s*$")
_SUBJECT_PATTERN = re.compile(r"/((?:P|F)\d+)/")
_GTEA_SUBJECT_PATTERN = re.compile(r"^(?P<subject>[A-Za-z]*P\d+)-")
_MPII_SEQUENCE_PATTERN = re.compile(r"(?P<sequence>s(?P<subject>\d+)-d(?P<dish>\d+))(?:-cam-\d+)?$")
_FRAME_FILE_PATTERN = re.compile(r"(\d+)")
_BACKBONE_TEXT_MODELS = {
    "b32": "openai/clip-vit-base-patch32",
    "b16": "openai/clip-vit-base-patch16",
    "l14": "openai/clip-vit-large-patch14",
    "res50": "res50",
    "rn50": "res50",
    "siglip": "google/siglip-base-patch16-224",
    "siglipl14": "google/siglip-so400m-patch14-384",
    "pe-l14": "pe-l14",
    "pe_l14": "pe-l14",
    "pe-g14": "pe-g14",
    "pe_g14": "pe-g14",
}


@dataclass(frozen=True)
class AnnotationSegment:
    start_frame: int
    end_frame: int
    label: str


@dataclass(frozen=True)
class WindowSequence:
    video_id: str
    video_path: str
    subject: str
    recipe_label: str
    concepts: np.ndarray
    activity_labels: np.ndarray
    raw_features: np.ndarray | None = None
    example_mask: np.ndarray | None = None

    @property
    def length(self) -> int:
        return int(self.activity_labels.shape[0])


class CompactArrayDict(dict):
    """Dictionary that displays large array payloads as shapes in notebooks."""

    def _summary_items(self) -> List[str]:
        items = []
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                items.append(f"{key}: ndarray(shape={value.shape}, dtype={value.dtype})")
            else:
                items.append(f"{key}: {value!r}")
        return items

    def __repr__(self) -> str:
        return "{\n  " + ",\n  ".join(self._summary_items()) + "\n}"

    def _repr_pretty_(self, printer, cycle: bool) -> None:
        if cycle:
            printer.text(f"{type(self).__name__}(...)")
            return
        printer.text(repr(self))


def prepare_data(
    embeddings,
    concept_set: str | os.PathLike[str],
    test_split: str | Mapping[str, Sequence[str]],
    backbone: str,
    device: str | torch.device | None = None,
    *,
    dataset: str | None = None,
    annotation_root: str | os.PathLike[str] | None = None,
    similarity_scale: float = 5.0,
    text_embedding_cache: str | os.PathLike[str] | None = None,
    binary: bool = False,
    activity_label_mode: str = "action",
    activity_label_fill_mode: str = "sil",
    dataset_hparams: Mapping[str, object] | None = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Prepare window-level concept activations and labels for forecasting.

    Returns a dictionary with ``train``, ``val`` and ``test`` splits. Each split
    contains padded ``concepts`` with shape ``[N, T_max, C]``, window
    ``activity_labels``, ``mask``, ``lengths`` and identifying metadata.
    Set ``binary=True`` to threshold concept activations at 0.5 instead of
    keeping continuous sigmoid-scaled similarities.
    """

    dataset_name = _resolve_dataset_name(embeddings, dataset)
    if dataset_name == "synthetic_delayed_edge":
        return _prepare_synthetic_delayed_edge(
            dataset_hparams=dataset_hparams or {},
            verbose=verbose,
        )
    preparer = _DATASET_PREPARERS.get(dataset_name)
    if preparer is None:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Available preparers: {sorted(_DATASET_PREPARERS)}"
        )
    return preparer(
        embeddings=embeddings,
        concept_set=concept_set,
        test_split=test_split,
        backbone=backbone,
        annotation_root=annotation_root,
        similarity_scale=similarity_scale,
        device=device,
        text_embedding_cache=text_embedding_cache,
        binary=binary,
        activity_label_mode=activity_label_mode,
        activity_label_fill_mode=activity_label_fill_mode,
        verbose=verbose,
    )


def _prepare_breakfast(
    *,
    embeddings,
    concept_set: str | os.PathLike[str],
    test_split: str | Mapping[str, Sequence[str]],
    backbone: str,
    annotation_root: str | os.PathLike[str] | None,
    similarity_scale: float,
    device: str | torch.device | None,
    text_embedding_cache: str | os.PathLike[str] | None,
    binary: bool,
    activity_label_mode: str,
    activity_label_fill_mode: str,
    verbose: bool,
) -> Dict[str, object]:
    del activity_label_mode
    del activity_label_fill_mode
    video_embeddings = _extract_video_embeddings(embeddings)
    labels = _extract_recipe_labels(embeddings, video_embeddings)
    if len(labels) != len(video_embeddings):
        raise ValueError(
            f"Embedding object has {len(video_embeddings)} videos but {len(labels)} labels."
        )

    concept_set_name, concept_names = _load_concept_set(concept_set)
    if verbose:
        print(
            f"[prepare_data] Embedding {len(concept_names)} text concepts with backbone={backbone} "
            f"on device={device or 'auto'}",
            flush=True,
        )
    text_embeddings = _load_or_embed_text_concepts(
        concept_names,
        backbone,
        device=device,
        cache_path=text_embedding_cache,
        concept_set_name=concept_set_name,
    )
    first_video = next(iter(video_embeddings.values()))
    embedding_dim = int(np.asarray(first_video).shape[-1])
    if int(text_embeddings.shape[-1]) != embedding_dim:
        raise ValueError(
            "Text concept embeddings and video embeddings have different dimensions: "
            f"text={text_embeddings.shape[-1]}, video={embedding_dim}. "
            "Use the same backbone for `backbone` and `embeddings`."
        )

    annotation_index = _build_annotation_index(
        Path(annotation_root) if annotation_root is not None else DEFAULT_BREAKFAST_ANNOTATION_ROOT
    )
    split_subjects = _resolve_breakfast_split(test_split)

    if verbose:
        print(f"[prepare_data] Aligning {len(video_embeddings)} videos to Breakfast annotations", flush=True)
    raw_sequences: List[Tuple[str, str, np.ndarray]] = []
    activity_names = set()
    dropped: List[Dict[str, str]] = []
    for (video_path, window_embeddings), recipe_label in zip(video_embeddings.items(), labels):
        num_windows = int(np.asarray(window_embeddings).shape[0])
        if num_windows == 0:
            dropped.append({"video_path": video_path, "reason": "zero_windows"})
            continue
        video_id = _breakfast_annotation_key(video_path)
        annotation_path = annotation_index.get(video_id)
        if annotation_path is None:
            dropped.append({"video_path": video_path, "reason": "missing_annotation"})
            continue
        labels_for_windows = _window_labels_from_segments(
            _load_annotation_segments(annotation_path),
            num_windows=num_windows,
            window_spans=_get_window_spans(embeddings, video_path),
            fps=_get_video_fps(embeddings, video_path),
        )
        activity_names.update(labels_for_windows)
        raw_sequences.append((video_path, recipe_label, np.asarray(labels_for_windows)))

    if not raw_sequences:
        raise RuntimeError("No Breakfast videos could be aligned with annotation files.")

    activity_names = sorted(activity_names)
    activity_to_index = {name: index for index, name in enumerate(activity_names)}

    sequences_by_split = {"train": [], "val": [], "test": []}
    subject_to_split = {
        subject: split_name
        for split_name, subjects in split_subjects.items()
        for subject in subjects
    }

    for video_path, recipe_label, string_labels in raw_sequences:
        subject = _subject_from_path(video_path)
        split_name = subject_to_split.get(subject)
        if split_name is None:
            dropped.append({"video_path": video_path, "reason": f"subject_not_in_split:{subject}"})
            continue
        if verbose and len(sequences_by_split[split_name]) == 0:
            mode = "binary" if binary else "continuous"
            print(f"[prepare_data] Computing {mode} concept activations for {split_name} split", flush=True)
        window_embeddings = np.asarray(video_embeddings[video_path], dtype=np.float32)
        concepts = _concept_activations(
            window_embeddings,
            text_embeddings,
            similarity_scale,
            binary=binary,
        )
        activity_labels = np.asarray(
            [activity_to_index[label] for label in string_labels], dtype=np.int64
        )
        sequences_by_split[split_name].append(
            WindowSequence(
                video_id=_breakfast_annotation_key(video_path),
                video_path=video_path,
                subject=subject,
                recipe_label=recipe_label,
                concepts=concepts,
                activity_labels=activity_labels,
                raw_features=window_embeddings,
            )
        )

    splits = {name: _pack_split(sequences) for name, sequences in sequences_by_split.items()}
    if verbose:
        print(
            "[prepare_data] Done: "
            + ", ".join(f"{name}={splits[name]['concepts'].shape}" for name in ("train", "val", "test")),
            flush=True,
        )
    return CompactArrayDict({
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "metadata": {
            "dataset": "breakfast",
            "concept_set": concept_set_name,
            "concept_names": concept_names,
            "num_concepts": len(concept_names),
            "activity_names": activity_names,
            "num_activities": len(activity_names),
            "backbone": str(backbone),
            "similarity_scale": float(similarity_scale),
            "binary_concepts": bool(binary),
            "concept_activation_mode": "binary" if binary else "continuous",
            "split_subjects": split_subjects,
            "split_sizes": {name: int(splits[name]["concepts"].shape[0]) for name in splits},
            "dropped_videos": dropped,
        },
    })


def _prepare_gtea_gaze(
    *,
    embeddings,
    concept_set: str | os.PathLike[str],
    test_split: str | Mapping[str, Sequence[str]],
    backbone: str,
    annotation_root: str | os.PathLike[str] | None,
    similarity_scale: float,
    device: str | torch.device | None,
    text_embedding_cache: str | os.PathLike[str] | None,
    binary: bool,
    activity_label_mode: str,
    activity_label_fill_mode: str,
    verbose: bool,
) -> Dict[str, object]:
    video_embeddings = _extract_video_embeddings(embeddings)
    labels = _extract_recipe_labels(embeddings, video_embeddings)
    if len(labels) != len(video_embeddings):
        raise ValueError(
            f"Embedding object has {len(video_embeddings)} videos but {len(labels)} labels."
        )

    concept_set_name, concept_names = _load_concept_set(concept_set)
    if verbose:
        print(
            f"[prepare_data] Embedding {len(concept_names)} text concepts with backbone={backbone} "
            f"on device={device or 'auto'}",
            flush=True,
        )
    text_embeddings = _load_or_embed_text_concepts(
        concept_names,
        backbone,
        device=device,
        cache_path=text_embedding_cache,
        concept_set_name=concept_set_name,
    )
    first_video = next(iter(video_embeddings.values()))
    embedding_dim = int(np.asarray(first_video).shape[-1])
    if int(text_embeddings.shape[-1]) != embedding_dim:
        raise ValueError(
            "Text concept embeddings and video embeddings have different dimensions: "
            f"text={text_embeddings.shape[-1]}, video={embedding_dim}. "
            "Use the same backbone for `backbone` and `embeddings`."
        )

    if verbose:
        print(f"[prepare_data] Aligning {len(video_embeddings)} GTEA Gaze sessions", flush=True)

    split_key = _gtea_split_membership_key(test_split)
    action_label_index = _load_gtea_action_label_index(
        Path(annotation_root) if annotation_root is not None else DEFAULT_GTEA_GAZE_ANNOTATION_ROOT
    )
    label_mode = _canonical_gtea_label_mode(activity_label_mode)
    fill_mode = _canonical_gtea_fill_mode(activity_label_fill_mode)
    raw_sequences: List[Tuple[str, str, str, str, np.ndarray]] = []
    activity_names = set()
    dropped: List[Dict[str, str]] = []
    for (video_path, window_embeddings), recipe_label in zip(video_embeddings.items(), labels):
        num_windows = int(np.asarray(window_embeddings).shape[0])
        if num_windows == 0:
            dropped.append({"video_path": video_path, "reason": "zero_windows"})
            continue
        meta = _get_video_meta(embeddings, video_path)
        segments = _gtea_segments_from_meta(
            meta,
            action_label_index=action_label_index,
            split_key=split_key,
            label_mode=label_mode,
        )
        if not segments:
            dropped.append({"video_path": video_path, "reason": "missing_action_segments"})
            continue
        labels_for_windows = _window_labels_from_segments(
            segments,
            num_windows=num_windows,
            window_spans=_get_window_spans(embeddings, video_path),
            fps=_get_video_fps(embeddings, video_path),
            background_label=_IGNORE_ACTIVITY_LABEL if fill_mode == "annotated_only" else "SIL" if fill_mode == "sil" else None,
            empty_overlap_strategy="sil" if fill_mode == "annotated_only" else fill_mode,
        )
        session_id = str(meta.get("session_id") or Path(video_path).name)
        subject = _gtea_subject_from_session(session_id)
        activity_names.update(label for label in labels_for_windows if label != _IGNORE_ACTIVITY_LABEL)
        raw_sequences.append(
            (video_path, session_id, subject, str(recipe_label), np.asarray(labels_for_windows))
        )

    if not raw_sequences:
        raise RuntimeError("No GTEA Gaze sessions could be aligned with embedded action segments.")

    activity_names = sorted(activity_names)
    activity_to_index = {name: index for index, name in enumerate(activity_names)}
    split_sessions = _resolve_gtea_session_split(
        [session_id for _, session_id, _, _, _ in raw_sequences],
        test_split,
    )
    session_to_split = {
        session_id: split_name
        for split_name, session_ids in split_sessions.items()
        for session_id in session_ids
    }

    sequences_by_split = {"train": [], "val": [], "test": []}
    for video_path, session_id, subject, recipe_label, string_labels in raw_sequences:
        split_name = session_to_split.get(session_id)
        if split_name is None:
            dropped.append({"video_path": video_path, "reason": f"session_not_in_split:{session_id}"})
            continue
        if verbose and len(sequences_by_split[split_name]) == 0:
            mode = "binary" if binary else "continuous"
            print(f"[prepare_data] Computing {mode} concept activations for {split_name} split", flush=True)
        window_embeddings = np.asarray(video_embeddings[video_path], dtype=np.float32)
        concepts = _concept_activations(
            window_embeddings,
            text_embeddings,
            similarity_scale,
            binary=binary,
        )
        activity_labels = np.asarray(
            [
                -1 if label == _IGNORE_ACTIVITY_LABEL else activity_to_index[label]
                for label in string_labels
            ],
            dtype=np.int64,
        )
        sequences_by_split[split_name].append(
            WindowSequence(
                video_id=session_id,
                video_path=video_path,
                subject=subject,
                recipe_label=recipe_label,
                concepts=concepts,
                activity_labels=activity_labels,
                raw_features=window_embeddings,
            )
        )

    splits = {name: _pack_split(sequences) for name, sequences in sequences_by_split.items()}
    if verbose:
        print(
            "[prepare_data] Done: "
            + ", ".join(f"{name}={splits[name]['concepts'].shape}" for name in ("train", "val", "test")),
            flush=True,
        )
    return CompactArrayDict({
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "metadata": {
            "dataset": "gtea_gaze",
            "concept_set": concept_set_name,
            "concept_names": concept_names,
            "num_concepts": len(concept_names),
            "activity_names": activity_names,
            "num_activities": len(activity_names),
            "backbone": str(backbone),
            "similarity_scale": float(similarity_scale),
            "binary_concepts": bool(binary),
            "concept_activation_mode": "binary" if binary else "continuous",
            "split_sessions": split_sessions,
            "split_sizes": {name: int(splits[name]["concepts"].shape[0]) for name in splits},
            "dropped_videos": dropped,
            "split_note": "Deterministic session-level split to avoid train/test leakage across windows.",
            "action_label_source": "official_gtea_action_id",
            "activity_label_mode": label_mode,
            "activity_label_fill_mode": fill_mode,
            "action_label_index": str(
                Path(annotation_root) if annotation_root is not None else DEFAULT_GTEA_GAZE_ANNOTATION_ROOT
            ),
        },
    })


def _prepare_mpii_cooking_2(
    *,
    embeddings,
    concept_set: str | os.PathLike[str],
    test_split: str | Mapping[str, Sequence[str]],
    backbone: str,
    annotation_root: str | os.PathLike[str] | None,
    similarity_scale: float,
    device: str | torch.device | None,
    text_embedding_cache: str | os.PathLike[str] | None,
    binary: bool,
    activity_label_mode: str,
    activity_label_fill_mode: str,
    verbose: bool,
) -> Dict[str, object]:
    video_embeddings = _extract_video_embeddings(embeddings)
    concept_set_name, concept_names = _load_concept_set(concept_set)
    if verbose:
        print(
            f"[prepare_data] Embedding {len(concept_names)} text concepts with backbone={backbone} "
            f"on device={device or 'auto'}",
            flush=True,
        )
    text_embeddings = _load_or_embed_text_concepts(
        concept_names,
        backbone,
        device=device,
        cache_path=text_embedding_cache,
        concept_set_name=concept_set_name,
    )
    first_video = next(iter(video_embeddings.values()))
    embedding_dim = int(np.asarray(first_video).shape[-1])
    if int(text_embeddings.shape[-1]) != embedding_dim:
        raise ValueError(
            "Text concept embeddings and video embeddings have different dimensions: "
            f"text={text_embeddings.shape[-1]}, video={embedding_dim}. "
            "Use the same backbone for `backbone` and `embeddings`."
        )

    root = Path(annotation_root) if annotation_root is not None else DEFAULT_MPII_COOKING_2_ANNOTATION_ROOT
    label_mode = _canonical_mpii_label_mode(activity_label_mode)
    fill_mode = _canonical_mpii_fill_mode(activity_label_fill_mode)
    annotation_by_video = _load_mpii_annotation_segments(root, label_mode=label_mode)
    split_sequences = _resolve_mpii_sequence_split(root, test_split)
    sequence_to_split = {
        sequence_id: split_name
        for split_name, sequence_ids in split_sequences.items()
        for sequence_id in sequence_ids
    }

    if verbose:
        print(f"[prepare_data] Aligning {len(video_embeddings)} MPII Cooking 2 sequences", flush=True)

    raw_sequences: List[Tuple[str, str, str, str, np.ndarray]] = []
    activity_names = set()
    dropped: List[Dict[str, str]] = []
    for video_path, window_embeddings in video_embeddings.items():
        num_windows = int(np.asarray(window_embeddings).shape[0])
        if num_windows == 0:
            dropped.append({"video_path": video_path, "reason": "zero_windows"})
            continue
        video_id = _mpii_video_id_from_path(video_path)
        sequence_id = _mpii_sequence_id(video_id)
        segments = annotation_by_video.get(video_id) or annotation_by_video.get(sequence_id)
        if not segments:
            dropped.append({"video_path": video_path, "reason": f"missing_annotation:{video_id}"})
            continue
        labels_for_windows = _mpii_window_labels_from_segments(
            video_path,
            segments,
            num_windows=num_windows,
            window_spans=_get_window_spans(embeddings, video_path),
            fps=_get_video_fps(embeddings, video_path),
            background_label=(
                _IGNORE_ACTIVITY_LABEL
                if fill_mode == "annotated_only"
                else "SIL" if fill_mode in {"sil", "nearest_internal"} else None
            ),
            empty_overlap_strategy=(
                "sil" if fill_mode in {"annotated_only", "nearest_internal"} else fill_mode
            ),
        )
        if fill_mode == "nearest_internal":
            labels_for_windows = _fill_internal_background_labels(labels_for_windows, background_label="SIL")
        subject = _mpii_subject_from_sequence(sequence_id)
        recipe_label = _mpii_dish_from_sequence(sequence_id)
        activity_names.update(label for label in labels_for_windows if label != _IGNORE_ACTIVITY_LABEL)
        raw_sequences.append(
            (video_path, video_id, subject, recipe_label, np.asarray(labels_for_windows))
        )

    if not raw_sequences:
        raise RuntimeError("No MPII Cooking 2 sequences could be aligned with official annotations.")

    activity_names = sorted(activity_names)
    activity_to_index = {name: index for index, name in enumerate(activity_names)}
    sequences_by_split = {"train": [], "val": [], "test": []}
    for video_path, video_id, subject, recipe_label, string_labels in raw_sequences:
        sequence_id = _mpii_sequence_id(video_id)
        split_name = sequence_to_split.get(sequence_id)
        if split_name is None:
            dropped.append({"video_path": video_path, "reason": f"sequence_not_in_split:{sequence_id}"})
            continue
        if verbose and len(sequences_by_split[split_name]) == 0:
            mode = "binary" if binary else "continuous"
            print(f"[prepare_data] Computing {mode} concept activations for {split_name} split", flush=True)
        window_embeddings = np.asarray(video_embeddings[video_path], dtype=np.float32)
        concepts = _concept_activations(
            window_embeddings,
            text_embeddings,
            similarity_scale,
            binary=binary,
        )
        activity_labels = np.asarray(
            [
                -1 if label == _IGNORE_ACTIVITY_LABEL else activity_to_index[label]
                for label in string_labels
            ],
            dtype=np.int64,
        )
        sequences_by_split[split_name].append(
            WindowSequence(
                video_id=video_id,
                video_path=video_path,
                subject=subject,
                recipe_label=recipe_label,
                concepts=concepts,
                activity_labels=activity_labels,
                raw_features=window_embeddings,
            )
        )

    splits = {name: _pack_split(sequences) for name, sequences in sequences_by_split.items()}
    if verbose:
        print(
            "[prepare_data] Done: "
            + ", ".join(f"{name}={splits[name]['concepts'].shape}" for name in ("train", "val", "test")),
            flush=True,
        )
    return CompactArrayDict({
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "metadata": {
            "dataset": "mpii_cooking_2",
            "concept_set": concept_set_name,
            "concept_names": concept_names,
            "num_concepts": len(concept_names),
            "activity_names": activity_names,
            "num_activities": len(activity_names),
            "backbone": str(backbone),
            "similarity_scale": float(similarity_scale),
            "binary_concepts": bool(binary),
            "concept_activation_mode": "binary" if binary else "continuous",
            "split_sequences": split_sequences,
            "split_sizes": {name: int(splits[name]["concepts"].shape[0]) for name in splits},
            "dropped_videos": dropped,
            "split_note": "Official MPII Cooking 2 sequence split from experimentalSetup.",
            "activity_label_source": "attributesAnnotations_MPII-Cooking-2.mat",
            "activity_label_mode": label_mode,
            "activity_label_fill_mode": fill_mode,
            "annotation_root": str(root),
        },
    })


def _prepare_barista(
    *,
    embeddings,
    concept_set: str | os.PathLike[str],
    test_split: str | Mapping[str, Sequence[str]],
    backbone: str,
    annotation_root: str | os.PathLike[str] | None,
    similarity_scale: float,
    device: str | torch.device | None,
    text_embedding_cache: str | os.PathLike[str] | None,
    binary: bool,
    activity_label_mode: str,
    activity_label_fill_mode: str,
    verbose: bool,
) -> Dict[str, object]:
    video_embeddings = _extract_video_embeddings(embeddings)
    concept_set_name, concept_names = _load_concept_set(concept_set)
    if verbose:
        print(
            f"[prepare_data] Embedding {len(concept_names)} text concepts with backbone={backbone} "
            f"on device={device or 'auto'}",
            flush=True,
        )
    text_embeddings = _load_or_embed_text_concepts(
        concept_names,
        backbone,
        device=device,
        cache_path=text_embedding_cache,
        concept_set_name=concept_set_name,
    )
    first_video = next(iter(video_embeddings.values()))
    embedding_dim = int(np.asarray(first_video).shape[-1])
    if int(text_embeddings.shape[-1]) != embedding_dim:
        raise ValueError(
            "Text concept embeddings and video embeddings have different dimensions: "
            f"text={text_embeddings.shape[-1]}, video={embedding_dim}. "
            "Use the same backbone for `backbone` and `embeddings`."
        )

    label_root = _resolve_barista_label_root(annotation_root)
    label_mode = _canonical_barista_label_mode(activity_label_mode)
    fill_mode = _canonical_mpii_fill_mode(activity_label_fill_mode)
    frame_labels_by_sequence = _load_barista_frame_labels(label_root, label_mode=label_mode)
    short_gap_fill_frames = _embedding_window_size(embeddings)
    split_sequences = _resolve_barista_sequence_split(frame_labels_by_sequence.keys(), test_split)
    sequence_to_split = {
        sequence_id: split_name
        for split_name, sequence_ids in split_sequences.items()
        for sequence_id in sequence_ids
    }

    if verbose:
        print(f"[prepare_data] Aligning {len(video_embeddings)} BARISTA sequences", flush=True)

    raw_sequences: List[Tuple[str, str, str, str, np.ndarray]] = []
    activity_names = set()
    dropped: List[Dict[str, str]] = []
    for video_path, window_embeddings in video_embeddings.items():
        num_windows = int(np.asarray(window_embeddings).shape[0])
        if num_windows == 0:
            dropped.append({"video_path": video_path, "reason": "zero_windows"})
            continue
        sequence_id = _barista_sequence_id_from_path(video_path)
        frame_labels = frame_labels_by_sequence.get(sequence_id)
        if not frame_labels:
            dropped.append({"video_path": video_path, "reason": f"missing_frame_labels:{sequence_id}"})
            continue
        if fill_mode in {"sil", "nearest_internal"} and short_gap_fill_frames is not None:
            frame_labels = _fill_short_internal_frame_gaps(
                frame_labels,
                background_label=_IGNORE_ACTIVITY_LABEL,
                max_gap_frames=short_gap_fill_frames,
            )
        labels_for_windows = _barista_window_labels_from_frame_labels(
            video_path,
            frame_labels,
            num_windows=num_windows,
            background_label=(
                _IGNORE_ACTIVITY_LABEL
                if fill_mode == "annotated_only"
                else "SIL" if fill_mode in {"sil", "nearest_internal"} else None
            ),
            empty_overlap_strategy=(
                "sil" if fill_mode in {"annotated_only", "nearest_internal"} else fill_mode
            ),
        )
        subject = sequence_id.split("__", 1)[0]
        activity_names.update(label for label in labels_for_windows if label != _IGNORE_ACTIVITY_LABEL)
        raw_sequences.append(
            (video_path, sequence_id, subject, "barista", np.asarray(labels_for_windows))
        )

    if not raw_sequences:
        raise RuntimeError("No BARISTA sequences could be aligned with labels/activity_labels.csv.")

    activity_names = sorted(activity_names)
    activity_to_index = {name: index for index, name in enumerate(activity_names)}
    sequences_by_split = {"train": [], "val": [], "test": []}
    for video_path, sequence_id, subject, recipe_label, string_labels in raw_sequences:
        split_name = sequence_to_split.get(sequence_id)
        if split_name is None:
            dropped.append({"video_path": video_path, "reason": f"sequence_not_in_split:{sequence_id}"})
            continue
        if verbose and len(sequences_by_split[split_name]) == 0:
            mode = "binary" if binary else "continuous"
            print(f"[prepare_data] Computing {mode} concept activations for {split_name} split", flush=True)
        window_embeddings = np.asarray(video_embeddings[video_path], dtype=np.float32)
        concepts = _concept_activations(
            window_embeddings,
            text_embeddings,
            similarity_scale,
            binary=binary,
        )
        activity_labels = np.asarray(
            [
                -1 if label == _IGNORE_ACTIVITY_LABEL else activity_to_index[label]
                for label in string_labels
            ],
            dtype=np.int64,
        )
        sequences_by_split[split_name].append(
            WindowSequence(
                video_id=sequence_id,
                video_path=video_path,
                subject=subject,
                recipe_label=recipe_label,
                concepts=concepts,
                activity_labels=activity_labels,
                raw_features=window_embeddings,
            )
        )

    splits = {name: _pack_split(sequences) for name, sequences in sequences_by_split.items()}
    if verbose:
        print(
            "[prepare_data] Done: "
            + ", ".join(f"{name}={splits[name]['concepts'].shape}" for name in ("train", "val", "test")),
            flush=True,
        )
    return CompactArrayDict({
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "metadata": {
            "dataset": "barista",
            "concept_set": concept_set_name,
            "concept_names": concept_names,
            "num_concepts": len(concept_names),
            "activity_names": activity_names,
            "num_activities": len(activity_names),
            "backbone": str(backbone),
            "similarity_scale": float(similarity_scale),
            "binary_concepts": bool(binary),
            "concept_activation_mode": "binary" if binary else "continuous",
            "split_sequences": split_sequences,
            "split_sizes": {name: int(splits[name]["concepts"].shape[0]) for name in splits},
            "dropped_videos": dropped,
            "split_note": "Deterministic BARISTA sequence-level hash split.",
            "activity_label_source": "labels/activity_labels.csv",
            "activity_label_mode": label_mode,
            "activity_label_fill_mode": fill_mode,
            "barista_short_gap_fill_max_frames": short_gap_fill_frames,
            "annotation_root": str(label_root),
        },
    })


def _prepare_epic_kitchens(
    *,
    embeddings,
    concept_set: str | os.PathLike[str],
    test_split: str | Mapping[str, Sequence[str]],
    backbone: str,
    annotation_root: str | os.PathLike[str] | None,
    similarity_scale: float,
    device: str | torch.device | None,
    text_embedding_cache: str | os.PathLike[str] | None,
    binary: bool,
    activity_label_mode: str,
    activity_label_fill_mode: str,
    verbose: bool,
) -> Dict[str, object]:
    video_embeddings = _extract_video_embeddings(embeddings)
    concept_set_name, concept_names = _load_concept_set(concept_set)
    if verbose:
        print(
            f"[prepare_data] Embedding {len(concept_names)} text concepts with backbone={backbone} "
            f"on device={device or 'auto'}",
            flush=True,
        )
    text_embeddings = _load_or_embed_text_concepts(
        concept_names,
        backbone,
        device=device,
        cache_path=text_embedding_cache,
        concept_set_name=concept_set_name,
    )
    first_video = next(iter(video_embeddings.values()))
    embedding_dim = int(np.asarray(first_video).shape[-1])
    if int(text_embeddings.shape[-1]) != embedding_dim:
        raise ValueError(
            "Text concept embeddings and video embeddings have different dimensions: "
            f"text={text_embeddings.shape[-1]}, video={embedding_dim}. "
            "Use the same backbone for `backbone` and `embeddings`."
        )

    label_mode = _canonical_epic_label_mode(activity_label_mode)
    fill_mode = _canonical_mpii_fill_mode(activity_label_fill_mode)
    annotation_dir = _resolve_epic_annotation_root(annotation_root)
    annotations_by_video, official_split_by_video = _load_epic_annotations(
        annotation_dir,
        label_mode=label_mode,
    )

    embedded_video_ids = {_epic_video_id_from_path(path) for path in video_embeddings}
    split_videos = _resolve_epic_video_split(
        [
            video_id
            for video_id, split_name in official_split_by_video.items()
            if split_name == "train" and video_id in embedded_video_ids
        ],
        [
            video_id
            for video_id, split_name in official_split_by_video.items()
            if split_name == "validation" and video_id in embedded_video_ids
        ],
        test_split,
    )
    video_to_split = {
        video_id: split_name
        for split_name, video_ids in split_videos.items()
        for video_id in video_ids
    }

    if verbose:
        print(f"[prepare_data] Aligning {len(video_embeddings)} EPIC-KITCHENS videos", flush=True)

    raw_sequences: List[Tuple[str, str, str, str, np.ndarray]] = []
    activity_names = set()
    dropped: List[Dict[str, str]] = []
    for video_path, window_embeddings in video_embeddings.items():
        num_windows = int(np.asarray(window_embeddings).shape[0])
        if num_windows == 0:
            dropped.append({"video_path": video_path, "reason": "zero_windows"})
            continue
        video_id = _epic_video_id_from_path(video_path)
        segments = annotations_by_video.get(video_id)
        if not segments:
            dropped.append({"video_path": video_path, "reason": f"missing_annotation:{video_id}"})
            continue
        split_name = video_to_split.get(video_id)
        if split_name is None:
            dropped.append({"video_path": video_path, "reason": f"video_not_in_split:{video_id}"})
            continue
        empty_overlap_strategy = "nearest" if fill_mode == "nearest_internal" else fill_mode
        labels_for_windows = _window_labels_from_segments(
            segments,
            num_windows=num_windows,
            window_spans=_get_window_spans(embeddings, video_path),
            fps=_get_video_fps(embeddings, video_path),
            background_label=(
                _IGNORE_ACTIVITY_LABEL
                if fill_mode == "annotated_only"
                else "SIL" if fill_mode == "sil" else None
            ),
            empty_overlap_strategy=(
                "sil" if fill_mode == "annotated_only" else empty_overlap_strategy
            ),
        )
        subject = video_id.split("_", 1)[0]
        activity_names.update(label for label in labels_for_windows if label != _IGNORE_ACTIVITY_LABEL)
        raw_sequences.append((video_path, video_id, subject, "epic", np.asarray(labels_for_windows)))

    if not raw_sequences:
        raise RuntimeError("No EPIC-KITCHENS videos could be aligned with EPIC_100 annotations.")

    activity_names = sorted(activity_names)
    activity_to_index = {name: index for index, name in enumerate(activity_names)}
    activity_label_components = _epic_activity_label_components(activity_names, label_mode)
    sequences_by_split = {"train": [], "val": [], "test": []}
    for video_path, video_id, subject, recipe_label, string_labels in raw_sequences:
        split_name = video_to_split.get(video_id)
        if split_name is None:
            continue
        if verbose and len(sequences_by_split[split_name]) == 0:
            mode = "binary" if binary else "continuous"
            print(f"[prepare_data] Computing {mode} concept activations for {split_name} split", flush=True)
        window_embeddings = np.asarray(video_embeddings[video_path], dtype=np.float32)
        concepts = _concept_activations(
            window_embeddings,
            text_embeddings,
            similarity_scale,
            binary=binary,
        )
        activity_labels = np.asarray(
            [
                -1 if label == _IGNORE_ACTIVITY_LABEL else activity_to_index[label]
                for label in string_labels
            ],
            dtype=np.int64,
        )
        sequences_by_split[split_name].append(
            WindowSequence(
                video_id=video_id,
                video_path=video_path,
                subject=subject,
                recipe_label=recipe_label,
                concepts=concepts,
                activity_labels=activity_labels,
                raw_features=window_embeddings,
            )
        )

    splits = {name: _pack_split(sequences) for name, sequences in sequences_by_split.items()}
    annotated_video_ids = set(annotations_by_video)
    if verbose:
        print(
            "[prepare_data] Done: "
            + ", ".join(f"{name}={splits[name]['concepts'].shape}" for name in ("train", "val", "test")),
            flush=True,
        )
    return CompactArrayDict({
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "metadata": {
            "dataset": "epic_kitchens_100",
            "concept_set": concept_set_name,
            "concept_names": concept_names,
            "num_concepts": len(concept_names),
            "activity_names": activity_names,
            "activity_label_components": activity_label_components,
            "activity_component_names": sorted(activity_label_components[0]) if activity_label_components else [],
            "num_activities": len(activity_names),
            "backbone": str(backbone),
            "similarity_scale": float(similarity_scale),
            "binary_concepts": bool(binary),
            "concept_activation_mode": "binary" if binary else "continuous",
            "split_videos": split_videos,
            "split_sizes": {name: int(splits[name]["concepts"].shape[0]) for name in splits},
            "dropped_videos": dropped,
            "downloaded_unannotated_videos": sorted(embedded_video_ids - annotated_video_ids),
            "annotated_missing_videos": sorted(annotated_video_ids - embedded_video_ids),
            "split_note": "Official train videos are hash-split into train/val; official validation videos are used as test.",
            "activity_label_source": "EPIC_100_train.csv + EPIC_100_validation.csv",
            "activity_label_mode": label_mode,
            "activity_label_fill_mode": fill_mode,
            "annotation_root": str(annotation_dir),
        },
    })


def _resolve_dataset_name(embeddings, dataset: str | None) -> str:
    if dataset is not None:
        return _canonical_dataset_key(dataset)
    if isinstance(embeddings, Mapping):
        config = embeddings.get("config")
        if isinstance(config, Mapping):
            for key in ("dataset_key", "dataset", "dataset_name"):
                if key in config:
                    return _canonical_dataset_key(config[key])
        return _canonical_dataset_key(embeddings.get("dataset_name", "breakfast"))
    return _canonical_dataset_key(getattr(embeddings, "dataset_name", "breakfast"))


def _canonical_dataset_key(dataset: object) -> str:
    key = str(dataset).strip().lower().replace("-", "_")
    aliases = {
        "gtea": "gtea_gaze",
        "egtea": "gtea_gaze",
        "egtea_gaze": "gtea_gaze",
        "gtea_gaze": "gtea_gaze",
        "gtea gaze": "gtea_gaze",
        "gtea_gaze_dataset": "gtea_gaze",
        "breakfast": "breakfast",
        "mpii": "mpii_cooking_2",
        "mpii_cooking": "mpii_cooking_2",
        "mpii_cooking_2": "mpii_cooking_2",
        "mpiicooking2": "mpii_cooking_2",
        "barista": "barista",
        "epic": "epic_kitchens_100",
        "epic100": "epic_kitchens_100",
        "epic_100": "epic_kitchens_100",
        "epic_kitchens": "epic_kitchens_100",
        "epic_kitchens_100": "epic_kitchens_100",
        "epicskitchen": "epic_kitchens_100",
        "synthetic_delayed_edge": "synthetic_delayed_edge",
        "synthetic_delayed": "synthetic_delayed_edge",
        "synthetic_graph_memory": "synthetic_delayed_edge",
    }
    return aliases.get(key, key)


def _extract_video_embeddings(embeddings) -> Dict[str, np.ndarray]:
    if isinstance(embeddings, Mapping):
        video_embeddings = embeddings.get("video_embeddings")
        if isinstance(video_embeddings, dict) and video_embeddings:
            return video_embeddings
    video_embeddings = getattr(embeddings, "video_embeddings", None)
    if not isinstance(video_embeddings, dict) or not video_embeddings:
        raise ValueError("Expected `embeddings.video_embeddings` to be a non-empty dict.")
    return video_embeddings


def _extract_recipe_labels(embeddings, video_embeddings: Mapping[str, np.ndarray]) -> List[str]:
    if isinstance(embeddings, Mapping):
        labels = embeddings.get("labels")
        if isinstance(labels, list) and len(labels) == len(video_embeddings):
            return [str(label) for label in labels]
        meta = embeddings.get("video_meta")
        if isinstance(meta, Mapping):
            inferred = []
            for path in video_embeddings:
                video_meta = meta.get(path)
                if isinstance(video_meta, Mapping):
                    inferred.append(str(video_meta.get("sequence_label") or video_meta.get("activity_label") or Path(path).name))
                else:
                    inferred.append(Path(path).stem.split("_", 1)[-1])
            return inferred
    labels = getattr(embeddings, "labels", None)
    if isinstance(labels, list) and len(labels) == len(video_embeddings):
        return [str(label) for label in labels]
    return [Path(path).stem.split("_", 1)[-1] for path in video_embeddings]


def _embedding_window_size(embeddings) -> int | None:
    config = embeddings.get("config") if isinstance(embeddings, Mapping) else getattr(embeddings, "config", None)
    if not isinstance(config, Mapping) or "window_size" not in config:
        return None
    try:
        value = int(config["window_size"])
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _load_concept_set(concept_set: str | os.PathLike[str]) -> Tuple[str, List[str]]:
    path = Path(concept_set)
    if not path.exists():
        name = str(concept_set)
        filename = name if name.endswith(".json") else f"{name}.json"
        bases = []
        configured_root = concept_root()
        if configured_root:
            bases.append(configured_root)
        bases.append(Path(__file__).resolve().parent.parent / "concepts")
        bases.append(Path(__file__).resolve().parent.parent / "concepts" / "cache")
        for base in bases:
            candidate = base / filename
            if candidate.exists():
                path = candidate
                break
    if not path.exists():
        raise FileNotFoundError(f"Concept set JSON not found: {concept_set}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    concepts = payload.get("concepts", payload)
    if isinstance(concepts, list):
        return path.stem, [str(item) for item in concepts]
    if not isinstance(concepts, dict) or not concepts:
        raise ValueError(f"Invalid concept set JSON: {path}")
    if len(concepts) == 1:
        name, values = next(iter(concepts.items()))
        return str(name), [str(item) for item in values]
    key = path.stem
    if key in concepts:
        return key, [str(item) for item in concepts[key]]
    raise ValueError(
        f"Concept set JSON has multiple sets {sorted(concepts)}; pass a file with one set or name it after a key."
    )


def _choose_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _canonical_backbone(backbone: str) -> str:
    key = str(backbone).strip().lower()
    return _BACKBONE_TEXT_MODELS.get(key, key)


def _embed_text_concepts(
    concepts: Sequence[str],
    backbone: str,
    device: str | torch.device | None = None,
) -> np.ndarray:
    model_id = _canonical_backbone(backbone)
    if model_id in {"pe-l14", "pe-g14"}:
        return _embed_pe_text_concepts(concepts, model_id, device=device)
    if model_id == "res50":
        return _embed_openai_clip_text_concepts(concepts, "RN50", device=device)
    if model_id in {
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-large-patch14",
    }:
        return _embed_openai_clip_text_concepts(concepts, model_id, device=device)
    if "siglip" in model_id:
        return _embed_siglip_text_concepts(concepts, model_id, device=device)
    return _embed_hf_clip_text_concepts(concepts, model_id, device=device)


def _load_or_embed_text_concepts(
    concepts: Sequence[str],
    backbone: str,
    *,
    device: str | torch.device | None,
    cache_path: str | os.PathLike[str] | None,
    concept_set_name: str,
) -> np.ndarray:
    cache = _resolve_text_embedding_cache(cache_path, concepts, backbone, concept_set_name)
    metadata_path = cache.with_suffix(".json")
    if cache.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("backbone") == str(backbone)
            and metadata.get("concept_set_name") == concept_set_name
            and metadata.get("concept_names") == list(concepts)
        ):
            return np.load(cache).astype(np.float32)

    embeddings = _embed_text_concepts(concepts, backbone, device=device).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings)
    metadata_path.write_text(
        json.dumps(
            {
                "backbone": str(backbone),
                "concept_set_name": concept_set_name,
                "concept_names": list(concepts),
                "shape": list(embeddings.shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return embeddings


def _resolve_text_embedding_cache(
    cache_path: str | os.PathLike[str] | None,
    concepts: Sequence[str],
    backbone: str,
    concept_set_name: str,
) -> Path:
    if cache_path is not None:
        return Path(cache_path)
    digest = sha1(
        json.dumps(
            {
                "backbone": str(backbone),
                "concept_set_name": concept_set_name,
                "concepts": list(concepts),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return Path(__file__).resolve().parent.parent / "concepts" / "cache" / f"{concept_set_name}_{backbone}_{digest}.npy"


def _embed_hf_clip_text_concepts(
    concepts: Sequence[str],
    model_id: str,
    device: str | torch.device | None = None,
) -> np.ndarray:
    from transformers import CLIPModel, CLIPTokenizer

    device = _choose_device(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(concepts), 32):
            encoded = tokenizer(
                list(concepts[start : start + 32]),
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            features = model.get_text_features(**encoded).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            chunks.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _embed_siglip_text_concepts(
    concepts: Sequence[str],
    model_id: str,
    device: str | torch.device | None = None,
) -> np.ndarray:
    from transformers import AutoModel, AutoProcessor

    device = _choose_device(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(concepts), 32):
            encoded = processor(
                text=list(concepts[start : start + 32]),
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            features = model.get_text_features(**encoded).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            chunks.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _embed_openai_clip_text_concepts(
    concepts: Sequence[str],
    model_id: str,
    device: str | torch.device | None = None,
) -> np.ndarray:
    import clip

    clip_model_names = {
        "openai/clip-vit-base-patch32": "ViT-B/32",
        "openai/clip-vit-base-patch16": "ViT-B/16",
        "openai/clip-vit-large-patch14": "ViT-L/14",
    }
    clip_model_name = clip_model_names.get(model_id, model_id)
    device = _choose_device(device)
    model, _ = clip.load(clip_model_name, device=str(device))
    model = model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(concepts), 32):
            tokens = clip.tokenize(list(concepts[start : start + 32])).to(device)
            features = model.encode_text(tokens).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            chunks.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _embed_pe_text_concepts(
    concepts: Sequence[str],
    model_id: str,
    device: str | torch.device | None = None,
) -> np.ndarray:
    try:
        from .core.vision_encoder import pe
        from .core.vision_encoder import transforms as pe_transforms
    except ImportError:
        from core.vision_encoder import pe
        from core.vision_encoder import transforms as pe_transforms

    config_name = "PE-Core-G14-448" if model_id == "pe-g14" else "PE-Core-L14-336"
    device = _choose_device(device)
    model = pe.CLIP.from_config(config_name, pretrained=True).to(device).eval()
    tokenizer = pe_transforms.get_text_tokenizer(model.context_length)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(concepts), 32):
            tokens = tokenizer(list(concepts[start : start + 32])).to(device)
            features = model.encode_text(tokens).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            chunks.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)

def _concept_activations(
    video_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    similarity_scale: float,
    *,
    binary: bool = False,
    mode: str = "cosine"
) -> np.ndarray:
    cosine = _normalize_rows(video_embeddings.astype(np.float32)) @ _normalize_rows(
        text_embeddings.astype(np.float32)
    ).T
    
    if mode == "cosine":
        activations = cosine

    elif mode == "sigmoid":
        activations = 1.0 / (1.0 + np.exp(-float(similarity_scale) * cosine))

    elif mode == "zscore":
        mean = cosine.mean(axis=0, keepdims=True)
        std = cosine.std(axis=0, keepdims=True)
        activations = (cosine - mean) / np.clip(std, 1e-6, None)

    else:
        raise ValueError(f"Unknown concept activation mode: {mode}")

    if binary:
        activations = activations >= 0.5

    return activations.astype(np.float32)


def _build_annotation_index(annotation_root: Path) -> Dict[str, Path]:
    if not annotation_root.exists():
        raise FileNotFoundError(f"Breakfast annotation root not found: {annotation_root}")
    index: Dict[str, Path] = {}
    for path in sorted(annotation_root.rglob("*")):
        if path.suffix.lower() not in {".txt", ".xml"}:
            continue
        existing = index.get(path.stem)
        if existing is None or (existing.suffix.lower() == ".xml" and path.suffix.lower() == ".txt"):
            index[path.stem] = path
    return index


def _load_annotation_segments(path: Path) -> List[AnnotationSegment]:
    if path.suffix.lower() == ".txt":
        segments = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            match = _RANGE_PATTERN.match(line)
            if match is None:
                raise ValueError(f"Could not parse annotation line {line!r} in {path}")
            start, end, label = match.groups()
            segments.append(AnnotationSegment(int(start), int(end), label.strip()))
        return segments
    if path.suffix.lower() == ".xml":
        root = ET.parse(path).getroot()
        return [
            AnnotationSegment(
                int(node.attrib["startPoint"]),
                int(node.attrib["endPoint"]),
                str(node.attrib.get("name", "")).strip(),
            )
            for node in root.findall(".//MotionLabel")
        ]
    raise ValueError(f"Unsupported annotation file: {path}")


def _window_labels_from_segments(
    segments: Sequence[AnnotationSegment],
    num_windows: int,
    window_spans: Sequence[Tuple[float, float]] | None,
    fps: float,
    background_label: str | None = None,
    empty_overlap_strategy: str = "last",
) -> List[str]:
    if not segments:
        raise ValueError("Cannot label windows from an empty annotation.")
    total_frames = max(segment.end_frame for segment in segments)
    labels = []
    for index in range(num_windows):
        if window_spans and index < len(window_spans):
            start_frame = max(1, int(round(window_spans[index][0] * fps)) + 1)
            end_frame = min(total_frames, max(start_frame, int(round(window_spans[index][1] * fps))))
        else:
            start_frame = min(total_frames, index + 1)
            end_frame = start_frame
        labels.append(
            _majority_overlap_label(
                segments,
                start_frame,
                end_frame,
                background_label,
                empty_overlap_strategy=empty_overlap_strategy,
            )
        )
    return labels


def _majority_overlap_label(
    segments: Sequence[AnnotationSegment],
    start_frame: int,
    end_frame: int,
    background_label: str | None = None,
    empty_overlap_strategy: str = "last",
) -> str:
    best_label = segments[-1].label
    best_overlap = -1
    for segment in segments:
        overlap = max(0, min(end_frame, segment.end_frame) - max(start_frame, segment.start_frame) + 1)
        if overlap > best_overlap:
            best_label = segment.label
            best_overlap = overlap
    if best_overlap <= 0 and background_label is not None:
        return background_label
    if best_overlap <= 0:
        return _empty_overlap_label(segments, start_frame, end_frame, empty_overlap_strategy)
    return best_label


def _empty_overlap_label(
    segments: Sequence[AnnotationSegment],
    start_frame: int,
    end_frame: int,
    strategy: str,
) -> str:
    strategy = str(strategy).strip().lower()
    center = 0.5 * (start_frame + end_frame)
    if strategy == "previous":
        previous = [segment for segment in segments if segment.end_frame < start_frame]
        if previous:
            return previous[-1].label
        return segments[0].label
    if strategy == "next":
        for segment in segments:
            if segment.start_frame > end_frame:
                return segment.label
        return segments[-1].label
    if strategy == "nearest":
        best_segment = segments[0]
        best_distance = float("inf")
        for segment in segments:
            if center < segment.start_frame:
                distance = float(segment.start_frame) - center
            elif center > segment.end_frame:
                distance = center - float(segment.end_frame)
            else:
                distance = 0.0
            if distance < best_distance:
                best_distance = distance
                best_segment = segment
        return best_segment.label
    return segments[-1].label


def _fill_internal_background_labels(labels: Sequence[str], *, background_label: str) -> List[str]:
    filled = [str(label) for label in labels]
    if len(filled) < 3:
        return filled

    previous_true: List[int | None] = [None] * len(filled)
    last_true: int | None = None
    for index, label in enumerate(filled):
        if label != background_label:
            last_true = index
        previous_true[index] = last_true

    next_true: List[int | None] = [None] * len(filled)
    last_true = None
    for index in range(len(filled) - 1, -1, -1):
        if filled[index] != background_label:
            last_true = index
        next_true[index] = last_true

    for index, label in enumerate(filled):
        if filled[index] != background_label:
            continue
        previous_index = previous_true[index]
        next_index = next_true[index]
        if previous_index is None or next_index is None:
            continue
        previous_distance = index - previous_index
        next_distance = next_index - index
        filled[index] = filled[previous_index] if previous_distance <= next_distance else filled[next_index]
    return filled


def _fill_short_internal_frame_gaps(
    labels: Sequence[str],
    *,
    background_label: str,
    max_gap_frames: int,
) -> List[str]:
    filled = [str(label) for label in labels]
    max_gap_frames = int(max_gap_frames)
    if len(filled) < 3 or max_gap_frames < 1:
        return filled

    index = 0
    while index < len(filled):
        if filled[index] != background_label:
            index += 1
            continue
        start = index
        while index < len(filled) and filled[index] == background_label:
            index += 1
        end = index - 1
        previous_index = start - 1
        next_index = index
        if previous_index < 0 or next_index >= len(filled):
            continue
        if filled[previous_index] == background_label or filled[next_index] == background_label:
            continue
        gap_length = end - start + 1
        if gap_length > max_gap_frames:
            continue
        for gap_index in range(start, end + 1):
            previous_distance = gap_index - previous_index
            next_distance = next_index - gap_index
            filled[gap_index] = (
                filled[previous_index]
                if previous_distance <= next_distance
                else filled[next_index]
            )
    return filled


def _get_window_spans(embeddings, video_path: str) -> Sequence[Tuple[float, float]] | None:
    if isinstance(embeddings, Mapping):
        spans = embeddings.get("video_window_spans")
        if isinstance(spans, dict):
            return spans.get(video_path)
    spans = getattr(embeddings, "video_window_spans", None)
    if isinstance(spans, dict):
        return spans.get(video_path)
    return None


def _get_video_fps(embeddings, video_path: str) -> float:
    meta = _get_video_meta(embeddings, video_path)
    if float(meta.get("fps", 0.0) or 0.0) > 0.0:
        return float(meta["fps"])
    return 15.0


def _get_video_meta(embeddings, video_path: str) -> Mapping[str, object]:
    if isinstance(embeddings, Mapping):
        meta = embeddings.get("video_meta")
        if isinstance(meta, Mapping):
            video_meta = meta.get(video_path)
            if isinstance(video_meta, Mapping):
                return video_meta
    meta = getattr(embeddings, "video_meta", None)
    if isinstance(meta, dict):
        video_meta = meta.get(video_path)
        if isinstance(video_meta, Mapping):
            return video_meta
    return {}


def _gtea_segments_from_meta(
    video_meta: Mapping[str, object],
    *,
    action_label_index: Mapping[int, str] | None = None,
    split_key: str = "split1",
    label_mode: str = "action",
) -> List[AnnotationSegment]:
    raw_segments = video_meta.get("action_segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
        return []
    fps = float(video_meta.get("fps", 12.0) or 12.0)
    segments: List[AnnotationSegment] = []
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            continue
        label = _gtea_segment_label(
            raw,
            action_label_index=action_label_index,
            split_key=split_key,
            label_mode=label_mode,
        )
        if not label:
            continue
        if raw.get("start_time") is not None and raw.get("end_time") is not None:
            start_frame = int(round(float(raw["start_time"]) * fps)) + 1
            end_frame = int(round(float(raw["end_time"]) * fps))
        elif raw.get("start_frame") is not None and raw.get("end_frame") is not None:
            start_frame = int(raw["start_frame"])
            end_frame = int(raw["end_frame"])
        elif raw.get("start_ms") is not None and raw.get("end_ms") is not None:
            start_frame = int(round(float(raw["start_ms"]) * fps / 1000.0)) + 1
            end_frame = int(round(float(raw["end_ms"]) * fps / 1000.0))
        else:
            continue
        segments.append(AnnotationSegment(max(1, start_frame), max(start_frame, end_frame), label))
    return sorted(segments, key=lambda segment: (segment.start_frame, segment.end_frame))


def _canonical_gtea_label_mode(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "action": "action",
        "action_label": "action",
        "official_action": "action",
        "official": "action",
        "verb": "verb",
        "verb_label": "verb",
        "noun": "noun",
        "noun_label": "noun",
        "noun_labels": "noun",
        "coarse_noun": "coarse_noun",
        "noun_coarse": "coarse_noun",
        "coarse_object": "coarse_noun",
        "raw_action": "raw_action",
        "raw": "raw_action",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported GTEA activity_label_mode={value!r}. "
            "Use one of: action, verb, noun, coarse_noun, raw_action."
        )
    return aliases[key]


def _canonical_gtea_fill_mode(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "sil": "sil",
        "background": "sil",
        "nearest": "nearest",
        "near": "nearest",
        "previous": "previous",
        "prev": "previous",
        "forward_fill": "previous",
        "ffill": "previous",
        "next": "next",
        "backward_fill": "next",
        "bfill": "next",
        "last": "previous",
        "annotated": "annotated_only",
        "annotated_only": "annotated_only",
        "drop_empty": "annotated_only",
        "ignore_empty": "annotated_only",
        "segments_only": "annotated_only",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported GTEA activity_label_fill_mode={value!r}. "
            "Use one of: sil, nearest, previous, next."
        )
    return aliases[key]


def _canonical_mpii_label_mode(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "action": "activity",
        "activity": "activity",
        "coarse": "coarse_activity",
        "coarse_action": "coarse_activity",
        "coarse_activity": "coarse_activity",
        "activity_coarse": "coarse_activity",
        "verb": "activity",
        "raw_action": "class_name",
        "class": "class_name",
        "class_name": "class_name",
        "composite": "class_name",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported MPII activity_label_mode={value!r}. "
            "Use action/activity for verb-level labels, coarse_action for grouped labels, "
            "or raw_action/class_name for composite labels."
        )
    return aliases[key]


def _canonical_mpii_fill_mode(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "annotated": "annotated_only",
        "annotated_only": "annotated_only",
        "drop_empty": "annotated_only",
        "ignore_empty": "annotated_only",
        "segments_only": "annotated_only",
        "nearest_internal": "nearest_internal",
        "internal_nearest": "nearest_internal",
        "fill_internal": "nearest_internal",
        "sil_nearest_internal": "nearest_internal",
    }
    if key in aliases:
        return aliases[key]
    return _canonical_gtea_fill_mode(value)


def _canonical_barista_label_mode(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "action": "display_name",
        "activity": "display_name",
        "display": "display_name",
        "display_name": "display_name",
        "raw_action": "label_index",
        "label": "label_index",
        "label_index": "label_index",
        "index": "label_index",
        "slug": "slug",
        "class_id": "class_id",
        "activity_class_id": "class_id",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported BARISTA activity_label_mode={value!r}. "
            "Use action/display_name, raw_action/label_index, slug, or class_id."
        )
    return aliases[key]


def _canonical_epic_label_mode(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "action": "action",
        "activity": "action",
        "verb_noun": "action",
        "verb+noun": "action",
        "narration": "narration",
        "raw_action": "narration",
        "verb": "verb",
        "verb_class": "verb_class",
        "verb_id": "verb_class",
        "noun": "noun",
        "noun_class": "noun_class",
        "noun_id": "noun_class",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported EPIC activity_label_mode={value!r}. "
            "Use one of: action, narration, verb, noun, verb_class, noun_class."
        )
    return aliases[key]


def _resolve_epic_annotation_root(annotation_root: str | os.PathLike[str] | None) -> Path:
    root = Path(annotation_root) if annotation_root is not None else DEFAULT_EPIC_KITCHENS_100_ROOT
    candidates = [
        root,
        root / "annotations" / "epic-kitchens-100-annotations",
        root / "epic-kitchens-100-annotations",
    ]
    for candidate in candidates:
        if (candidate / "EPIC_100_train.csv").exists() and (
            candidate / "EPIC_100_validation.csv"
        ).exists():
            return candidate
    raise FileNotFoundError(f"EPIC-KITCHENS annotation CSVs not found under {root}")


def _load_epic_annotations(
    annotation_root: Path,
    *,
    label_mode: str,
) -> Tuple[Dict[str, List[AnnotationSegment]], Dict[str, str]]:
    by_video: Dict[str, List[AnnotationSegment]] = {}
    split_by_video: Dict[str, str] = {}
    for filename, split_name in (
        ("EPIC_100_train.csv", "train"),
        ("EPIC_100_validation.csv", "validation"),
    ):
        path = annotation_root / filename
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                video_id = str(row.get("video_id") or "").strip()
                if not video_id:
                    continue
                try:
                    start_frame = int(row.get("start_frame", ""))
                    stop_frame = int(row.get("stop_frame", ""))
                except ValueError:
                    continue
                label = _epic_segment_label(row, label_mode)
                if not label:
                    continue
                by_video.setdefault(video_id, []).append(
                    AnnotationSegment(max(1, start_frame), max(start_frame, stop_frame), label)
                )
                split_by_video[video_id] = split_name
    return (
        {
            video_id: sorted(segments, key=lambda segment: (segment.start_frame, segment.end_frame))
            for video_id, segments in by_video.items()
        },
        split_by_video,
    )


def _epic_segment_label(row: Mapping[str, str], label_mode: str) -> str:
    if label_mode == "verb":
        return str(row.get("verb") or "").strip()
    if label_mode == "noun":
        return str(row.get("noun") or "").strip()
    if label_mode == "verb_class":
        return str(row.get("verb_class") or "").strip()
    if label_mode == "noun_class":
        return str(row.get("noun_class") or "").strip()
    if label_mode == "narration":
        return str(row.get("narration") or "").strip()
    verb = str(row.get("verb") or "").strip()
    noun = str(row.get("noun") or "").strip()
    return " ".join(part for part in (verb, noun) if part).strip()


def _epic_activity_label_components(
    activity_names: Sequence[str],
    label_mode: str,
) -> List[Dict[str, str]]:
    if label_mode != "action":
        return []
    components: List[Dict[str, str]] = []
    for label in activity_names:
        if str(label) == "SIL":
            components.append({"verb": "SIL", "noun": "SIL"})
            continue
        verb, _, noun = str(label).partition(" ")
        components.append({"verb": verb.strip(), "noun": noun.strip()})
    return components


def _epic_video_id_from_path(video_path: str) -> str:
    return Path(video_path).stem


def _resolve_epic_video_split(
    train_video_ids: Sequence[str],
    validation_video_ids: Sequence[str],
    test_split: str | Mapping[str, Sequence[str]],
) -> Dict[str, List[str]]:
    if isinstance(test_split, Mapping):
        required = {"train", "val", "test"}
        missing = required - set(test_split)
        if missing:
            raise ValueError(f"Split mapping is missing keys: {sorted(missing)}")
        return {key: [str(item) for item in test_split[key]] for key in ("train", "val", "test")}

    split_key = str(test_split).strip().lower().replace("-", "_")
    if split_key not in {"s1", "split1", "official", "train_holdout", "train_val_test"}:
        raise ValueError(
            "EPIC-KITCHENS supports split 's1'/'official' as train-holdout plus official "
            "validation-as-test, or an explicit train/val/test mapping."
        )
    ordered_train = sorted(set(str(video_id) for video_id in train_video_ids))
    ordered_train = sorted(ordered_train, key=lambda value: sha1(value.encode("utf-8")).hexdigest())
    official_validation = sorted(set(str(video_id) for video_id in validation_video_ids))
    if len(ordered_train) < 2:
        raise ValueError("EPIC-KITCHENS needs at least two official-train videos to create train/val.")
    if not official_validation:
        raise ValueError("EPIC-KITCHENS official validation videos are required for the repo test split.")
    n_val = max(1, int(round(0.15 * len(ordered_train))))
    n_train = max(1, len(ordered_train) - n_val)
    return {
        "train": sorted(ordered_train[:n_train]),
        "val": sorted(ordered_train[n_train:]),
        "test": official_validation,
    }


def _gtea_segment_label(
    raw_segment: Mapping[str, object],
    *,
    action_label_index: Mapping[int, str] | None,
    split_key: str,
    label_mode: str,
) -> str:
    if label_mode == "verb":
        return str(raw_segment.get("verb_label") or "").strip()
    if label_mode == "noun":
        noun_text = str(raw_segment.get("noun_label_text") or "").strip()
        if noun_text:
            return noun_text
        nouns = raw_segment.get("noun_labels")
        if isinstance(nouns, Sequence) and not isinstance(nouns, (str, bytes)):
            return ",".join(str(noun).strip() for noun in nouns if str(noun).strip())
        return ""
    if label_mode == "coarse_noun":
        noun_text = str(raw_segment.get("noun_label_text") or "").strip()
        if noun_text:
            return _coarse_gtea_noun_label(noun_text)
        nouns = raw_segment.get("noun_labels")
        if isinstance(nouns, Sequence) and not isinstance(nouns, (str, bytes)):
            for noun in nouns:
                coarse = _coarse_gtea_noun_label(str(noun))
                if coarse != "other_object":
                    return coarse
            return "other_object"
        return ""
    if label_mode == "raw_action":
        return str(raw_segment.get("action_label") or "").strip()
    return _gtea_action_label(raw_segment, action_label_index=action_label_index, split_key=split_key)


def _gtea_action_label(
    raw_segment: Mapping[str, object],
    *,
    action_label_index: Mapping[int, str] | None,
    split_key: str,
) -> str:
    action_id = _gtea_action_id(raw_segment, split_key)
    if action_label_index and action_id is not None:
        if action_id - 1 in action_label_index:
            return action_label_index[action_id - 1]
        if action_id in action_label_index:
            return action_label_index[action_id]
    return str(raw_segment.get("action_label") or "").strip()


def _gtea_action_id(raw_segment: Mapping[str, object], split_key: str) -> int | None:
    membership = raw_segment.get("split_membership")
    if isinstance(membership, Mapping):
        split_info = membership.get(split_key)
        if isinstance(split_info, Mapping):
            value = split_info.get("action_id")
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
    value = raw_segment.get("action_id")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _coarse_gtea_noun_label(label: str) -> str:
    key = _clean_gtea_noun_token(label)
    if not key:
        return ""
    first = _clean_gtea_noun_token(re.split(r"[,;/+]", key)[0])
    aliases = {
        "eating utensil": "utensil_tool",
        "cooking utensil": "utensil_tool",
        "paper towel": "cleaning_paper",
        "cutting board": "surface_board",
        "condiment container": "container_package",
        "seasoning container": "container_package",
        "bread container": "container_package",
        "tomato container": "container_package",
        "cheese container": "container_package",
        "oil container": "container_package",
        "pasta container": "container_package",
        "trash container": "container_package",
        "fridge drawer": "appliance_furniture",
    }
    if first in aliases:
        return aliases[first]
    food_tokens = {
        "tomato", "cucumber", "onion", "carrot", "lettuce", "bread", "bacon",
        "patty", "cheese", "bell pepper", "olive", "pasta", "egg", "salad",
        "condiment", "seasoning", "oil", "mixture",
    }
    utensil_tokens = {"knife", "fork", "spoon", "utensil", "sponge"}
    cookware_tokens = {"pan", "pot", "bowl", "plate", "cup", "dishwasher"}
    appliance_tokens = {"fridge", "cabinet", "drawer", "faucet", "stove"}
    cleaning_tokens = {"trash", "paper", "towel", "hand"}
    container_tokens = {"container", "box", "bag", "cap"}
    if first in food_tokens or any(token in first for token in food_tokens):
        return "food_ingredient"
    if first in cookware_tokens or any(token in first for token in cookware_tokens):
        return "cookware_dishware"
    if first in utensil_tokens or any(token in first for token in utensil_tokens):
        return "utensil_tool"
    if first in appliance_tokens or any(token in first for token in appliance_tokens):
        return "appliance_furniture"
    if first in cleaning_tokens or any(token in first for token in cleaning_tokens):
        return "cleaning_paper"
    if first in container_tokens or any(token in first for token in container_tokens):
        return "container_package"
    return "other_object"


def _clean_gtea_noun_token(value: str) -> str:
    cleaned = str(value).strip().replace("_", " ").replace("-", " ").lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _load_mpii_annotation_segments(
    annotation_root: Path,
    *,
    label_mode: str,
) -> Dict[str, List[AnnotationSegment]]:
    mat_path = annotation_root / "attributesAnnotations_MPII-Cooking-2.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"MPII annotation MAT file not found: {mat_path}")
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "MPII Cooking 2 annotations are stored as a MATLAB v7.3/HDF5 file. "
            "Install h5py to read attributesAnnotations_MPII-Cooking-2.mat."
        ) from error

    label_field = "activity" if label_mode in {"activity", "coarse_activity"} else "className"
    by_video: Dict[str, List[AnnotationSegment]] = {}
    with h5py.File(mat_path, "r") as handle:
        annos = handle["annos"]
        starts = np.asarray(annos["startFrame"][...]).reshape(-1)
        ends = np.asarray(annos["endFrame"][...]).reshape(-1)
        file_refs = np.asarray(annos["fileName"][...], dtype=object).reshape(-1)
        label_refs = np.asarray(annos[label_field][...], dtype=object).reshape(-1)
        for start, end, file_ref, label_ref in zip(starts, ends, file_refs, label_refs):
            video_id = _hdf5_matlab_string(handle, file_ref)
            label = _clean_mpii_label(_hdf5_matlab_string(handle, label_ref))
            if label_mode == "coarse_activity":
                label = _coarse_mpii_activity_label(label)
            if not video_id or not label:
                continue
            segment = AnnotationSegment(int(round(float(start))), int(round(float(end))), label)
            by_video.setdefault(video_id, []).append(segment)
            by_video.setdefault(_mpii_sequence_id(video_id), []).append(segment)
    return {
        video_id: sorted(segments, key=lambda segment: (segment.start_frame, segment.end_frame))
        for video_id, segments in by_video.items()
    }


def _hdf5_matlab_string(handle, ref) -> str:
    if not ref:
        return ""
    dataset = handle[ref]
    values = dataset[...]
    if values.dtype.kind in {"u", "i"}:
        return "".join(chr(int(value)) for value in values.ravel(order="F") if int(value) != 0)
    if values.dtype == object:
        return " ".join(_hdf5_matlab_string(handle, item) for item in values.ravel(order="F")).strip()
    return str(values)


def _clean_mpii_label(value: str) -> str:
    cleaned = str(value).strip().replace("-", " ")
    cleaned = re.sub(r"V\b", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _coarse_mpii_activity_label(label: str) -> str:
    key = _clean_mpii_label(label).lower()
    exact = {
        "add": "add_transfer",
        "fill": "add_transfer",
        "pour": "add_transfer",
        "sprinkle": "add_transfer",
        "throw in garbage": "discard_cleanup",
        "wash": "wash_clean",
        "clean": "wash_clean",
        "dry": "wash_clean",
        "wipe": "wash_clean",
        "cut": "cut_slice",
        "cut apart": "cut_slice",
        "cut dice": "cut_slice",
        "cut off ends": "cut_slice",
        "cut out inside": "cut_slice",
        "cut stripes": "cut_slice",
        "slice": "cut_slice",
        "chop": "cut_slice",
        "peel": "peel_open_remove",
        "pull apart": "peel_open_remove",
        "remove": "peel_open_remove",
        "remove label": "peel_open_remove",
        "rip off": "peel_open_remove",
        "open": "peel_open_remove",
        "open cap": "peel_open_remove",
        "open tin": "peel_open_remove",
        "take out": "take_put_move",
        "put in": "take_put_move",
        "put on": "take_put_move",
        "move": "take_put_move",
        "place": "take_put_move",
        "arrange": "take_put_move",
        "gather": "take_put_move",
        "stir": "mix_process",
        "mix": "mix_process",
        "knead": "mix_process",
        "puree": "mix_process",
        "whip": "mix_process",
        "squeeze": "press_shape",
        "press": "press_shape",
        "shape": "press_shape",
        "squash": "press_shape",
        "roll": "press_shape",
        "unroll": "press_shape",
        "fold": "press_shape",
        "grate": "tool_process",
        "grind": "tool_process",
        "sharpen": "tool_process",
        "test sharpness": "tool_process",
        "stamp": "tool_process",
        "change temperature": "heat_cook",
        "turn on": "heat_cook",
        "turn off": "heat_cook",
        "turn over": "heat_cook",
        "flip": "heat_cook",
        "lock": "container_fastener",
        "unlock": "container_fastener",
        "put rubber band": "container_fastener",
        "remove rubber band": "container_fastener",
        "hang": "container_fastener",
        "assemble": "assemble_other",
        "apply plaster": "assemble_other",
        "poke": "assemble_other",
    }
    if key in exact:
        return exact[key]
    if any(token in key for token in ("cut", "slice", "chop")):
        return "cut_slice"
    if any(token in key for token in ("wash", "clean", "wipe", "dry")):
        return "wash_clean"
    if any(token in key for token in ("open", "remove", "peel", "pull", "rip")):
        return "peel_open_remove"
    if any(token in key for token in ("take", "put", "move", "place", "arrange")):
        return "take_put_move"
    if any(token in key for token in ("stir", "mix", "knead", "puree", "whip")):
        return "mix_process"
    if any(token in key for token in ("pour", "add", "fill", "sprinkle")):
        return "add_transfer"
    if any(token in key for token in ("press", "squeeze", "shape", "roll", "fold", "squash")):
        return "press_shape"
    if any(token in key for token in ("grate", "grind", "sharpen", "stamp")):
        return "tool_process"
    if any(token in key for token in ("temperature", "turn", "flip")):
        return "heat_cook"
    if any(token in key for token in ("lock", "rubber band", "hang")):
        return "container_fastener"
    return "assemble_other"


def _resolve_mpii_sequence_split(
    annotation_root: Path,
    test_split: str | Mapping[str, Sequence[str]],
) -> Dict[str, List[str]]:
    if isinstance(test_split, Mapping):
        required = {"train", "val", "test"}
        missing = required - set(test_split)
        if missing:
            raise ValueError(f"Split mapping is missing keys: {sorted(missing)}")
        return {key: [str(item) for item in test_split[key]] for key in ("train", "val", "test")}

    split_key = str(test_split).strip().lower().replace("-", "_")
    if split_key in {"s1", "split1", "attr", "attribute", "attributes"}:
        suffix = "Attr"
    elif split_key in {"dish", "dishes"}:
        suffix = "Dishes"
    else:
        raise ValueError(
            "MPII Cooking 2 supports split 'attr'/'s1', split 'dishes', "
            "or an explicit train/val/test mapping."
        )
    split_root = _mpii_experimental_setup_root(annotation_root)
    return {
        "train": _read_mpii_sequence_list(split_root / f"sequencesTrain{suffix}.txt"),
        "val": _read_mpii_sequence_list(split_root / "sequencesVal.txt"),
        "test": _read_mpii_sequence_list(split_root / "sequencesTest.txt"),
    }


def _mpii_experimental_setup_root(annotation_root: Path) -> Path:
    candidates = [
        annotation_root / "experimentalSetup",
        annotation_root / "experimentalSetup" / "experimentalSetup",
    ]
    for candidate in candidates:
        if (candidate / "sequencesTest.txt").exists():
            return candidate
    raise FileNotFoundError(f"MPII experimentalSetup split files not found under {annotation_root}")


def _read_mpii_sequence_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"MPII split file not found: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _mpii_window_labels_from_segments(
    video_path: str,
    segments: Sequence[AnnotationSegment],
    *,
    num_windows: int,
    window_spans: Sequence[Tuple[float, float]] | None,
    fps: float,
    background_label: str | None,
    empty_overlap_strategy: str,
) -> List[str]:
    frame_ranges = _mpii_window_frame_ranges_from_images(video_path, num_windows)
    if frame_ranges is None:
        return _window_labels_from_segments(
            segments,
            num_windows=num_windows,
            window_spans=window_spans,
            fps=fps,
            background_label=background_label,
            empty_overlap_strategy=empty_overlap_strategy,
        )
    return [
        _majority_overlap_label(
            segments,
            start_frame,
            end_frame,
            background_label,
            empty_overlap_strategy=empty_overlap_strategy,
        )
        for start_frame, end_frame in frame_ranges
    ]


def _mpii_window_frame_ranges_from_images(
    video_path: str,
    num_windows: int,
) -> List[Tuple[int, int]] | None:
    path = Path(video_path)
    if not path.is_dir():
        return None
    image_paths = sorted(
        item for item in path.iterdir()
        if item.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths or num_windows <= 0:
        return None
    step = max(1, int(np.ceil(len(image_paths) / float(num_windows))))
    ranges: List[Tuple[int, int]] = []
    for start in range(0, len(image_paths), step):
        chunk = image_paths[start : start + step]
        if not chunk:
            continue
        ranges.append((_frame_number_from_name(chunk[0]), _frame_number_from_name(chunk[-1])))
        if len(ranges) >= int(num_windows):
            break
    if len(ranges) != int(num_windows):
        return None
    return ranges


def _frame_number_from_name(path: Path) -> int:
    match = _FRAME_FILE_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Could not infer frame number from image filename: {path}")
    return int(match.group(1))


def _mpii_video_id_from_path(video_path: str) -> str:
    stem = Path(video_path).stem
    match = _MPII_SEQUENCE_PATTERN.search(stem)
    if match:
        return stem
    return Path(video_path).name


def _mpii_sequence_id(video_id: str) -> str:
    match = _MPII_SEQUENCE_PATTERN.search(str(video_id))
    if match:
        return match.group("sequence")
    return str(video_id).rsplit("-cam-", 1)[0]


def _mpii_subject_from_sequence(sequence_id: str) -> str:
    match = _MPII_SEQUENCE_PATTERN.search(str(sequence_id))
    if match:
        return f"s{match.group('subject')}"
    return str(sequence_id).split("-", 1)[0]


def _mpii_dish_from_sequence(sequence_id: str) -> str:
    match = _MPII_SEQUENCE_PATTERN.search(str(sequence_id))
    if match:
        return f"d{match.group('dish')}"
    parts = str(sequence_id).split("-")
    return parts[1] if len(parts) > 1 else ""


def _resolve_barista_label_root(annotation_root: str | os.PathLike[str] | None) -> Path:
    root = Path(annotation_root) if annotation_root is not None else DEFAULT_BARISTA_LABEL_ROOT
    if (root / "activity_segments.csv").exists():
        return root
    if (root / "labels" / "activity_segments.csv").exists():
        return root / "labels"
    raise FileNotFoundError(f"BARISTA label files not found under {root}")


def _load_barista_annotation_segments(
    label_root: Path,
    *,
    label_mode: str,
) -> Dict[str, List[AnnotationSegment]]:
    classes = _load_barista_activity_classes(label_root)
    segments_path = label_root / "activity_segments.csv"
    if not segments_path.exists():
        raise FileNotFoundError(f"BARISTA segment label file not found: {segments_path}")

    by_sequence: Dict[str, List[AnnotationSegment]] = {}
    with segments_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sequence_id = str(row.get("sequence_id") or "").strip()
            if not sequence_id:
                continue
            try:
                label_index = int(row.get("activity_label_index", ""))
                start_frame = int(row.get("start_frame", ""))
                end_frame = int(row.get("end_frame", ""))
            except ValueError:
                continue
            label = _barista_segment_label(row, label_index, classes, label_mode)
            if not label:
                continue
            # BARISTA segment end_frame is exclusive in the exported CSV.
            by_sequence.setdefault(sequence_id, []).append(
                AnnotationSegment(start_frame, max(start_frame, end_frame - 1), label)
            )
    return {
        sequence_id: sorted(segments, key=lambda segment: (segment.start_frame, segment.end_frame))
        for sequence_id, segments in by_sequence.items()
    }


def _load_barista_frame_labels(
    label_root: Path,
    *,
    label_mode: str,
) -> Dict[str, List[str]]:
    classes = _load_barista_activity_classes(label_root)
    labels_path = label_root / "activity_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"BARISTA frame label file not found: {labels_path}")

    by_sequence: Dict[str, Dict[int, str]] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sequence_id = str(row.get("sequence_id") or "").strip()
            if not sequence_id:
                continue
            try:
                frame_index = int(row.get("source_frame_index", ""))
                label_index = int(row.get("activity_label_index", ""))
            except ValueError:
                continue
            label = _IGNORE_ACTIVITY_LABEL
            if label_index >= 0:
                label = _barista_segment_label(row, label_index, classes, label_mode)
            by_sequence.setdefault(sequence_id, {})[frame_index] = label or _IGNORE_ACTIVITY_LABEL

    dense_by_sequence: Dict[str, List[str]] = {}
    for sequence_id, labels_by_frame in by_sequence.items():
        if not labels_by_frame:
            continue
        max_frame = max(labels_by_frame)
        labels = [
            labels_by_frame.get(frame_index, _IGNORE_ACTIVITY_LABEL)
            for frame_index in range(max_frame + 1)
        ]
        dense_by_sequence[sequence_id] = labels
    return dense_by_sequence


def _load_barista_activity_classes(label_root: Path) -> Dict[int, Dict[str, str]]:
    path = label_root / "activity_classes.csv"
    if not path.exists():
        return {}
    classes: Dict[int, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                index = int(row.get("activity_label_index", ""))
            except ValueError:
                continue
            classes[index] = {str(key): str(value) for key, value in row.items()}
    return classes


def _barista_segment_label(
    row: Mapping[str, object],
    label_index: int,
    classes: Mapping[int, Mapping[str, str]],
    label_mode: str,
) -> str:
    class_row = classes.get(label_index, {})
    if label_mode == "class_id":
        return str(row.get("activity_class_id") or class_row.get("activity_class_id") or "").strip()
    if label_mode == "slug":
        return str(class_row.get("slug") or row.get("display_name") or "").strip()
    if label_mode == "label_index":
        return f"activity_{label_index:03d}"
    return str(row.get("display_name") or class_row.get("display_name") or "").strip()


def _barista_sequence_id_from_path(video_path: str) -> str:
    return Path(video_path).name


def _barista_window_labels_from_segments(
    video_path: str,
    segments: Sequence[AnnotationSegment],
    *,
    num_windows: int,
    window_spans: Sequence[Tuple[float, float]] | None,
    fps: float,
    background_label: str | None,
    empty_overlap_strategy: str,
) -> List[str]:
    frame_ranges = _barista_window_frame_ranges_from_images(video_path, num_windows)
    if frame_ranges is None:
        return _window_labels_from_segments(
            segments,
            num_windows=num_windows,
            window_spans=window_spans,
            fps=fps,
            background_label=background_label,
            empty_overlap_strategy=empty_overlap_strategy,
        )
    return [
        _majority_overlap_label(
            segments,
            start_frame,
            end_frame,
            background_label,
            empty_overlap_strategy=empty_overlap_strategy,
        )
        for start_frame, end_frame in frame_ranges
    ]


def _barista_window_labels_from_frame_labels(
    video_path: str,
    frame_labels: Sequence[str],
    *,
    num_windows: int,
    background_label: str | None,
    empty_overlap_strategy: str,
) -> List[str]:
    frame_ranges = _barista_window_frame_ranges_from_images(video_path, num_windows)
    if frame_ranges is None:
        frame_count = len(frame_labels)
        step = max(1, int(np.ceil(frame_count / float(num_windows))))
        frame_ranges = [
            (start, min(frame_count - 1, start + step - 1))
            for start in range(0, frame_count, step)
        ][:num_windows]
    labels = []
    for start_frame, end_frame in frame_ranges:
        start = max(0, int(start_frame))
        end = min(len(frame_labels) - 1, int(end_frame))
        window = [
            str(label)
            for label in frame_labels[start : end + 1]
            if str(label) != _IGNORE_ACTIVITY_LABEL
        ]
        if window:
            labels.append(_most_common_stable(window))
        elif background_label is not None:
            labels.append(background_label)
        else:
            labels.append(
                _empty_frame_label(
                    frame_labels,
                    start,
                    end,
                    strategy=empty_overlap_strategy,
                )
            )
    return labels


def _most_common_stable(labels: Sequence[str]) -> str:
    counts: Dict[str, int] = {}
    best_label = str(labels[0])
    best_count = 0
    for label in labels:
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > best_count:
            best_label = key
            best_count = counts[key]
    return best_label


def _empty_frame_label(
    frame_labels: Sequence[str],
    start_frame: int,
    end_frame: int,
    *,
    strategy: str,
) -> str:
    strategy = str(strategy).strip().lower()
    previous = [
        str(label)
        for label in frame_labels[: max(0, start_frame)]
        if str(label) != _IGNORE_ACTIVITY_LABEL
    ]
    next_labels = [
        str(label)
        for label in frame_labels[min(len(frame_labels), end_frame + 1) :]
        if str(label) != _IGNORE_ACTIVITY_LABEL
    ]
    if strategy == "previous" and previous:
        return previous[-1]
    if strategy == "next" and next_labels:
        return next_labels[0]
    if strategy == "nearest":
        prev_index = None
        for index in range(min(start_frame, len(frame_labels)) - 1, -1, -1):
            if str(frame_labels[index]) != _IGNORE_ACTIVITY_LABEL:
                prev_index = index
                break
        next_index = None
        for index in range(max(end_frame + 1, 0), len(frame_labels)):
            if str(frame_labels[index]) != _IGNORE_ACTIVITY_LABEL:
                next_index = index
                break
        if prev_index is not None and next_index is not None:
            return (
                str(frame_labels[prev_index])
                if start_frame - prev_index <= next_index - end_frame
                else str(frame_labels[next_index])
            )
        if prev_index is not None:
            return str(frame_labels[prev_index])
        if next_index is not None:
            return str(frame_labels[next_index])
    if previous:
        return previous[-1]
    if next_labels:
        return next_labels[0]
    return _IGNORE_ACTIVITY_LABEL


def _barista_window_frame_ranges_from_images(
    video_path: str,
    num_windows: int,
) -> List[Tuple[int, int]] | None:
    path = Path(video_path)
    if not path.is_dir():
        return None
    image_paths = sorted(
        item for item in path.iterdir()
        if item.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths or num_windows <= 0:
        return None
    step = max(1, int(np.ceil(len(image_paths) / float(num_windows))))
    ranges: List[Tuple[int, int]] = []
    for start in range(0, len(image_paths), step):
        chunk = image_paths[start : start + step]
        if not chunk:
            continue
        ranges.append((
            max(0, _frame_number_from_name(chunk[0]) - 1),
            max(0, _frame_number_from_name(chunk[-1]) - 1),
        ))
        if len(ranges) >= int(num_windows):
            break
    if len(ranges) != int(num_windows):
        return None
    return ranges


def _resolve_barista_sequence_split(
    sequence_ids: Sequence[str],
    test_split: str | Mapping[str, Sequence[str]],
) -> Dict[str, List[str]]:
    unique_sequences = sorted(set(str(sequence_id) for sequence_id in sequence_ids))
    if isinstance(test_split, Mapping):
        required = {"train", "val", "test"}
        missing = required - set(test_split)
        if missing:
            raise ValueError(f"Split mapping is missing keys: {sorted(missing)}")
        return {key: [str(item) for item in test_split[key]] for key in ("train", "val", "test")}

    split_key = str(test_split).strip().lower()
    if split_key not in {"s1", "split1", "1", "sequence", "sequence_hash"}:
        raise ValueError(
            "BARISTA supports split 's1' as a deterministic sequence-level split, "
            "or an explicit train/val/test mapping."
        )
    ordered = sorted(unique_sequences, key=lambda value: sha1(value.encode("utf-8")).hexdigest())
    n_sequences = len(ordered)
    n_test = max(1, int(round(0.15 * n_sequences)))
    n_val = max(1, int(round(0.15 * n_sequences)))
    n_train = max(1, n_sequences - n_val - n_test)
    return {
        "train": sorted(ordered[:n_train]),
        "val": sorted(ordered[n_train : n_train + n_val]),
        "test": sorted(ordered[n_train + n_val :]),
    }


def _breakfast_annotation_key(video_path: str) -> str:
    path = Path(video_path)
    subject = path.parts[-3]
    camera = path.parts[-2]
    stem = path.stem
    if stem.endswith("_ch0") or stem.endswith("_ch1"):
        stem = stem.rsplit("_", 1)[0]
    camera_token = "stereo01" if camera == "stereo" else camera
    return f"{subject}_{camera_token}_{stem}"


def _subject_from_path(video_path: str) -> str:
    match = _SUBJECT_PATTERN.search(video_path)
    if match:
        return match.group(1)
    return Path(video_path).parts[-3]


def _gtea_subject_from_session(session_id: str) -> str:
    match = _GTEA_SUBJECT_PATTERN.match(session_id)
    if match:
        return match.group("subject")
    return session_id.split("-", 1)[0]


def _resolve_breakfast_split(test_split: str | Mapping[str, Sequence[str]]) -> Dict[str, List[str]]:
    if isinstance(test_split, Mapping):
        required = {"train", "val", "test"}
        missing = required - set(test_split)
        if missing:
            raise ValueError(f"Split mapping is missing keys: {sorted(missing)}")
        return {key: [str(item) for item in test_split[key]] for key in ("train", "val", "test")}
    split_key = str(test_split).strip().lower().replace("-", "_")
    if split_key == "s1":
        return {key: list(value) for key, value in BREAKFAST_S1_SPLIT.items()}
    for benchmark_split, test_participants in BREAKFAST_BENCHMARK_TEST_PARTICIPANTS.items():
        if split_key not in {f"official_{benchmark_split}", f"benchmark_{benchmark_split}"}:
            continue
        all_participants = {
            f"P{participant:02d}"
            for participant in range(3, 55)
        }
        train_val_participants = all_participants - set(test_participants)
        ordered_train_val = sorted(
            train_val_participants,
            key=lambda value: sha1(value.encode("utf-8")).hexdigest(),
        )
        n_val = max(1, int(round(0.15 * len(ordered_train_val))))
        return {
            "train": sorted(ordered_train_val[:-n_val]),
            "val": sorted(ordered_train_val[-n_val:]),
            "test": list(test_participants),
        }
    raise ValueError(
        "Breakfast supports legacy split 's1', benchmark splits "
        "'official_s1' through 'official_s4', or an explicit train/val/test mapping."
    )


def _gtea_split_membership_key(test_split: str | Mapping[str, Sequence[str]]) -> str:
    if isinstance(test_split, Mapping):
        return "split1"
    value = str(test_split).strip().lower()
    if value in {"s1", "split1", "1", "session", "session_hash"}:
        return "split1"
    if value in {"s2", "split2", "2"}:
        return "split2"
    if value in {"s3", "split3", "3"}:
        return "split3"
    return "split1"


def _load_gtea_action_label_index(annotation_root: Path) -> Dict[int, str]:
    path = annotation_root / "cls_label_index.csv"
    if not path.exists():
        return {}
    labels: Dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped.split(";")]
        if len(parts) < 2:
            continue
        try:
            action_id = int(parts[0])
        except ValueError:
            continue
        labels[action_id] = parts[1]
    return labels


def _resolve_gtea_session_split(
    session_ids: Sequence[str],
    test_split: str | Mapping[str, Sequence[str]],
) -> Dict[str, List[str]]:
    unique_sessions = sorted(set(str(session_id) for session_id in session_ids))
    if isinstance(test_split, Mapping):
        required = {"train", "val", "test"}
        missing = required - set(test_split)
        if missing:
            raise ValueError(f"Split mapping is missing keys: {sorted(missing)}")
        return {key: [str(item) for item in test_split[key]] for key in ("train", "val", "test")}

    split_key = str(test_split).strip().lower()
    if split_key not in {"s1", "split1", "1", "session", "session_hash"}:
        raise ValueError(
            "GTEA Gaze supports split 's1' as a deterministic session-level split, "
            "or an explicit train/val/test mapping."
        )
    ordered = sorted(unique_sessions, key=lambda value: sha1(value.encode("utf-8")).hexdigest())
    n_sessions = len(ordered)
    n_test = max(1, int(round(0.15 * n_sessions)))
    n_val = max(1, int(round(0.15 * n_sessions)))
    n_train = max(1, n_sessions - n_val - n_test)
    return {
        "train": sorted(ordered[:n_train]),
        "val": sorted(ordered[n_train : n_train + n_val]),
        "test": sorted(ordered[n_train + n_val :]),
    }


def _pack_split(sequences: Sequence[WindowSequence]) -> Dict[str, np.ndarray]:
    if not sequences:
        raise ValueError("Cannot pack an empty split.")
    num_videos = len(sequences)
    max_length = max(sequence.length for sequence in sequences)
    num_concepts = int(sequences[0].concepts.shape[1])
    first_raw_features = sequences[0].raw_features if sequences[0].raw_features is not None else sequences[0].concepts
    raw_feature_dim = int(first_raw_features.shape[1])
    has_example_mask = any(sequence.example_mask is not None for sequence in sequences)

    concepts = np.zeros((num_videos, max_length, num_concepts), dtype=np.float32)
    raw_features = np.zeros((num_videos, max_length, raw_feature_dim), dtype=np.float32)
    activity_labels = np.full((num_videos, max_length), -1, dtype=np.int64)
    example_mask = np.zeros((num_videos, max_length), dtype=np.float32) if has_example_mask else None
    mask = np.zeros((num_videos, max_length), dtype=np.float32)
    lengths = np.zeros(num_videos, dtype=np.int64)
    video_ids = []
    video_paths = []
    subjects = []
    recipe_labels = []

    for index, sequence in enumerate(sequences):
        length = sequence.length
        concepts[index, :length] = sequence.concepts
        sequence_raw_features = sequence.raw_features if sequence.raw_features is not None else sequence.concepts
        raw_features[index, :length] = np.asarray(sequence_raw_features, dtype=np.float32)
        activity_labels[index, :length] = sequence.activity_labels
        if example_mask is not None:
            if sequence.example_mask is None:
                example_mask[index, :length] = 1.0
            else:
                example_mask[index, :length] = np.asarray(sequence.example_mask, dtype=np.float32)
        mask[index, :length] = 1.0
        lengths[index] = length
        video_ids.append(sequence.video_id)
        video_paths.append(sequence.video_path)
        subjects.append(sequence.subject)
        recipe_labels.append(sequence.recipe_label)

    packed = {
        "concepts": concepts,
        "raw_features": raw_features,
        "activity_labels": activity_labels,
        "mask": mask,
        "lengths": lengths,
        "video_ids": np.asarray(video_ids),
        "video_paths": np.asarray(video_paths),
        "subjects": np.asarray(subjects),
        "recipe_labels": np.asarray(recipe_labels),
    }
    if example_mask is not None:
        packed["example_mask"] = example_mask
    return CompactArrayDict(packed)


def _synthetic_int(params: Mapping[str, object], key: str, default: int, *, minimum: int = 1) -> int:
    value = int(params.get(key, default))
    if value < minimum:
        raise ValueError(f"synthetic_delayed_edge {key} must be >= {minimum}.")
    return value


def _prepare_synthetic_delayed_edge(
    *,
    dataset_hparams: Mapping[str, object],
    verbose: bool,
) -> Dict[str, object]:
    params = dict(dataset_hparams or {})
    variant = str(params.get("variant", "event_memory")).strip().lower()
    nonlinear_variant = variant in {"nonlinear", "nonlinear_event_memory", "rich_event_memory_v3", "event_memory_v3", "v3"}
    rich_variant = nonlinear_variant or variant in {"rich", "rich_event_memory", "event_memory_v2", "v2"}
    seed = _synthetic_int(params, "seed", 42, minimum=0)
    num_sources = _synthetic_int(params, "num_sources", 4, minimum=1)
    num_memory_slots = _synthetic_int(params, "num_memory_slots", num_sources, minimum=1)
    num_label_bits = _synthetic_int(params, "num_label_bits", 4 if nonlinear_variant else num_sources, minimum=1)
    num_phase_concepts = _synthetic_int(params, "num_phase_concepts", 4 if rich_variant else 0, minimum=0)
    min_concepts = num_sources + num_memory_slots + num_phase_concepts + 2
    num_concepts = _synthetic_int(params, "num_concepts", max(20, min_concepts), minimum=min_concepts)
    sequence_length = _synthetic_int(params, "sequence_length", 96, minimum=8)
    event_stride = _synthetic_int(params, "event_stride", int(params.get("label_delay", 4)), minimum=1)
    plateau_length = _synthetic_int(params, "plateau_length", 4, minimum=1)
    split_sizes = {
        "train": _synthetic_int(params, "train_sequences", 768, minimum=1),
        "val": _synthetic_int(params, "val_sequences", 192, minimum=1),
        "test": _synthetic_int(params, "test_sequences", 192, minimum=1),
    }
    noise_scale = float(params.get("noise_scale", 0.15))
    distractor_ar = float(params.get("distractor_ar", 0.2))
    source_low = float(params.get("source_low", 0.15 if rich_variant else 0.0))
    source_high = float(params.get("source_high", 0.85 if rich_variant else 1.0))
    source_signal_noise = float(params.get("source_signal_noise", 0.08 if rich_variant else 0.0))
    source_background_noise = float(params.get("source_background_noise", 0.05 if rich_variant else 0.0))
    source_dropout_rate = float(params.get("source_dropout_rate", 0.05 if rich_variant else 0.0))
    memory_noise_scale = float(params.get("memory_noise_scale", 0.03 if rich_variant else 0.0))
    phase_noise_scale = float(params.get("phase_noise_scale", 0.05 if rich_variant else 0.0))
    spurious_distractors = _synthetic_int(params, "spurious_distractors", 4 if rich_variant else 0, minimum=0)
    spurious_strength = float(params.get("spurious_strength", 0.35 if rich_variant else 0.0))
    spurious_test_strength = float(params.get("spurious_test_strength", 0.05 if rich_variant else spurious_strength))
    class_zipf_exponent = float(params.get("class_zipf_exponent", 0.35 if rich_variant else 0.0))
    label_noise_rate = float(params.get("label_noise_rate", 0.02 if rich_variant else 0.0))
    event_jitter = _synthetic_int(params, "event_jitter", 1 if rich_variant else 0, minimum=0)
    if not 0.0 <= distractor_ar < 1.0:
        raise ValueError("synthetic_delayed_edge distractor_ar must be in [0, 1).")
    if noise_scale < 0.0:
        raise ValueError("synthetic_delayed_edge noise_scale must be >= 0.")
    for key, value in {
        "source_signal_noise": source_signal_noise,
        "source_background_noise": source_background_noise,
        "memory_noise_scale": memory_noise_scale,
        "phase_noise_scale": phase_noise_scale,
    }.items():
        if value < 0.0:
            raise ValueError(f"synthetic_delayed_edge {key} must be >= 0.")
    for key, value in {
        "source_dropout_rate": source_dropout_rate,
        "label_noise_rate": label_noise_rate,
    }.items():
        if not 0.0 <= value < 1.0:
            raise ValueError(f"synthetic_delayed_edge {key} must be in [0, 1).")
    if plateau_length + 1 > sequence_length:
        raise ValueError("synthetic_delayed_edge plateau_length + 1 must fit into sequence_length.")
    if num_sources > 8:
        raise ValueError("synthetic_delayed_edge num_sources must be <= 8 to keep class count manageable.")
    if num_label_bits > 8:
        raise ValueError("synthetic_delayed_edge num_label_bits must be <= 8 to keep class count manageable.")
    if nonlinear_variant and num_sources < 6:
        raise ValueError("synthetic_delayed_edge nonlinear_event_memory requires num_sources >= 6.")
    if event_stride < plateau_length:
        raise ValueError("synthetic_delayed_edge event_stride must be >= plateau_length.")

    memory_start = num_sources
    phase_start = memory_start + num_memory_slots
    distractor_start = phase_start + num_phase_concepts
    spurious_distractors = min(spurious_distractors, max(num_concepts - distractor_start, 0))
    concept_names = (
        [f"event_source_bit_{idx}" for idx in range(num_sources)]
        + [f"graph_memory_slot_{idx}" for idx in range(num_memory_slots)]
        + [f"event_phase_{idx}" for idx in range(num_phase_concepts)]
        + [f"distractor_{idx}" for idx in range(num_concepts - distractor_start)]
    )
    num_activities = 2 ** num_label_bits
    activity_names = [f"event_bits_{value:0{num_label_bits}b}" for value in range(num_activities)]
    source_class_weights = (2 ** np.arange(num_sources - 1, -1, -1, dtype=np.int64)).reshape(1, -1)
    label_class_weights = (2 ** np.arange(num_label_bits - 1, -1, -1, dtype=np.int64)).reshape(1, -1)
    class_mask = int(num_activities - 1)
    if rich_variant and class_zipf_exponent > 0.0:
        class_probs = 1.0 / np.power(np.arange(1, num_activities + 1, dtype=np.float64), class_zipf_exponent)
        class_probs = class_probs[np.random.default_rng(seed + 17).permutation(num_activities)]
        class_probs = class_probs / class_probs.sum()
    else:
        class_probs = None
    if class_probs is not None and num_label_bits != num_sources:
        class_probs = None

    rng = np.random.default_rng(seed)

    def label_to_bits(label: int) -> np.ndarray:
        return ((int(label) >> np.arange(num_label_bits - 1, -1, -1)) & 1).astype(np.int64).reshape(1, -1)

    def nonlinear_label_from_bits(bits: np.ndarray) -> int:
        flat = np.asarray(bits, dtype=np.int64).reshape(-1)
        label_bits = np.asarray(
            [
                flat[0] ^ flat[1],
                flat[2] ^ flat[3],
                flat[4] ^ flat[5],
                (flat[0] & flat[2]) ^ (flat[1] & flat[4]) ^ flat[3],
            ],
            dtype=np.int64,
        )
        if num_label_bits > 4:
            extra = [
                flat[(idx + 1) % num_sources] ^ (flat[idx % num_sources] & flat[(idx + 2) % num_sources])
                for idx in range(num_label_bits - 4)
            ]
            label_bits = np.concatenate([label_bits, np.asarray(extra, dtype=np.int64)])
        return int((label_bits.reshape(1, -1) * label_class_weights).sum(axis=1)[0])

    def transition_label(label: int, offset: int) -> int:
        if not rich_variant:
            return int(label)
        shift = int(offset) % int(num_label_bits)
        rotated = ((int(label) << shift) | (int(label) >> (num_label_bits - shift))) & class_mask
        xor_masks = (0, 0b0011, 0b0101, 0b1001, 0b0110, 0b1010)
        return int(rotated ^ (xor_masks[int(offset) % len(xor_masks)] & class_mask))

    sequences_by_split: Dict[str, List[WindowSequence]] = {"train": [], "val": [], "test": []}
    for split_name, count in split_sizes.items():
        for sequence_idx in range(count):
            concepts = np.zeros((sequence_length, num_concepts), dtype=np.float32)
            if rich_variant and source_background_noise > 0.0:
                concepts[:, :num_sources] = rng.normal(
                    0.0,
                    source_background_noise,
                    size=(sequence_length, num_sources),
                ).astype(np.float32)
            if rich_variant and memory_noise_scale > 0.0:
                concepts[:, memory_start : memory_start + num_memory_slots] = rng.normal(
                    0.0,
                    memory_noise_scale,
                    size=(sequence_length, num_memory_slots),
                ).astype(np.float32)
            if num_concepts > distractor_start:
                distractors = rng.normal(0.0, noise_scale, size=(sequence_length, num_concepts - distractor_start)).astype(np.float32)
                for timestep in range(1, sequence_length):
                    distractors[timestep] += float(distractor_ar) * distractors[timestep - 1]
                concepts[:, distractor_start:] = distractors

            labels = np.full(sequence_length, -1, dtype=np.int64)
            example_mask = np.zeros(sequence_length, dtype=np.float32)
            for event_time in range(0, sequence_length - plateau_length, event_stride):
                if rich_variant and event_jitter > 0:
                    jitter = int(rng.integers(-event_jitter, event_jitter + 1))
                    event_time = int(np.clip(event_time + jitter, 0, sequence_length - plateau_length - 1))
                anchor_time = event_time + 1
                if anchor_time + plateau_length > sequence_length:
                    continue
                if class_probs is None:
                    source_bits = rng.integers(0, 2, size=(1, num_sources), dtype=np.int64)
                    label = (
                        nonlinear_label_from_bits(source_bits)
                        if nonlinear_variant
                        else int((source_bits * source_class_weights).sum(axis=1)[0])
                    )
                else:
                    label = int(rng.choice(num_activities, p=class_probs))
                    source_bits = label_to_bits(label)
                source_values = source_low + (source_high - source_low) * source_bits.astype(np.float32)
                if source_signal_noise > 0.0:
                    source_values = source_values + rng.normal(0.0, source_signal_noise, size=source_values.shape)
                if source_dropout_rate > 0.0:
                    keep = rng.random(size=source_values.shape) >= source_dropout_rate
                    dropped = rng.normal(0.0, source_background_noise, size=source_values.shape)
                    source_values = np.where(keep, source_values, dropped)
                concepts[event_time, :num_sources] = source_values.astype(np.float32)
                for offset in range(plateau_length):
                    timestep = anchor_time + offset
                    step_label = transition_label(label, offset)
                    if label_noise_rate > 0.0 and rng.random() < label_noise_rate:
                        step_label = int(rng.integers(0, num_activities))
                    labels[timestep] = step_label
                    if num_phase_concepts > 0:
                        concepts[timestep, phase_start : phase_start + num_phase_concepts] = rng.normal(
                            0.0,
                            phase_noise_scale,
                            size=(num_phase_concepts,),
                        ).astype(np.float32)
                        concepts[timestep, phase_start + (offset % num_phase_concepts)] += 1.0
                    if spurious_distractors > 0:
                        strength = spurious_test_strength if split_name == "test" else spurious_strength
                        bits = label_to_bits(step_label).reshape(-1)
                        for idx in range(spurious_distractors):
                            bit = float(bits[idx % num_label_bits])
                            concepts[timestep, distractor_start + idx] += strength * (2.0 * bit - 1.0)
                example_mask[anchor_time] = 1.0

            sequences_by_split[split_name].append(
                WindowSequence(
                    video_id=f"{split_name}_{sequence_idx:05d}",
                    video_path=f"synthetic_delayed_edge/{split_name}/{sequence_idx:05d}",
                    subject=split_name,
                    recipe_label="synthetic",
                    concepts=concepts,
                    activity_labels=labels,
                    example_mask=example_mask,
                )
            )

    if verbose:
        print(
            "[prepare_data] Generated synthetic_delayed_edge "
            f"splits={split_sizes} length={sequence_length} concepts={num_concepts} "
            f"sources={num_sources} memory_slots={num_memory_slots} "
            f"event_stride={event_stride} plateau_length={plateau_length} variant={variant}",
            flush=True,
        )

    return CompactArrayDict({
        "train": _pack_split(sequences_by_split["train"]),
        "val": _pack_split(sequences_by_split["val"]),
        "test": _pack_split(sequences_by_split["test"]),
        "metadata": {
            "dataset": "synthetic_delayed_edge",
            "num_activities": num_activities,
            "activity_names": activity_names,
            "concept_names": concept_names,
            "concept_activation_mode": "continuous",
            "binary_concepts": False,
            "synthetic_task": (
                "nonlinear rich event-memory: labels are XOR/AND motifs over noisy source pulses, "
                "with shifted forecast labels, distractors, and supervised anchors requiring graph memory"
                if nonlinear_variant
                else "rich event-memory: noisy source pulses, shifted forecast labels, distractors, "
                "and supervised anchors require graph memory"
                if rich_variant
                else "event-memory: supervised anchors at t=e+1 require graph memory from "
                "source_bits[e] for both activity and forecast labels"
            ),
            "synthetic_hparams": {
                "variant": variant,
                "seed": seed,
                "num_sources": num_sources,
                "num_memory_slots": num_memory_slots,
                "num_label_bits": num_label_bits,
                "num_phase_concepts": num_phase_concepts,
                "num_concepts": num_concepts,
                "sequence_length": sequence_length,
                "event_stride": event_stride,
                "plateau_length": plateau_length,
                **split_sizes,
                "noise_scale": noise_scale,
                "distractor_ar": distractor_ar,
                "source_low": source_low,
                "source_high": source_high,
                "source_signal_noise": source_signal_noise,
                "source_background_noise": source_background_noise,
                "source_dropout_rate": source_dropout_rate,
                "memory_noise_scale": memory_noise_scale,
                "phase_noise_scale": phase_noise_scale,
                "spurious_distractors": spurious_distractors,
                "spurious_strength": spurious_strength,
                "spurious_test_strength": spurious_test_strength,
                "class_zipf_exponent": class_zipf_exponent,
                "label_noise_rate": label_noise_rate,
                "event_jitter": event_jitter,
            },
            "ground_truth_temporal_edges": [
                {"source": int(source), "target": int(memory_start + source), "lag": 1}
                for source in range(min(num_sources, num_memory_slots))
            ],
        },
    })


_DATASET_PREPARERS = {
    "breakfast": _prepare_breakfast,
    "gtea": _prepare_gtea_gaze,
    "gtea_gaze": _prepare_gtea_gaze,
    "egtea": _prepare_gtea_gaze,
    "egtea_gaze": _prepare_gtea_gaze,
    "mpii": _prepare_mpii_cooking_2,
    "mpii_cooking": _prepare_mpii_cooking_2,
    "mpii_cooking_2": _prepare_mpii_cooking_2,
    "mpiicooking2": _prepare_mpii_cooking_2,
    "barista": _prepare_barista,
    "epic": _prepare_epic_kitchens,
    "epic100": _prepare_epic_kitchens,
    "epic_100": _prepare_epic_kitchens,
    "epic_kitchens": _prepare_epic_kitchens,
    "epic_kitchens_100": _prepare_epic_kitchens,
    "synthetic_delayed_edge": _prepare_synthetic_delayed_edge,
}
