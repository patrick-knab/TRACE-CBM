#!/usr/bin/env python3
"""Convert the official BARISTA release into the layout consumed by TRACE.

The official release (https://huggingface.co/datasets/ramblr/BARISTA) stores one
directory per video::

    <document_id>/
        coco_annotation.json   # activities, video_metadata, split, ...
        video.mp4

TRACE embeds image folders and reads frame-level activity labels from CSV files.
This script writes, below ``$TRACE_DATASET_ROOT/Barista`` by default::

    Image_data/<sequence_id>/frame_000001.jpg ...   every source frame, native resolution
    Image_data/<sequence_id>/source_info.json       fps, frame_count, resolution, official split
    labels/activity_classes.csv
    labels/activity_segments.csv                    inclusive 0-based start_frame / end_frame
    labels/activity_labels.csv                      one row per source frame
    splits/s1_hash.json                             the deterministic hash split (``--test-split s1``)
    splits/official.json                            official test set + validation carved from train
    metadata.json

``sequence_id`` is ``<document_id>__video_<video_index>`` (``video_index`` from
``video_metadata``; ``0`` for every released video). Frames are extracted with
ffmpeg at the source frame rate (about 30 FPS) without resampling or resizing,
which is the preparation used for the paper. ``--max-side`` optionally downsizes
frames to save disk space.

Overlapping activity segments: ``activity_labels.csv`` keeps one primary label
per frame (the segment that starts last, ties broken by the longer segment, then
by annotation order) and preserves all overlapping labels in the pipe-separated
``activity_label_indices`` column. Frames outside every activity have
``activity_label_index = -1`` and become ``SIL`` windows in TRACE.

Usage::

    huggingface-cli download ramblr/BARISTA --repo-type dataset --local-dir /path/to/BARISTA
    python utils/prepare_barista.py prepare --source-root /path/to/BARISTA
    python utils/prepare_barista.py validate
    # (re)write only splits/ for an already prepared folder, e.g. from the checked-in official assignment:
    python utils/prepare_barista.py splits --official-split-field configs/splits/barista_official_split_field.json

The nested layout of the pre-release export
(``<document_id>/<video_name>/coco_*_part_*.json`` + ``video.mp4``) is accepted
as well.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.paths import dataset_root  # noqa: E402

OFFICIAL_ANNOTATION_NAME = "coco_annotation.json"
OFFICIAL_VIDEO_NAME = "video.mp4"
SOURCE_INFO_NAME = "source_info.json"
FRAMES_COMPLETE_MARKER = ".frames_complete"
FRAME_FILE_POLICY = "frame_{source_frame_index + 1:06d}.jpg"
SEQUENCE_ID_POLICY = "<document_id>__video_<video_index>, slugified"
DEFAULT_VAL_FRACTION = 0.15
HASH_SPLIT_FRACTIONS = {"val": 0.15, "test": 0.15}

CLASS_FIELDS = ["activity_label_index", "activity_class_id", "display_name", "slug"]
SEGMENT_FIELDS = [
    "sequence_id",
    "video_id",
    "segment_id",
    "start_frame",
    "end_frame",
    "activity_label_index",
    "display_name",
    "activity_class_id",
]
FRAME_LABEL_FIELDS = [
    "sequence_id",
    "video_id",
    "source_frame_index",
    "frame_file",
    "activity_label_index",
    "display_name",
    "activity_class_id",
    "activity_label_indices",
    "display_names",
    "activity_class_ids",
]


# --------------------------------------------------------------------------- helpers
def frame_file_name(source_frame_index: int) -> str:
    return f"frame_{int(source_frame_index) + 1:06d}.jpg"


def slugify(value: str) -> str:
    slug = str(value).strip().lower()
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"[^a-z0-9_.-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown"


def sequence_id_for(document_id: str, video_name: str) -> str:
    return f"{slugify(document_id)}__{slugify(video_name)}"


def _hash_key(value: str) -> str:
    return sha1(value.encode("utf-8")).hexdigest()


def hash_split(sequence_ids: Sequence[str]) -> Dict[str, List[str]]:
    """Deterministic sequence-level split used for ``--test-split s1`` (paper protocol).

    Sequence ids are ordered by their SHA-1 digest; the first ~70% are training,
    the next ~15% validation and the last ~15% test sequences.
    """
    ordered = sorted(set(str(item) for item in sequence_ids), key=_hash_key)
    n_sequences = len(ordered)
    n_test = max(1, int(round(HASH_SPLIT_FRACTIONS["test"] * n_sequences)))
    n_val = max(1, int(round(HASH_SPLIT_FRACTIONS["val"] * n_sequences)))
    n_train = max(1, n_sequences - n_val - n_test)
    return {
        "train": sorted(ordered[:n_train]),
        "val": sorted(ordered[n_train : n_train + n_val]),
        "test": sorted(ordered[n_train + n_val :]),
    }


def official_split(
    split_by_sequence: Mapping[str, Optional[str]],
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> Optional[Dict[str, List[str]]]:
    """Train/val/test mapping that preserves the official test set.

    ``split_by_sequence`` maps sequence ids to the ``split`` field of the official
    annotation files. Validation sequences are carved deterministically (SHA-1
    order) from the official training sequences. Returns ``None`` when no
    sequence carries a split field.
    """
    known = {sid: str(split).strip().lower() for sid, split in split_by_sequence.items() if split}
    if not known:
        return None
    unexpected = sorted({split for split in known.values()} - {"train", "test"})
    if unexpected:
        raise ValueError(f"Unexpected BARISTA split values: {unexpected}")
    test = sorted(sid for sid, split in known.items() if split == "test")
    pool = sorted((sid for sid, split in known.items() if split == "train"), key=_hash_key)
    n_val = int(round(float(val_fraction) * len(pool))) if len(pool) > 1 else 0
    n_val = min(n_val, len(pool) - 1) if len(pool) > 1 else 0
    return {"train": sorted(pool[n_val:]), "val": sorted(pool[:n_val]), "test": test}


# --------------------------------------------------------------------------- source discovery
@dataclass(frozen=True)
class SourceVideo:
    document_id: str
    video_name: Optional[str]  # None for the official flat layout (resolved from video_index)
    video_dir: Path
    annotation_path: Path
    video_path: Path


@dataclass(frozen=True)
class ActivitySegment:
    segment_id: int
    start_frame: int
    end_frame: int  # inclusive
    activity_class_id: str
    display_name: str
    slug: str


@dataclass(frozen=True)
class VideoExport:
    document_id: str
    video_name: str
    video_id: str
    sequence_id: str
    video_path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    length_in_ms: float
    recording_device: str
    split: Optional[str]
    activities: List[ActivitySegment]


def discover_source_videos(source_root: Path) -> List[SourceVideo]:
    source_root = Path(source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"BARISTA source root does not exist: {source_root}")
    videos: List[SourceVideo] = []
    for child in sorted(source_root.iterdir()):
        if not child.is_dir():
            continue
        official = child / OFFICIAL_ANNOTATION_NAME
        if official.exists():
            videos.append(SourceVideo(child.name, None, child, official, child / OFFICIAL_VIDEO_NAME))
            continue
        for sub in sorted(child.iterdir()):
            if not sub.is_dir():
                continue
            candidates = sorted(sub.glob("coco_*_part_*.json"))
            if not candidates and (sub / OFFICIAL_ANNOTATION_NAME).exists():
                candidates = [sub / OFFICIAL_ANNOTATION_NAME]
            if candidates:
                videos.append(SourceVideo(child.name, sub.name, sub, candidates[0], sub / OFFICIAL_VIDEO_NAME))
    if not videos:
        raise FileNotFoundError(
            f"No BARISTA videos found under {source_root}. Expected <document_id>/{OFFICIAL_ANNOTATION_NAME}"
        )
    return videos


def build_video_export(source: SourceVideo) -> VideoExport:
    with source.annotation_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    metadata = data["video_metadata"][0]
    video_index = int(metadata.get("video_index", 0) or 0)
    video_name = source.video_name or f"video_{video_index}"
    document_id = str(metadata.get("document_id") or source.document_id)
    activities: List[ActivitySegment] = []
    for segment_id, activity in enumerate(data.get("activities", [])):
        image_range = activity["image_range"]
        display_name = str(activity.get("display_name", ""))
        activities.append(
            ActivitySegment(
                segment_id=segment_id,
                start_frame=int(image_range["image_id_start"]),
                end_frame=int(image_range["image_id_end"]),
                activity_class_id=str(activity["activity_class_id"]),
                display_name=display_name,
                slug=slugify(display_name),
            )
        )
    recording_device = " ".join(
        str(item)
        for item in (metadata.get("recording_device_type", ""), metadata.get("recording_device_version", ""))
        if item
    )
    split = data.get("split")
    return VideoExport(
        document_id=document_id,
        video_name=video_name,
        video_id=f"{document_id}/{video_name}",
        sequence_id=sequence_id_for(document_id, video_name),
        video_path=source.video_path,
        frame_count=int(metadata["frame_count"]),
        fps=float(metadata.get("fps", 0.0) or 0.0),
        width=int(metadata.get("width", 0) or 0),
        height=int(metadata.get("height", 0) or 0),
        length_in_ms=float(metadata.get("length_in_ms", 0.0) or 0.0),
        recording_device=recording_device,
        split=str(split).strip().lower() if split else None,
        activities=activities,
    )


def build_exports(source_root: Path) -> List[VideoExport]:
    exports = [build_video_export(source) for source in discover_source_videos(source_root)]
    duplicates = [sid for sid, count in Counter(export.sequence_id for export in exports).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate BARISTA sequence ids: {duplicates[:5]}")
    return exports


# --------------------------------------------------------------------------- label tables
def activity_class_rows(exports: Sequence[VideoExport]) -> List[Dict[str, object]]:
    by_class: Dict[str, tuple] = {}
    for export in exports:
        for activity in export.activities:
            by_class.setdefault(activity.activity_class_id, (activity.display_name, activity.slug))
    rows = []
    ordered = sorted(by_class.items(), key=lambda item: (item[1][0].lower(), item[0]))
    for index, (activity_class_id, (display_name, slug)) in enumerate(ordered):
        rows.append(
            {
                "activity_label_index": index,
                "activity_class_id": activity_class_id,
                "display_name": display_name,
                "slug": slug,
            }
        )
    return rows


def _index_by_class(class_rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    return {str(row["activity_class_id"]): int(row["activity_label_index"]) for row in class_rows}


def segment_rows(exports: Sequence[VideoExport], class_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    index_by_class = _index_by_class(class_rows)
    rows = []
    for export in exports:
        for activity in export.activities:
            rows.append(
                {
                    "sequence_id": export.sequence_id,
                    "video_id": export.video_id,
                    "segment_id": activity.segment_id,
                    "start_frame": activity.start_frame,
                    "end_frame": activity.end_frame,
                    "activity_label_index": index_by_class[activity.activity_class_id],
                    "display_name": activity.display_name,
                    "activity_class_id": activity.activity_class_id,
                }
            )
    return rows


def primary_activity(activities: Sequence[ActivitySegment]) -> Optional[ActivitySegment]:
    """Primary label of a frame covered by several segments.

    The most recently started segment wins; ties go to the longer segment and
    then to the later annotation. This is the rule used for the paper export.
    """
    if not activities:
        return None
    return sorted(
        activities,
        key=lambda activity: (
            activity.start_frame,
            -(activity.end_frame - activity.start_frame),
            activity.segment_id,
        ),
    )[-1]


def frame_label_rows(exports: Sequence[VideoExport], class_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    index_by_class = _index_by_class(class_rows)
    rows = []
    for export in exports:
        activities_by_frame: List[List[ActivitySegment]] = [[] for _ in range(export.frame_count)]
        for activity in export.activities:
            start = max(0, activity.start_frame)
            end = min(export.frame_count - 1, activity.end_frame)
            for frame_index in range(start, end + 1):
                activities_by_frame[frame_index].append(activity)
        for frame_index, activities in enumerate(activities_by_frame):
            primary = primary_activity(activities)
            rows.append(
                {
                    "sequence_id": export.sequence_id,
                    "video_id": export.video_id,
                    "source_frame_index": frame_index,
                    "frame_file": frame_file_name(frame_index),
                    "activity_label_index": index_by_class[primary.activity_class_id] if primary else -1,
                    "display_name": primary.display_name if primary else "",
                    "activity_class_id": primary.activity_class_id if primary else "",
                    "activity_label_indices": "|".join(str(index_by_class[a.activity_class_id]) for a in activities),
                    "display_names": "|".join(a.display_name for a in activities),
                    "activity_class_ids": "|".join(a.activity_class_id for a in activities),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_labels(exports: Sequence[VideoExport], output_root: Path) -> Dict[str, object]:
    class_rows = activity_class_rows(exports)
    segments = segment_rows(exports, class_rows)
    frame_labels = frame_label_rows(exports, class_rows)
    labels_root = Path(output_root) / "labels"
    _write_csv(labels_root / "activity_classes.csv", class_rows, CLASS_FIELDS)
    _write_csv(labels_root / "activity_segments.csv", segments, SEGMENT_FIELDS)
    _write_csv(labels_root / "activity_labels.csv", frame_labels, FRAME_LABEL_FIELDS)
    return {
        "num_activity_classes": len(class_rows),
        "num_activity_segments": len(segments),
        "num_frame_label_rows": len(frame_labels),
        "num_multilabel_frame_rows": sum(1 for row in frame_labels if "|" in str(row["activity_label_indices"])),
    }


def write_split_files(
    output_root: Path,
    split_by_sequence: Mapping[str, Optional[str]],
    val_fraction: float,
) -> Dict[str, object]:
    """Write ``splits/s1_hash.json`` and ``splits/official.json`` below ``output_root``.

    ``split_by_sequence`` maps every sequence id to its official split value
    (``"train"``/``"test"``) or ``None`` when unknown. The official field is also
    stored verbatim in ``splits/official_split_field.json`` so the derived
    mapping can be regenerated.
    """
    splits_root = Path(output_root) / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)
    sequence_ids = sorted(split_by_sequence)
    summary: Dict[str, object] = {}

    s1 = hash_split(sequence_ids)
    (splits_root / "s1_hash.json").write_text(json.dumps(s1, indent=2) + "\n", encoding="utf-8")
    summary["s1_hash"] = {name: len(items) for name, items in s1.items()}

    official = official_split(split_by_sequence, val_fraction)
    if official is None:
        print("[prepare_barista] No official split field available; splits/official.json not written.", flush=True)
        summary["official"] = None
    else:
        field = {sid: split_by_sequence[sid] for sid in sequence_ids if split_by_sequence[sid]}
        (splits_root / "official_split_field.json").write_text(json.dumps(field, indent=2) + "\n", encoding="utf-8")
        (splits_root / "official.json").write_text(json.dumps(official, indent=2) + "\n", encoding="utf-8")
        summary["official"] = {name: len(items) for name, items in official.items()}
        summary["official_val_fraction_of_train"] = float(val_fraction)
        summary["official_split_field"] = dict(Counter(field.values()))
    return summary


def write_splits(exports: Sequence[VideoExport], output_root: Path, val_fraction: float) -> Dict[str, object]:
    return write_split_files(output_root, {export.sequence_id: export.split for export in exports}, val_fraction)


def labelled_sequence_ids(dataset_dir: Path) -> List[str]:
    labels_path = Path(dataset_dir) / "labels" / "activity_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"{labels_path} not found; run `prepare_barista.py prepare` first")
    ids = set()
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("sequence_id"):
                ids.add(row["sequence_id"])
    return sorted(ids)


def load_official_split_field(path: Path, sequence_ids: Sequence[str]) -> Dict[str, Optional[str]]:
    """Read ``{document_id or sequence_id: "train"|"test"}`` and key it by sequence id."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must map ids to 'train'/'test'")
    by_document: Dict[str, List[str]] = {}
    for sid in sequence_ids:
        by_document.setdefault(sid.split("__", 1)[0], []).append(sid)
    result: Dict[str, Optional[str]] = {sid: None for sid in sequence_ids}
    unknown = []
    for key, value in raw.items():
        key = str(key)
        targets = [key] if key in result else by_document.get(slugify(key), [])
        if not targets:
            unknown.append(key)
            continue
        for sid in targets:
            result[sid] = str(value).strip().lower() if value else None
    if unknown:
        print(f"[prepare_barista] WARNING: {len(unknown)} id(s) in {path} are not prepared sequences (e.g. {unknown[:3]})", flush=True)
    missing = [sid for sid, value in result.items() if value is None]
    if missing:
        raise ValueError(f"{len(missing)} prepared sequence(s) have no official split value (e.g. {missing[:3]})")
    return result


def rewrite_splits(
    dataset_dir: Path,
    *,
    source_root: Optional[Path] = None,
    official_split_field: Optional[Path] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> Dict[str, object]:
    """Write split files for an already prepared ``Barista`` folder."""
    sequence_ids = labelled_sequence_ids(dataset_dir)
    split_by_sequence: Dict[str, Optional[str]] = {sid: None for sid in sequence_ids}
    if official_split_field is not None:
        split_by_sequence = load_official_split_field(official_split_field, sequence_ids)
    elif source_root is not None:
        for export in build_exports(Path(source_root)):
            if export.sequence_id in split_by_sequence:
                split_by_sequence[export.sequence_id] = export.split
    else:
        candidate = Path(dataset_dir) / "splits" / "official_split_field.json"
        if candidate.exists():
            split_by_sequence = load_official_split_field(candidate, sequence_ids)
    summary = write_split_files(dataset_dir, split_by_sequence, val_fraction)
    print(f"[prepare_barista] splits written to {Path(dataset_dir) / 'splits'}: {summary}", flush=True)
    return summary


# --------------------------------------------------------------------------- frames
def source_info(export: VideoExport, max_side: Optional[int]) -> Dict[str, object]:
    return {
        "sequence_id": export.sequence_id,
        "document_id": export.document_id,
        "video_name": export.video_name,
        "video_id": export.video_id,
        "fps": export.fps,
        "frame_count": export.frame_count,
        "width": export.width,
        "height": export.height,
        "length_in_ms": export.length_in_ms,
        "recording_device": export.recording_device,
        "split": export.split,
        "frame_file_policy": FRAME_FILE_POLICY,
        "resize_max_side": max_side,
    }


def _ffmpeg_command(video_path: Path, output_dir: Path, jpeg_quality: int, max_side: Optional[int]) -> List[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path)]
    if max_side:
        side = int(max_side)
        command += [
            "-vf",
            f"scale='if(gt(iw,ih),min(iw,{side}),-2)':'if(gt(iw,ih),-2,min(ih,{side}))'",
        ]
    command += ["-q:v", str(jpeg_quality), "-start_number", "1", str(output_dir / "frame_%06d.jpg")]
    return command


def extract_one_video(
    export: VideoExport,
    image_root: Path,
    jpeg_quality: int,
    max_side: Optional[int],
    force: bool,
) -> Dict[str, object]:
    output_dir = image_root / export.sequence_id
    marker = output_dir / FRAMES_COMPLETE_MARKER
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SOURCE_INFO_NAME).write_text(json.dumps(source_info(export, max_side), indent=2), encoding="utf-8")
    if marker.exists() and not force:
        extracted = sum(1 for _ in output_dir.glob("*.jpg"))
        return {"status": "skipped", "extracted_frames": extracted, "expected_frames": export.frame_count}
    if not export.video_path.exists():
        raise FileNotFoundError(f"Missing video file: {export.video_path}")
    for path in output_dir.glob("*.jpg"):
        path.unlink()
    subprocess.run(_ffmpeg_command(export.video_path, output_dir, jpeg_quality, max_side), check=True)
    extracted = sum(1 for _ in output_dir.glob("*.jpg"))
    if extracted == 0:
        raise RuntimeError(f"ffmpeg produced zero frames for {export.video_id}")
    marker.write_text(f"{extracted}\n", encoding="utf-8")
    return {"status": "done", "extracted_frames": extracted, "expected_frames": export.frame_count}


def extract_frames(
    exports: Sequence[VideoExport],
    output_root: Path,
    workers: int,
    jpeg_quality: int,
    max_side: Optional[int],
    force: bool,
) -> Dict[str, object]:
    image_root = Path(output_root) / "Image_data"
    image_root.mkdir(parents=True, exist_ok=True)
    failures: List[tuple] = []
    mismatches: Dict[str, Dict[str, int]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(extract_one_video, export, image_root, jpeg_quality, max_side, force): export
            for export in exports
        }
        for index, future in enumerate(as_completed(futures), start=1):
            export = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append((export.video_id, str(exc)))
                print(f"[frames] FAILED {export.video_id}: {exc}", flush=True)
                continue
            if int(result["extracted_frames"]) != int(result["expected_frames"]):
                mismatches[export.sequence_id] = {
                    "extracted_frames": int(result["extracted_frames"]),
                    "annotation_frame_count": int(result["expected_frames"]),
                }
            print(
                f"[frames] {index}/{len(exports)} {result['status']} "
                f"({result['extracted_frames']} frames): {export.sequence_id}",
                flush=True,
            )
    if failures:
        details = "\n".join(f"{video_id}: {error}" for video_id, error in failures[:20])
        raise RuntimeError(f"{len(failures)} video(s) failed during frame extraction:\n{details}")
    if mismatches:
        print(
            f"[frames] WARNING: {len(mismatches)} sequence(s) have a different number of extracted frames "
            "than annotation frame_count; see metadata.json -> frame_count_mismatches.",
            flush=True,
        )
    return {"frame_count_mismatches": mismatches}


# --------------------------------------------------------------------------- metadata
def write_metadata(
    exports: Sequence[VideoExport],
    output_root: Path,
    source_root: Path,
    label_summary: Mapping[str, object],
    split_summary: Mapping[str, object],
    frame_summary: Mapping[str, object],
    *,
    labels_only: bool,
    max_side: Optional[int],
    jpeg_quality: int,
) -> None:
    label_counts: Counter = Counter()
    for export in exports:
        for activity in export.activities:
            label_counts[activity.display_name] += 1
    fps_values = sorted({round(export.fps, 3) for export in exports})
    metadata = {
        "dataset_name": "Barista",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "image_data_root": str(Path(output_root) / "Image_data"),
        "labels_root": str(Path(output_root) / "labels"),
        "splits_root": str(Path(output_root) / "splits"),
        "frame_file_policy": FRAME_FILE_POLICY,
        "sequence_id_policy": SEQUENCE_ID_POLICY,
        "frame_extraction": {
            "tool": "ffmpeg",
            "every_source_frame": True,
            "temporal_resampling": None,
            "resize_max_side": max_side,
            "jpeg_quality": jpeg_quality,
            "source_fps_range": [fps_values[0], fps_values[-1]] if fps_values else None,
        },
        "labels_only": bool(labels_only),
        "num_videos": len(exports),
        "num_frames": sum(export.frame_count for export in exports),
        "label_summary": dict(label_summary),
        "split_summary": dict(split_summary),
        "activity_label_counts": dict(sorted(label_counts.items())),
        "label_schema": {
            "activity_classes.csv": "activity_label_index, activity_class_id, display_name, slug",
            "activity_segments.csv": "one row per activity segment with inclusive 0-based start_frame/end_frame",
            "activity_labels.csv": (
                "one row per source frame; activity_label_index is the primary label "
                "(-1 outside annotated activities), pipe-separated columns preserve overlapping labels"
            ),
        },
        "overlap_policy": "primary label = latest-starting segment; ties: longer segment, then later annotation",
        **dict(frame_summary),
    }
    Path(output_root).mkdir(parents=True, exist_ok=True)
    (Path(output_root) / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def prepare_barista(
    source_root: Path,
    output_root: Path,
    *,
    workers: int = 4,
    jpeg_quality: int = 2,
    max_side: Optional[int] = None,
    force: bool = False,
    labels_only: bool = False,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> List[VideoExport]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    exports = build_exports(source_root)
    print(f"[prepare_barista] {len(exports)} videos found under {source_root}", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    label_summary = write_labels(exports, output_root)
    split_summary = write_splits(exports, output_root, val_fraction)
    frame_summary: Dict[str, object] = {}
    if not labels_only:
        frame_summary = extract_frames(exports, output_root, workers, jpeg_quality, max_side, force)
    write_metadata(
        exports,
        output_root,
        source_root,
        label_summary,
        split_summary,
        frame_summary,
        labels_only=labels_only,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
    )
    return exports


# --------------------------------------------------------------------------- validation
def validate_barista_layout(dataset_dir: Path, *, check_frames: bool = True) -> Dict[str, object]:
    """Check that ``dataset_dir`` (``.../Barista``) is usable by the TRACE BARISTA loader.

    Returns a report with ``problems`` (blocking) and ``warnings``.
    """
    dataset_dir = Path(dataset_dir)
    problems: List[str] = []
    warnings: List[str] = []
    report: Dict[str, object] = {"dataset_dir": str(dataset_dir), "problems": problems, "warnings": warnings}

    labels_root = dataset_dir / "labels"
    labels_path = labels_root / "activity_labels.csv"
    if not labels_path.exists():
        problems.append(f"missing {labels_path} (run utils/prepare_barista.py prepare)")
        return report
    for optional in ("activity_classes.csv", "activity_segments.csv"):
        if not (labels_root / optional).exists():
            warnings.append(f"missing optional {labels_root / optional}")

    rows_by_sequence: Counter = Counter()
    max_index_by_sequence: Dict[str, int] = {}
    label_values: Counter = Counter()
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = {"sequence_id", "source_frame_index", "activity_label_index"} - set(reader.fieldnames or [])
        if missing_fields:
            problems.append(f"{labels_path} lacks columns {sorted(missing_fields)}")
            return report
        for row in reader:
            sid = row["sequence_id"]
            rows_by_sequence[sid] += 1
            index = int(row["source_frame_index"])
            max_index_by_sequence[sid] = max(index, max_index_by_sequence.get(sid, -1))
            label_values[int(row["activity_label_index"])] += 1
    if not rows_by_sequence:
        problems.append(f"{labels_path} has no rows")
        return report
    report["num_sequences"] = len(rows_by_sequence)
    report["num_frame_rows"] = sum(rows_by_sequence.values())
    report["num_activity_classes_in_use"] = len([k for k in label_values if k >= 0])
    report["unlabeled_frame_fraction"] = label_values.get(-1, 0) / max(1, sum(label_values.values()))
    for sid, count in rows_by_sequence.items():
        if max_index_by_sequence[sid] != count - 1:
            problems.append(f"{sid}: source_frame_index is not contiguous from 0 ({count} rows, max {max_index_by_sequence[sid]})")

    image_root = dataset_dir / "Image_data"
    if not image_root.is_dir():
        problems.append(f"missing {image_root} (frame folders; run utils/prepare_barista.py prepare)")
    elif check_frames:
        image_dirs = {p.name for p in image_root.iterdir() if p.is_dir()}
        for sid in sorted(set(rows_by_sequence) - image_dirs):
            problems.append(f"{sid}: labelled sequence has no Image_data/{sid}/ folder")
        for extra in sorted(image_dirs - set(rows_by_sequence)):
            warnings.append(f"Image_data/{extra}/ has no label rows and will be dropped")
        fps_missing = 0
        for sid in sorted(set(rows_by_sequence) & image_dirs):
            names = [n for n in os.listdir(image_root / sid) if n.lower().endswith((".jpg", ".jpeg", ".png"))]
            expected = rows_by_sequence[sid]
            if len(names) != expected:
                problems.append(f"{sid}: {len(names)} frame files but {expected} label rows")
            elif names:
                first, last = min(names), max(names)
                if first != frame_file_name(0) or last != frame_file_name(expected - 1):
                    problems.append(f"{sid}: frame files should run {frame_file_name(0)}..{frame_file_name(expected - 1)}, found {first}..{last}")
            if not (image_root / sid / SOURCE_INFO_NAME).exists():
                fps_missing += 1
        if fps_missing:
            warnings.append(
                f"{fps_missing} sequence folder(s) lack {SOURCE_INFO_NAME}; window timestamps will assume 12 FPS "
                "(labels are frame-index based and unaffected)"
            )

    splits_root = dataset_dir / "splits"
    report["splits"] = {}
    for name in ("s1_hash.json", "official.json"):
        path = splits_root / name
        if not path.exists():
            warnings.append(f"missing {path}" + (" (official test split unavailable)" if name == "official.json" else ""))
            continue
        mapping = json.loads(path.read_text(encoding="utf-8"))
        unknown = sorted({sid for items in mapping.values() for sid in items} - set(rows_by_sequence))
        if unknown:
            problems.append(f"{path}: {len(unknown)} sequence ids are not in activity_labels.csv (e.g. {unknown[:2]})")
        report["splits"][name] = {k: len(v) for k, v in mapping.items()}
    report["ok"] = not problems
    return report


def print_report(report: Mapping[str, object]) -> None:
    print(f"BARISTA layout check: {report['dataset_dir']}")
    for key in ("num_sequences", "num_frame_rows", "num_activity_classes_in_use", "unlabeled_frame_fraction", "splits"):
        if key in report:
            value = report[key]
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    for warning in report.get("warnings", []):
        print(f"  WARNING {warning}")
    for problem in report.get("problems", []):
        print(f"  PROBLEM {problem}")
    print("  OK" if report.get("ok") else "  FAILED")


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Convert the official BARISTA release into the TRACE layout.")
    prep.add_argument("--source-root", type=Path, required=True, help="Downloaded ramblr/BARISTA directory.")
    prep.add_argument("--output-root", type=Path, default=dataset_root() / "Barista")
    prep.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    prep.add_argument("--jpeg-quality", type=int, default=2, help="ffmpeg -q:v (2 = high quality, paper setting).")
    prep.add_argument("--max-side", type=int, default=None, help="Optionally downscale frames (paper: none).")
    prep.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION, help="Share of official train sequences used as validation in splits/official.json.")
    prep.add_argument("--force", action="store_true", help="Re-extract frames even when a folder is complete.")
    prep.add_argument("--labels-only", action="store_true", help="Write labels, splits and metadata without extracting frames.")

    spl = sub.add_parser("splits", help="(Re)write splits/ for an already prepared Barista directory.")
    spl.add_argument("--dataset-dir", type=Path, default=dataset_root() / "Barista")
    spl.add_argument("--source-root", type=Path, default=None, help="Official download; the split field is read from coco_annotation.json.")
    spl.add_argument(
        "--official-split-field",
        type=Path,
        default=None,
        help="JSON {document_id or sequence_id: 'train'|'test'} (e.g. configs/splits/barista_official_split_field.json).",
    )
    spl.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)

    val = sub.add_parser("validate", help="Check an existing Barista directory.")
    val.add_argument("--dataset-dir", type=Path, default=dataset_root() / "Barista")
    val.add_argument("--skip-frames", action="store_true", help="Do not count frame files.")
    val.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare_barista(
            args.source_root,
            args.output_root,
            workers=args.workers,
            jpeg_quality=args.jpeg_quality,
            max_side=args.max_side,
            force=args.force,
            labels_only=args.labels_only,
            val_fraction=args.val_fraction,
        )
        report = validate_barista_layout(args.output_root, check_frames=not args.labels_only)
        print_report(report)
        return 0 if report.get("ok") else 1
    if args.command == "splits":
        rewrite_splits(
            args.dataset_dir,
            source_root=args.source_root,
            official_split_field=args.official_split_field,
            val_fraction=args.val_fraction,
        )
        return 0
    report = validate_barista_layout(args.dataset_dir, check_frames=not args.skip_frames)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
