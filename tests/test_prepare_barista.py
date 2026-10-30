import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from utils.prepare_barista import (
    frame_file_name,
    hash_split,
    official_split,
    prepare_barista,
    validate_barista_layout,
)
from utils.prepare_datasets import (
    _IGNORE_ACTIVITY_LABEL,
    _barista_sequence_id_from_path,
    _barista_window_frame_ranges_from_images,
    _barista_window_labels_from_frame_labels,
    _barista_window_step,
    _load_barista_frame_labels,
    _resolve_barista_label_root,
    _resolve_barista_sequence_split,
)
from utils.video_embedder import _image_folder_fps, _image_window_span

FFMPEG = shutil.which("ffmpeg")


def _write_official_video(root: Path, document_id: str, *, frame_count: int, split: str, activities) -> Path:
    video_dir = root / document_id
    video_dir.mkdir(parents=True)
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=32x32:rate=30",
            "-frames:v", str(frame_count), "-pix_fmt", "yuv420p", str(video_dir / "video.mp4"),
        ],
        check=True,
    )
    annotation = {
        "video_metadata": [
            {
                "id": 0,
                "document_id": document_id,
                "video_index": 0,
                "width": 32,
                "height": 32,
                "frame_count": frame_count,
                "fps": 30.0,
                "length_in_ms": frame_count / 30.0 * 1000.0,
                "recording_device_type": "test",
                "recording_device_version": "",
            }
        ],
        "categories": [],
        "annotations": [],
        "attributes": [],
        "relations": [],
        "activities": [
            {
                "id": index,
                "display_name": name,
                "activity_class_id": f"class-{name.lower().replace(' ', '-')}",
                "image_range": {"image_id_start": start, "image_id_end": end},
            }
            for index, (name, start, end) in enumerate(activities)
        ],
        "split": split,
    }
    (video_dir / "coco_annotation.json").write_text(json.dumps(annotation), encoding="utf-8")
    return video_dir


@unittest.skipUnless(FFMPEG, "ffmpeg is required to extract frames")
class PrepareBaristaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.source = base / "BARISTA"
        cls.output = base / "datasets" / "Barista"
        # Segment "Grab cup" overlaps "Place cup"; frames 21-39 are unlabeled.
        _write_official_video(
            cls.source, "doc-a", frame_count=70, split="train",
            activities=[("Place cup", 0, 9), ("Grab cup", 5, 20), ("Place cup", 40, 69)],
        )
        _write_official_video(
            cls.source, "doc-b", frame_count=65, split="test",
            activities=[("Press button", 0, 64)],
        )
        cls.exports = prepare_barista(cls.source, cls.output, workers=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_layout_and_sequence_ids(self):
        self.assertEqual(sorted(e.sequence_id for e in self.exports), ["doc-a__video_0", "doc-b__video_0"])
        image_dir = self.output / "Image_data" / "doc-a__video_0"
        frames = sorted(p.name for p in image_dir.glob("*.jpg"))
        self.assertEqual(len(frames), 70)
        self.assertEqual(frames[0], frame_file_name(0))
        self.assertEqual(frames[-1], frame_file_name(69))
        info = json.loads((image_dir / "source_info.json").read_text())
        self.assertAlmostEqual(info["fps"], 30.0)
        self.assertEqual(info["split"], "train")
        self.assertAlmostEqual(_image_folder_fps(str(image_dir)), 30.0)
        self.assertEqual(_image_folder_fps(self.tmp.name), 12.0)
        report = validate_barista_layout(self.output)
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["num_sequences"], 2)

    def test_labels_follow_export_conventions(self):
        label_root = _resolve_barista_label_root(self.output)
        self.assertEqual(label_root, self.output / "labels")
        frame_labels = _load_barista_frame_labels(label_root, label_mode="action")
        labels_a = frame_labels["doc-a__video_0"]
        self.assertEqual(len(labels_a), 70)
        self.assertEqual(labels_a[2], "Place cup")
        self.assertEqual(labels_a[7], "Grab cup")  # overlap: latest-starting segment wins
        self.assertEqual(labels_a[15], "Grab cup")
        self.assertEqual(labels_a[69], "Place cup")  # inclusive end frame
        self.assertEqual(labels_a[30], _IGNORE_ACTIVITY_LABEL)
        classes = (self.output / "labels" / "activity_classes.csv").read_text().splitlines()
        self.assertEqual(len(classes), 1 + 3)
        segments = (self.output / "labels" / "activity_segments.csv").read_text().splitlines()
        self.assertEqual(len(segments), 1 + 4)
        self.assertIn(",0,9,", segments[1])

    def test_window_alignment_uses_embedding_window_size(self):
        image_dir = str(self.output / "Image_data" / "doc-b__video_0")  # 65 frames
        self.assertEqual(
            _barista_window_frame_ranges_from_images(image_dir, 3, window_size=32),
            [(0, 31), (32, 63), (64, 64)],
        )
        # Without the window size the legacy even split is returned.
        self.assertEqual(
            _barista_window_frame_ranges_from_images(image_dir, 3),
            [(0, 21), (22, 43), (44, 64)],
        )
        self.assertEqual(_barista_window_step(65, 3, 32), 32)
        self.assertEqual(_barista_window_step(5149, 161, 32), 32)
        self.assertEqual(_barista_window_step(65, 4, 32), 17)  # inconsistent -> fallback
        frame_labels = ["A"] * 32 + ["B"] * 32 + ["C"]
        self.assertEqual(
            _barista_window_labels_from_frame_labels(
                image_dir, frame_labels, num_windows=3, window_size=32,
                background_label="SIL", empty_overlap_strategy="sil",
            ),
            ["A", "B", "C"],
        )

    def test_sequence_id_from_media_path(self):
        self.assertEqual(_barista_sequence_id_from_path("/d/Barista/Image_data/doc-a__video_0"), "doc-a__video_0")
        self.assertEqual(_barista_sequence_id_from_path("/d/Barista/Video_data/doc-a/video.mp4"), "doc-a__video_0")
        self.assertEqual(_barista_sequence_id_from_path("/d/raw/doc-a/video_0/video.mp4"), "doc-a__video_0")

    def test_split_resolution(self):
        ids = ["doc-a__video_0", "doc-b__video_0"]
        label_root = self.output / "labels"
        official = _resolve_barista_sequence_split(ids, "official", label_root=label_root)
        self.assertEqual(official["test"], ["doc-b__video_0"])
        self.assertEqual(official["train"], ["doc-a__video_0"])
        from_file = _resolve_barista_sequence_split(ids, str(self.output / "splits" / "official.json"))
        self.assertEqual(from_file, official)
        self.assertEqual(_resolve_barista_sequence_split(ids, "s1"), hash_split(ids))
        self.assertEqual(
            json.loads((self.output / "splits" / "s1_hash.json").read_text()), hash_split(ids)
        )
        with self.assertRaises(ValueError):
            _resolve_barista_sequence_split(ids, "s2")

    def test_image_window_span_covers_source_window(self):
        self.assertEqual(_image_window_span(32, 32, 30.0), (32 / 30.0, 63 / 30.0))
        self.assertEqual(_image_window_span(64, 1, 30.0), (64 / 30.0, 64 / 30.0))


class CheckedInSplitTests(unittest.TestCase):
    """The split files under configs/splits/ must be reproducible from the sequence ids and the official field."""

    SPLIT_DIR = Path(__file__).resolve().parents[1] / "configs" / "splits"

    def test_s1_hash_split_matches_hash_of_its_ids(self):
        s1 = json.loads((self.SPLIT_DIR / "barista_s1_hash.json").read_text())
        ids = sorted(sum(s1.values(), []))
        self.assertEqual(len(ids), 185)
        self.assertEqual(hash_split(ids), s1)
        self.assertEqual({k: len(v) for k, v in s1.items()}, {"train": 129, "val": 28, "test": 28})

    def test_official_split_matches_official_field(self):
        field = json.loads((self.SPLIT_DIR / "barista_official_split_field.json").read_text())
        official = json.loads((self.SPLIT_DIR / "barista_official.json").read_text())
        by_sequence = {f"{doc}__video_0": split for doc, split in field.items()}
        self.assertEqual(official_split(by_sequence, 0.15), official)
        self.assertEqual(sorted(official["test"]), sorted(sid for sid, s in by_sequence.items() if s == "test"))
        self.assertEqual({k: len(v) for k, v in official.items()}, {"train": 126, "val": 22, "test": 37})

    def test_rewrite_splits_from_field_file(self):
        from utils.prepare_barista import rewrite_splits

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels").mkdir()
            rows = ["sequence_id,source_frame_index,activity_label_index"]
            for sid in ("doc-a__video_0", "doc-b__video_0", "doc-c__video_0"):
                rows += [f"{sid},{i},-1" for i in range(3)]
            (root / "labels" / "activity_labels.csv").write_text("\n".join(rows) + "\n")
            field = root / "field.json"
            field.write_text(json.dumps({"doc-a": "train", "doc-b": "test", "doc-c": "train"}))
            summary = rewrite_splits(root, official_split_field=field, val_fraction=0.5)
            official = json.loads((root / "splits" / "official.json").read_text())
            self.assertEqual(official["test"], ["doc-b__video_0"])
            self.assertEqual(len(official["train"]) + len(official["val"]), 2)
            self.assertEqual(summary["official_split_field"], {"train": 2, "test": 1})
            self.assertTrue((root / "splits" / "s1_hash.json").exists())


class SplitHelperTests(unittest.TestCase):
    def test_hash_split_sizes(self):
        ids = [f"seq_{i:03d}" for i in range(185)]
        split = hash_split(ids)
        self.assertEqual({k: len(v) for k, v in split.items()}, {"train": 129, "val": 28, "test": 28})
        self.assertEqual(sorted(sum(split.values(), [])), ids)

    def test_official_split_preserves_test_set(self):
        assignment = {f"s{i}": "train" for i in range(20)}
        assignment.update({"t1": "test", "t2": "test"})
        split = official_split(assignment, val_fraction=0.15)
        self.assertEqual(split["test"], ["t1", "t2"])
        self.assertEqual(len(split["val"]), 3)
        self.assertEqual(len(split["train"]), 17)
        self.assertIsNone(official_split({"a": None}))


if __name__ == "__main__":
    unittest.main()
