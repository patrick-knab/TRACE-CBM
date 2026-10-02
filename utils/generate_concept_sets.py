#!/usr/bin/env python3
"""Iteratively generate visually grounded, dataset-specific concept sets.

The generator intentionally reuses ``utils.prepare_datasets.prepare_data`` to
resolve the same train splits, labels, and PE-L14 windows used by training.
Qwen is accessed through a local OpenAI-compatible vLLM endpoint.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import pickle
import random
import re
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.prepare_datasets import prepare_data
from utils.paths import embedding_root


CATEGORIES = (
    "object_material",
    "state_attribute",
    "spatial_interaction",
    "action_primitive",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def normalize_phrase(value: str) -> str:
    """Normalize separators/case for exact equality without semantic expansion."""
    value = unicodedata.normalize("NFKC", str(value)).lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def phrase_word_count(value: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", str(value)))


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)


def parse_json_content(content: str) -> Any:
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_candidates = [index for index in (text.find("{"), text.find("[")) if index >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(text.rfind("}"), text.rfind("]"))
        if end < start:
            raise
        return json.loads(text[start : end + 1])


class TruncatedVLMResponse(RuntimeError):
    """Raised when vLLM stops before completing the structured response."""


@dataclass(frozen=True)
class WindowRef:
    row: int
    video_path: str
    video_id: str
    window_index: int
    start_time: float
    end_time: float
    fps: float
    label_index: int
    label_name: str

    @property
    def uid(self) -> str:
        digest = sha1(f"{self.video_path}|{self.window_index}".encode("utf-8")).hexdigest()[:12]
        return f"W{digest}"


@dataclass
class Candidate:
    text: str
    normalized: str
    category: str
    round_index: int
    proposal_evidence: list[str] = field(default_factory=list)
    proposal_rationale: str = ""
    retrieved_evidence: list[str] = field(default_factory=list)
    verified_evidence: list[str] = field(default_factory=list)
    verification_confidence: float = 0.0
    support_score: float = 0.0
    nearest_target: str = ""
    target_similarity: float = -1.0
    rejection_reason: str = ""

    @property
    def grounded(self) -> bool:
        return not self.rejection_reason and bool(self.verified_evidence)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Candidate":
        fields = cls.__dataclass_fields__
        return cls(**{key: payload[key] for key in fields if key in payload})


@dataclass
class DatasetIndex:
    dataset: str
    label_names: list[str]
    forbidden_labels: list[str]
    windows: list[WindowRef]
    features: np.ndarray
    embedding_path: Path

    def sample_storyboards(
        self,
        count: int,
        sampled_uids: set[str],
        label_sample_counts: Mapping[int, int],
        rng: random.Random,
    ) -> list[WindowRef]:
        eligible = [window for window in self.windows if window.uid not in sampled_uids]
        if len(eligible) < count:
            eligible = list(self.windows)
        by_label: dict[int, list[WindowRef]] = {}
        for window in eligible:
            if window.label_index >= 0:
                by_label.setdefault(window.label_index, []).append(window)
        for values in by_label.values():
            rng.shuffle(values)

        chosen: list[WindowRef] = []
        used_videos: set[str] = set()
        labels = sorted(by_label, key=lambda label: (label_sample_counts.get(label, 0), label))
        while labels and len(chosen) < count:
            made_progress = False
            for label in list(labels):
                pool = by_label[label]
                if not pool:
                    labels.remove(label)
                    continue
                distinct = next((item for item in pool if item.video_id not in used_videos), None)
                item = distinct or pool[0]
                pool.remove(item)
                chosen.append(item)
                used_videos.add(item.video_id)
                made_progress = True
                if len(chosen) >= count:
                    break
            if not made_progress:
                break
        if len(chosen) < count:
            remaining = [item for item in eligible if item.uid not in {entry.uid for entry in chosen}]
            rng.shuffle(remaining)
            chosen.extend(remaining[: count - len(chosen)])
        return chosen[:count]

    def retrieve(self, embedding: np.ndarray, limit: int) -> tuple[list[WindowRef], list[float]]:
        scores = self.features @ normalize_rows(np.asarray(embedding).reshape(1, -1))[0]
        order = np.argsort(-scores)
        selected: list[WindowRef] = []
        selected_scores: list[float] = []
        used_videos: set[str] = set()
        for row in order:
            window = self.windows[int(row)]
            if window.video_id in used_videos:
                continue
            selected.append(window)
            selected_scores.append(float(scores[int(row)]))
            used_videos.add(window.video_id)
            if len(selected) >= limit:
                break
        return selected, selected_scores


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class PeTextEncoder:
    def __init__(self, device: str = "cuda:0") -> None:
        utils_dir = str(UTILS_DIR)
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)
        import torch
        from core.vision_encoder import pe
        from core.vision_encoder import transforms as pe_transforms

        self.torch = torch
        self.device = torch.device(device)
        self.model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).to(self.device).eval()
        self.tokenizer = pe_transforms.get_text_tokenizer(self.model.context_length)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), 64):
                tokens = self.tokenizer(list(texts[start : start + 64])).to(self.device)
                features = self.model.encode_text(tokens).float()
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                chunks.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1024), dtype=np.float32)


class ConceptVLM(Protocol):
    def analyze_storyboards(
        self,
        *,
        prompt: str,
        storyboards: Sequence[tuple[str, Path]],
        seed: int,
    ) -> list[dict[str, Any]]: ...

    def propose(
        self,
        *,
        prompt: str,
        proposal_count: int,
        seed: int,
    ) -> list[dict[str, Any]]: ...

    def verify(
        self,
        *,
        items: Sequence[tuple[str, Sequence[tuple[str, Path]]]],
        seed: int,
    ) -> dict[str, dict[str, Any]]: ...


class VLLMClient:
    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: int = 900,
        proposal_max_tokens: int = 16_000,
        proposal_retry_attempts: int = 2,
    ) -> None:
        self.url = endpoint.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout = int(timeout)
        self.proposal_max_tokens = int(proposal_max_tokens)
        self.proposal_retry_attempts = int(proposal_retry_attempts)
        if self.proposal_max_tokens < 1:
            raise ValueError("proposal_max_tokens must be at least 1")
        if self.proposal_retry_attempts < 0:
            raise ValueError("proposal_retry_attempts must be non-negative")

    @staticmethod
    def _image_content(path: Path) -> dict[str, Any]:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        }

    def _chat(
        self,
        content: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5 if temperature > 0 else 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "seed": int(seed),
            "max_tokens": int(max_tokens),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "concept_generation", "strict": True, "schema": schema},
            },
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request failed ({error.code}): {body[:1000]}") from error
        choice = result["choices"][0]
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise TruncatedVLMResponse(
                f"vLLM exhausted max_tokens={max_tokens} before completing the JSON response"
            )
        content_text = choice["message"]["content"]
        return parse_json_content(content_text)

    def analyze_storyboards(
        self,
        *,
        prompt: str,
        storyboards: Sequence[tuple[str, Path]],
        seed: int,
    ) -> list[dict[str, Any]]:
        storyboard_count = len(storyboards)
        schema = {
            "type": "object",
            "properties": {
                "storyboards": {
                    "type": "array",
                    "minItems": storyboard_count,
                    "maxItems": storyboard_count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "storyboard_id": {"type": "string"},
                            "visible_objects": {"type": "array", "items": {"type": "string"}},
                            "visible_states": {"type": "array", "items": {"type": "string"}},
                            "visible_interactions": {"type": "array", "items": {"type": "string"}},
                            "evidence_summary": {"type": "string"},
                        },
                        "required": [
                            "storyboard_id",
                            "visible_objects",
                            "visible_states",
                            "visible_interactions",
                            "evidence_summary",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["storyboards"],
            "additionalProperties": False,
        }
        content = [{"type": "text", "text": prompt}]
        for storyboard_id, path in storyboards:
            content.append({"type": "text", "text": f"Storyboard ID: {storyboard_id}"})
            content.append(self._image_content(path))
        result = self._chat(content, schema, seed=seed, temperature=0.0, max_tokens=6000)
        return list(result.get("storyboards", []))

    def propose(
        self,
        *,
        prompt: str,
        proposal_count: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        schema = {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": int(proposal_count),
                    "maxItems": int(proposal_count),
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "category": {"type": "string", "enum": list(CATEGORIES)},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "rationale": {"type": "string"},
                        },
                        "required": ["text", "category", "evidence_ids", "rationale"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["candidates"],
            "additionalProperties": False,
        }
        content = [{"type": "text", "text": prompt}]
        last_error: Exception | None = None
        for attempt in range(self.proposal_retry_attempts + 1):
            max_tokens = min(self.proposal_max_tokens * (attempt + 1), 48_000)
            try:
                result = self._chat(
                    content,
                    schema,
                    seed=seed,
                    temperature=0.7,
                    max_tokens=max_tokens,
                )
                return list(result.get("candidates", []))
            except (TruncatedVLMResponse, json.JSONDecodeError) as error:
                last_error = error
                if attempt >= self.proposal_retry_attempts:
                    break
                next_max_tokens = min(
                    self.proposal_max_tokens * (attempt + 2),
                    48_000,
                )
                print(
                    "[vlm] Incomplete proposal JSON "
                    f"({error}); retrying with max_tokens={next_max_tokens}.",
                    flush=True,
                )
        raise RuntimeError(
            "vLLM did not return complete proposal JSON after "
            f"{self.proposal_retry_attempts + 1} attempts"
        ) from last_error

    def verify(
        self,
        *,
        items: Sequence[tuple[str, Sequence[tuple[str, Path]]]],
        seed: int,
    ) -> dict[str, dict[str, Any]]:
        schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "concept": {"type": "string"},
                            "visible_storyboard_ids": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["concept", "visible_storyboard_ids", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        }
        instructions = (
            "Verify whether each short concept is visibly supported by its retrieved storyboards. "
            "Return only storyboard IDs where the concept is genuinely visible. Do not infer hidden "
            "objects or actions from labels; use the images only."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": instructions}]
        for concept, evidence in items:
            ids = [uid for uid, _ in evidence]
            content.append(
                {"type": "text", "text": f"Concept: {concept}\nCandidate storyboard IDs: {ids}"}
            )
            content.extend(self._image_content(path) for _, path in evidence)
        result = self._chat(content, schema, seed=seed, temperature=0.0, max_tokens=1600)
        return {normalize_phrase(item["concept"]): dict(item) for item in result.get("results", [])}


class StoryboardRenderer:
    def __init__(self, embedding_path: Path, output_dir: Path, panel_size: int = 336) -> None:
        self.embedding_path = embedding_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.panel_size = int(panel_size)
        self._image_lists: dict[str, list[Path]] = {}

    def resolve_media_path(self, value: str) -> Path:
        path = Path(value)
        if path.exists():
            return path.resolve()
        for base in (self.embedding_path.parent, self.embedding_path.parent.parent):
            candidate = (base / path).resolve()
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Media path not found: {value}")

    @staticmethod
    def _natural_key(path: Path) -> list[Any]:
        return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", path.name)]

    @staticmethod
    def _letterbox(frame: np.ndarray, size: int) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = min(size / max(width, 1), size / max(height, 1))
        resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        y = (size - resized.shape[0]) // 2
        x = (size - resized.shape[1]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return canvas

    def _read_frames(self, window: WindowRef) -> list[np.ndarray]:
        media_path = self.resolve_media_path(window.video_path)
        times = np.linspace(window.start_time, max(window.start_time, window.end_time), 4)
        frames: list[np.ndarray] = []
        if media_path.is_dir():
            key = str(media_path)
            if key not in self._image_lists:
                paths = [path for path in media_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
                self._image_lists[key] = sorted(paths, key=self._natural_key)
            paths = self._image_lists[key]
            if not paths:
                raise RuntimeError(f"No images found in {media_path}")
            indices = np.clip(np.rint(times * window.fps).astype(int), 0, len(paths) - 1)
            for index in indices:
                frame = cv2.imread(str(paths[int(index)]))
                if frame is None:
                    raise RuntimeError(f"Could not read image {paths[int(index)]}")
                frames.append(frame)
            return frames

        if media_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RuntimeError(f"Unsupported media path: {media_path}")
        capture = cv2.VideoCapture(str(media_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video {media_path}")
            for timestamp in times:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not read {media_path} at {timestamp:.3f}s")
                frames.append(frame)
        finally:
            capture.release()
        return frames

    def render(self, window: WindowRef) -> Path:
        path = self.output_dir / f"{window.uid}.jpg"
        if path.exists():
            return path
        panels = [self._letterbox(frame, self.panel_size) for frame in self._read_frames(window)]
        sheet = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
        cv2.rectangle(sheet, (0, 0), (sheet.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            sheet,
            f"{window.uid}  {window.start_time:.2f}-{window.end_time:.2f}s",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise RuntimeError(f"Could not write storyboard {path}")
        return path


def _embedding_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"Unsupported embedding payload: {type(value)!r}")


def load_embedding_payload(path: Path) -> dict[str, Any]:
    utils_dir = str(UTILS_DIR)
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    with path.open("rb") as handle:
        return _embedding_payload(pickle.load(handle))


def build_dataset_index(dataset_config: Mapping[str, Any]) -> DatasetIndex:
    embedding_path = resolve_embedding_path(dataset_config["embedding_path"])
    payload = load_embedding_payload(embedding_path)
    primary_mode = str(dataset_config["label_modes"][0])
    prepared = prepare_data(
        payload,
        concept_set=dataset_config["bootstrap_concept_set"],
        test_split=dataset_config.get("test_split", "s1"),
        backbone="pe-l14",
        device="cpu",
        dataset=dataset_config["dataset"],
        text_embedding_cache=dataset_config.get("bootstrap_text_cache"),
        activity_label_mode=primary_mode,
        activity_label_fill_mode=dataset_config.get("activity_label_fill_mode", "sil"),
        verbose=False,
    )
    forbidden_labels = set(str(value) for value in prepared["metadata"]["activity_names"])
    for extra_mode in dataset_config["label_modes"][1:]:
        extra = prepare_data(
            payload,
            concept_set=dataset_config["bootstrap_concept_set"],
            test_split=dataset_config.get("test_split", "s1"),
            backbone="pe-l14",
            device="cpu",
            dataset=dataset_config["dataset"],
            text_embedding_cache=dataset_config.get("bootstrap_text_cache"),
            activity_label_mode=extra_mode,
            activity_label_fill_mode=dataset_config.get("activity_label_fill_mode", "sil"),
            verbose=False,
        )
        forbidden_labels.update(str(value) for value in extra["metadata"]["activity_names"])
        del extra

    train = prepared["train"]
    label_names = [str(value) for value in prepared["metadata"]["activity_names"]]
    spans_by_path = payload.get("video_window_spans", {})
    meta_by_path = payload.get("video_meta", {})
    feature_chunks: list[np.ndarray] = []
    windows: list[WindowRef] = []
    row = 0
    for video_index, raw_path in enumerate(train["video_paths"]):
        video_path = str(raw_path)
        video_id = str(train["video_ids"][video_index])
        length = int(train["lengths"][video_index])
        features = np.asarray(train["raw_features"][video_index, :length], dtype=np.float32)
        feature_chunks.append(features)
        spans = spans_by_path.get(video_path, [])
        fps = float(meta_by_path.get(video_path, {}).get("fps", 30.0) or 30.0)
        for window_index in range(length):
            if window_index < len(spans):
                start_time, end_time = spans[window_index]
            else:
                window_size = int(payload.get("config", {}).get("window_size", 32))
                start_time = window_index * window_size / fps
                end_time = (window_index + 1) * window_size / fps
            label_index = int(train["activity_labels"][video_index, window_index])
            label_name = label_names[label_index] if 0 <= label_index < len(label_names) else ""
            windows.append(
                WindowRef(
                    row=row,
                    video_path=video_path,
                    video_id=video_id,
                    window_index=window_index,
                    start_time=float(start_time),
                    end_time=float(end_time),
                    fps=fps,
                    label_index=label_index,
                    label_name=label_name,
                )
            )
            row += 1
    flat_features = normalize_rows(np.concatenate(feature_chunks, axis=0))
    if len(windows) != int(flat_features.shape[0]):
        raise RuntimeError("Window metadata and PE feature rows do not align")
    return DatasetIndex(
        dataset=str(dataset_config["dataset"]),
        label_names=label_names,
        forbidden_labels=sorted(forbidden_labels, key=normalize_phrase),
        windows=windows,
        features=flat_features,
        embedding_path=embedding_path,
    )


def storyboard_analysis_prompt(*, dataset: str, storyboard_ids: Sequence[str]) -> str:
    return f"""Analyze the supplied four-frame training storyboards from dataset {dataset}.

Work label-blind: no activity labels are provided, and you must not infer a hidden activity.
For every storyboard, report only directly visible objects/materials, visible states/attributes,
and spatial or hand-object interactions. Motion may be reported only when the change is visible
across the four frames. Do not infer sound, intent, task progress, or objects outside the image.

The images are supplied with explicit storyboard ID markers in this order:
{json.dumps(list(storyboard_ids))}

Return one structured observation for every supplied storyboard ID. The evidence_summary must be
a short factual description of what is visibly present, not private chain-of-thought."""


def proposal_prompt(
    *,
    dataset: str,
    storyboard_context: Sequence[Mapping[str, Any]],
    selected: Sequence[Candidate],
    proposal_count: int,
) -> str:
    existing = [candidate.text for candidate in selected]
    return f"""You are building a reusable visual concept vocabulary for dataset {dataset}.

The following label-blind observations were produced from individual four-frame storyboards.
Activity labels are now attached only to audit whether the visual vocabulary covers useful
evidence for those activities:
{json.dumps(list(storyboard_context), ensure_ascii=False)}

Propose exactly {proposal_count} short concepts grounded in those observations. Each phrase must
contain 1-3 words and must be reusable across videos. Prefer concrete objects/materials, visible
states/attributes, and spatial or hand-object relations. Include an action primitive only when
motion is directly supported across the four frames. Do not satisfy a fixed category quota.

Already selected concepts to avoid repeating:
{json.dumps(existing, ensure_ascii=False)}

Do not repeat, restate, or semantically paraphrase an activity label. Do not use SIL/background
as a concept. The label must not be used to infer anything absent from the label-blind observation.
For every candidate, cite one or more supporting storyboard IDs and give one factual rationale
of at most 16 words covering visible evidence and reusability. Return the required JSON schema
only."""


def order_storyboard_analyses(
    analyses: Sequence[Mapping[str, Any]],
    storyboard_ids: Sequence[str],
) -> list[dict[str, Any]]:
    expected = list(storyboard_ids)
    by_id: dict[str, dict[str, Any]] = {}
    for raw in analyses:
        storyboard_id = str(raw.get("storyboard_id", ""))
        if storyboard_id in by_id:
            raise RuntimeError(f"Duplicate storyboard analysis for {storyboard_id!r}")
        by_id[storyboard_id] = dict(raw)
    missing = [storyboard_id for storyboard_id in expected if storyboard_id not in by_id]
    unexpected = sorted(set(by_id).difference(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"Storyboard analysis IDs do not match input: missing={missing}, unexpected={unexpected}"
        )
    return [by_id[storyboard_id] for storyboard_id in expected]


def hard_filter_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    forbidden_labels: Sequence[str],
    valid_evidence_ids: set[str],
    round_index: int,
    existing_normalized: set[str],
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    forbidden = {normalize_phrase(value) for value in forbidden_labels}
    accepted: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    seen = set(existing_normalized)
    for raw in proposals:
        text = " ".join(str(raw.get("text", "")).strip().split())
        normalized = normalize_phrase(text)
        category = str(raw.get("category", ""))
        evidence = [str(value) for value in raw.get("evidence_ids", [])]
        rationale = " ".join(str(raw.get("rationale", "")).strip().split())
        reason = ""
        if not normalized:
            reason = "empty"
        elif phrase_word_count(text) > 3:
            reason = "too_many_words"
        elif category not in CATEGORIES:
            reason = "invalid_category"
        elif normalized in forbidden:
            reason = "exact_target_label"
        elif normalized in seen:
            reason = "canonical_duplicate"
        elif not evidence or any(value not in valid_evidence_ids for value in evidence):
            reason = "invalid_evidence"
        elif not rationale:
            reason = "missing_rationale"
        if reason:
            rejected.append({"proposal": dict(raw), "reason": reason})
            continue
        seen.add(normalized)
        accepted.append(
            Candidate(
                text=text,
                normalized=normalized,
                category=category,
                round_index=round_index,
                proposal_evidence=sorted(set(evidence)),
                proposal_rationale=rationale,
            )
        )
    return accepted, rejected


def candidate_utility(candidate: Candidate, category_counts: Mapping[str, int]) -> float:
    undercoverage = 1.0 / (1.0 + float(category_counts.get(candidate.category, 0)))
    brevity = 1.0 / max(1, phrase_word_count(candidate.text))
    return (
        0.45 * float(candidate.verification_confidence)
        + 0.35 * float(candidate.support_score)
        + 0.15 * undercoverage
        + 0.05 * brevity
    )


def select_nonredundant(
    candidates: Sequence[Candidate],
    embeddings: np.ndarray,
    *,
    max_count: int,
    similarity_threshold: float,
) -> list[int]:
    if len(candidates) != int(embeddings.shape[0]):
        raise ValueError("Candidate and embedding counts differ")
    embeddings = normalize_rows(embeddings)
    remaining = {index for index, candidate in enumerate(candidates) if candidate.grounded}
    selected: list[int] = []
    category_counts: dict[str, int] = {}
    while remaining and len(selected) < max_count:
        ranked = sorted(
            remaining,
            key=lambda index: (
                candidate_utility(candidates[index], category_counts),
                -phrase_word_count(candidates[index].text),
                candidates[index].normalized,
            ),
            reverse=True,
        )
        choice = ranked[0]
        selected.append(choice)
        category = candidates[choice].category
        category_counts[category] = category_counts.get(category, 0) + 1
        remaining.remove(choice)
        if remaining:
            indices = np.asarray(sorted(remaining), dtype=np.int64)
            similarities = embeddings[indices] @ embeddings[choice]
            for index, similarity in zip(indices.tolist(), similarities.tolist()):
                if float(similarity) >= similarity_threshold:
                    remaining.discard(index)
    return selected


def validate_final_selection(
    selected: Sequence[Candidate],
    embeddings: np.ndarray,
    *,
    forbidden_labels: Sequence[str],
    expected_count: int,
    similarity_threshold: float,
) -> None:
    if len(selected) != expected_count:
        raise ValueError(f"Expected {expected_count} concepts, found {len(selected)}")
    normalized = [candidate.normalized for candidate in selected]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Final concept set contains canonical duplicates")
    forbidden = {normalize_phrase(value) for value in forbidden_labels}
    exact_overlap = sorted(set(normalized).intersection(forbidden))
    if exact_overlap:
        raise ValueError(f"Final concept set overlaps target labels: {exact_overlap[:10]}")
    invalid_length = [candidate.text for candidate in selected if not 1 <= phrase_word_count(candidate.text) <= 3]
    if invalid_length:
        raise ValueError(f"Final concept set contains invalid phrase lengths: {invalid_length[:10]}")
    ungrounded = [candidate.text for candidate in selected if not candidate.grounded]
    if ungrounded:
        raise ValueError(f"Final concept set contains ungrounded concepts: {ungrounded[:10]}")
    if len(selected) > 1:
        similarities = normalize_rows(embeddings) @ normalize_rows(embeddings).T
        np.fill_diagonal(similarities, -1.0)
        maximum = float(np.max(similarities))
        if maximum >= similarity_threshold:
            raise ValueError(
                f"Final concept redundancy {maximum:.6f} exceeds threshold {similarity_threshold:.6f}"
            )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "rounds_completed": 0,
            "stalls": 0,
            "sampled_uids": [],
            "label_sample_counts": {},
            "candidates": [],
            "selected": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(
    dataset_dir: Path,
    selected: Sequence[Candidate],
    selected_embeddings: np.ndarray,
    *,
    semantic_warning_similarity: float,
) -> None:
    selected_embeddings = normalize_rows(selected_embeddings)
    similarities = selected_embeddings @ selected_embeddings.T
    np.fill_diagonal(similarities, -1.0)
    rows: list[str] = []
    audit: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        nearest_index = int(np.argmax(similarities[index])) if len(selected) > 1 else index
        nearest_text = selected[nearest_index].text if len(selected) > 1 else ""
        nearest_similarity = float(similarities[index, nearest_index]) if len(selected) > 1 else -1.0
        warning = candidate.target_similarity >= semantic_warning_similarity
        record = asdict(candidate) | {
            "nearest_concept": nearest_text,
            "nearest_concept_similarity": nearest_similarity,
            "semantic_target_warning": warning,
        }
        audit.append(record)
        rows.append(
            "<tr>"
            f"<td>{index + 1}</td><td>{html.escape(candidate.text)}</td>"
            f"<td>{html.escape(candidate.category)}</td>"
            f"<td>{html.escape(candidate.proposal_rationale)}</td>"
            f"<td>{candidate.verification_confidence:.3f}</td>"
            f"<td>{candidate.support_score:.3f}</td>"
            f"<td>{html.escape(nearest_text)} ({nearest_similarity:.3f})</td>"
            f"<td>{html.escape(candidate.nearest_target)} ({candidate.target_similarity:.3f})"
            f"{' WARNING' if warning else ''}</td>"
            f"<td>{html.escape(', '.join(candidate.verified_evidence))}</td>"
            "</tr>"
        )
    _atomic_json(dataset_dir / "final_audit.json", audit)
    document = """<!doctype html><meta charset="utf-8"><title>Concept audit</title>
<style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.35rem}th{background:#eee}</style>
<h1>Final concept audit</h1><table><thead><tr><th>#</th><th>Concept</th><th>Category</th><th>Proposal rationale</th><th>VLM confidence</th><th>PE support</th><th>Nearest concept</th><th>Nearest target</th><th>Verified evidence</th></tr></thead><tbody>"""
    document += "".join(rows) + "</tbody></table>"
    (dataset_dir / "report.html").write_text(document, encoding="utf-8")


def run_dataset(
    *,
    dataset_config: Mapping[str, Any],
    global_config: Mapping[str, Any],
    run_dir: Path,
    encoder: TextEncoder,
    vlm: ConceptVLM,
) -> Path:
    dataset = str(dataset_config["dataset"])
    dataset_dir = run_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    state_path = dataset_dir / "state.json"
    state = _load_state(state_path)
    index = build_dataset_index(dataset_config)
    _atomic_json(dataset_dir / "resolved_labels.json", index.forbidden_labels)
    _atomic_json(
        dataset_dir / "train_manifest.json",
        {
            "dataset": dataset,
            "embedding_path": str(index.embedding_path),
            "num_train_windows": len(index.windows),
            "num_train_videos": len({window.video_id for window in index.windows}),
            "train_video_ids": sorted({window.video_id for window in index.windows}),
        },
    )
    renderer = StoryboardRenderer(index.embedding_path, dataset_dir / "storyboards")
    candidates = [Candidate.from_dict(value) for value in state.get("candidates", [])]
    sampled_uids = set(state.get("sampled_uids", []))
    label_sample_counts = {int(key): int(value) for key, value in state.get("label_sample_counts", {}).items()}
    candidate_embeddings = encoder.encode([candidate.text for candidate in candidates]) if candidates else np.zeros((0, index.features.shape[1]), dtype=np.float32)
    target_embeddings = encoder.encode(index.forbidden_labels)
    max_concepts = int(global_config["max_concepts"])
    max_rounds = int(global_config["max_rounds"])
    proposal_count = int(global_config["proposal_count"])
    storyboards_per_round = int(global_config["storyboards_per_round"])
    similarity_threshold = float(global_config["redundancy_similarity"])
    verify_batch_size = int(global_config.get("verify_batch_size", 4))
    seed = int(global_config.get("seed", 42))
    stalls = int(state.get("stalls", 0))
    selected_indices = select_nonredundant(
        candidates,
        candidate_embeddings,
        max_count=max_concepts,
        similarity_threshold=similarity_threshold,
    )
    selected = [candidates[index_value] for index_value in selected_indices]
    window_by_uid = {window.uid: window for window in index.windows}

    start_round = int(state.get("rounds_completed", 0)) + 1
    for round_index in range(start_round, max_rounds + 1):
        rng = random.Random(seed + round_index)
        sampled = index.sample_storyboards(
            storyboards_per_round,
            sampled_uids,
            label_sample_counts,
            rng,
        )
        storyboard_paths: list[Path] = []
        valid_sampled: list[WindowRef] = []
        for window in sampled:
            try:
                storyboard_paths.append(renderer.render(window))
                valid_sampled.append(window)
            except (FileNotFoundError, RuntimeError) as error:
                print(f"[{dataset}] skipping storyboard {window.uid}: {error}", flush=True)
        if not valid_sampled:
            raise RuntimeError(f"No storyboards could be rendered for {dataset}")
        for window in valid_sampled:
            sampled_uids.add(window.uid)
            label_sample_counts[window.label_index] = label_sample_counts.get(window.label_index, 0) + 1

        storyboard_ids = [window.uid for window in valid_sampled]
        round_dir = dataset_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        analysis_prompt = storyboard_analysis_prompt(
            dataset=dataset,
            storyboard_ids=storyboard_ids,
        )
        (round_dir / "storyboard_analysis_prompt.txt").write_text(
            analysis_prompt, encoding="utf-8"
        )
        analyses = vlm.analyze_storyboards(
            prompt=analysis_prompt,
            storyboards=list(zip(storyboard_ids, storyboard_paths)),
            seed=seed + round_index,
        )
        analyses = order_storyboard_analyses(analyses, storyboard_ids)
        _atomic_json(round_dir / "storyboard_analysis.json", analyses)
        label_by_uid = {window.uid: window.label_name for window in valid_sampled}
        storyboard_context = [
            analysis | {"activity_label": label_by_uid[str(analysis["storyboard_id"])]}
            for analysis in analyses
        ]
        _atomic_json(round_dir / "proposal_context.json", storyboard_context)
        prompt = proposal_prompt(
            dataset=dataset,
            storyboard_context=storyboard_context,
            selected=selected,
            proposal_count=proposal_count,
        )
        (round_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        proposals = vlm.propose(
            prompt=prompt,
            proposal_count=proposal_count,
            seed=seed + round_index,
        )
        _atomic_json(round_dir / "raw_proposals.json", proposals)
        new_candidates, rejected = hard_filter_proposals(
            proposals,
            forbidden_labels=index.forbidden_labels,
            valid_evidence_ids={window.uid for window in valid_sampled},
            round_index=round_index,
            existing_normalized={candidate.normalized for candidate in candidates},
        )
        _atomic_json(round_dir / "hard_rejections.json", rejected)
        before_selected = len(selected_indices)
        if new_candidates:
            new_embeddings = encoder.encode([candidate.text for candidate in new_candidates])
            for candidate, embedding in zip(new_candidates, new_embeddings):
                retrieved, support_scores = index.retrieve(
                    embedding, int(global_config.get("retrieval_storyboards", 6))
                )
                candidate.retrieved_evidence = [window.uid for window in retrieved]
                candidate.support_score = float(np.mean(support_scores)) if support_scores else 0.0
            for batch_start in range(0, len(new_candidates), verify_batch_size):
                batch = new_candidates[batch_start : batch_start + verify_batch_size]
                items: list[tuple[str, list[tuple[str, Path]]]] = []
                for candidate in batch:
                    evidence_windows = [window_by_uid[uid] for uid in candidate.retrieved_evidence]
                    items.append(
                        (
                            candidate.text,
                            [(window.uid, renderer.render(window)) for window in evidence_windows],
                        )
                    )
                verified = vlm.verify(items=items, seed=seed + round_index * 1000 + batch_start)
                for candidate in batch:
                    result = verified.get(candidate.normalized, {})
                    allowed = set(candidate.retrieved_evidence)
                    candidate.verified_evidence = sorted(
                        allowed.intersection(str(value) for value in result.get("visible_storyboard_ids", []))
                    )
                    candidate.verification_confidence = float(result.get("confidence", 0.0))
                    if len(candidate.verified_evidence) < int(global_config.get("min_verified_videos", 2)):
                        candidate.rejection_reason = "insufficient_visual_support"

            label_similarities = normalize_rows(new_embeddings) @ normalize_rows(target_embeddings).T
            for candidate_index, candidate in enumerate(new_candidates):
                target_index = int(np.argmax(label_similarities[candidate_index]))
                candidate.nearest_target = index.forbidden_labels[target_index]
                candidate.target_similarity = float(label_similarities[candidate_index, target_index])
            candidates.extend(new_candidates)
            candidate_embeddings = np.concatenate((candidate_embeddings, new_embeddings), axis=0)

        selected_indices = select_nonredundant(
            candidates,
            candidate_embeddings,
            max_count=max_concepts,
            similarity_threshold=similarity_threshold,
        )
        selected = [candidates[index_value] for index_value in selected_indices]
        stalls = stalls + 1 if len(selected_indices) <= before_selected else 0
        state = {
            "rounds_completed": round_index,
            "stalls": stalls,
            "sampled_uids": sorted(sampled_uids),
            "label_sample_counts": {str(key): value for key, value in label_sample_counts.items()},
            "candidates": [asdict(candidate) for candidate in candidates],
            "selected": [candidate.normalized for candidate in selected],
        }
        _atomic_json(state_path, state)
        print(
            f"[{dataset}] round={round_index} proposed={len(proposals)} "
            f"new={len(new_candidates)} selected={len(selected)}/{max_concepts}",
            flush=True,
        )
        if len(selected) >= max_concepts:
            break
        if stalls >= int(global_config.get("max_stalled_rounds", 3)):
            raise RuntimeError(
                f"{dataset} stalled for {stalls} rounds at {len(selected)}/{max_concepts}; "
                f"resume state is in {state_path}"
            )
    if len(selected) < max_concepts:
        raise RuntimeError(
            f"{dataset} reached max_rounds={max_rounds} with {len(selected)}/{max_concepts} concepts"
        )

    concept_set_name = str(dataset_config["output_concept_set"]).format(max_concepts=max_concepts)
    selected_embeddings = candidate_embeddings[selected_indices]
    validate_final_selection(
        selected,
        selected_embeddings,
        forbidden_labels=index.forbidden_labels,
        expected_count=max_concepts,
        similarity_threshold=similarity_threshold,
    )
    final_payload = {"concepts": {concept_set_name: [candidate.text for candidate in selected]}}
    final_path = Path(global_config["concepts_output_dir"]) / f"{concept_set_name}.json"
    _atomic_json(final_path, final_payload)
    _atomic_json(dataset_dir / "final_concepts.json", final_payload)
    _write_report(
        dataset_dir,
        selected,
        selected_embeddings,
        semantic_warning_similarity=float(global_config.get("semantic_warning_similarity", 0.90)),
    )
    return final_path


def validate_config(config: Mapping[str, Any], selected_datasets: set[str] | None = None) -> None:
    required_global = {
        "max_concepts",
        "proposal_count",
        "storyboards_per_round",
        "max_rounds",
        "redundancy_similarity",
        "concepts_output_dir",
        "run_output_dir",
        "vlm_endpoint",
        "vlm_model",
    }
    missing = required_global - set(config)
    if missing:
        raise ValueError(f"Config missing global keys: {sorted(missing)}")
    names: set[str] = set()
    for dataset_config in config.get("datasets", []):
        name = str(dataset_config.get("dataset", ""))
        if selected_datasets and name not in selected_datasets:
            continue
        names.add(name)
        for key in (
            "embedding_path",
            "bootstrap_concept_set",
            "label_modes",
            "output_concept_set",
        ):
            if key not in dataset_config:
                raise ValueError(f"Dataset {name!r} missing key {key!r}")
        embedding_path = resolve_embedding_path(dataset_config["embedding_path"])
        if not embedding_path.exists():
            raise FileNotFoundError(f"Dataset {name}: missing embedding_path: {embedding_path}")
        concept_value = str(dataset_config["bootstrap_concept_set"])
        concept_path = Path(concept_value)
        if not concept_path.exists():
            concept_path = PROJECT_ROOT / "concepts" / f"{concept_value}.json"
        if not concept_path.exists():
            raise FileNotFoundError(f"Dataset {name}: bootstrap concept set not found: {concept_value}")
    if not names:
        raise ValueError("No configured datasets matched the selection")


def resolve_embedding_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parts[:2] == ("data", "embeddings"):
        path = embedding_root().joinpath(*path.parts[2:])
    else:
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(UTILS_DIR / "concept_generation_qwen35.json"))
    parser.add_argument(
        "--max-concepts",
        type=int,
        help="Override the number of concepts per dataset from the config",
    )
    parser.add_argument("--datasets", nargs="*", help="Optional configured dataset names")
    parser.add_argument("--resume", help="Resume an existing run directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/config without loading models")
    parser.add_argument("--smoke", action="store_true", help="Run one local VLM response and PE text encode")
    parser.add_argument("--pe-device", default=os.environ.get("PE_DEVICE", "cuda:0"))
    parser.add_argument("--vlm-endpoint", help="Override the configured local vLLM endpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.max_concepts is not None:
        if args.max_concepts < 1:
            raise ValueError("--max-concepts must be at least 1")
        config["max_concepts"] = args.max_concepts
    selected_names = set(args.datasets) if args.datasets else None
    validate_config(config, selected_names)
    dataset_configs = [
        value for value in config["datasets"] if selected_names is None or value["dataset"] in selected_names
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": str(config_path),
                    "datasets": [value["dataset"] for value in dataset_configs],
                    "max_concepts": config["max_concepts"],
                    "proposal_count": config["proposal_count"],
                    "proposal_max_tokens": config.get("proposal_max_tokens", 16_000),
                    "proposal_retry_attempts": config.get("proposal_retry_attempts", 2),
                    "vlm_model": config["vlm_model"],
                },
                indent=2,
            )
        )
        return

    if args.resume:
        run_dir = Path(args.resume).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(config["run_output_dir"]).expanduser().resolve() / f"qwen35_pe_l14_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    encoder = PeTextEncoder(args.pe_device)
    vlm = VLLMClient(
        args.vlm_endpoint or config["vlm_endpoint"],
        config["vlm_model"],
        config.get("vlm_timeout", 900),
        config.get("proposal_max_tokens", 16_000),
        config.get("proposal_retry_attempts", 2),
    )
    if args.smoke:
        encoded = encoder.encode(["metal bowl", "coffee mug"])
        with tempfile.TemporaryDirectory(prefix="concept_qwen35_smoke_") as temporary_dir:
            image_path = Path(temporary_dir) / "SMOKE.jpg"
            image = np.zeros((336, 336, 3), dtype=np.uint8)
            cv2.rectangle(image, (80, 120), (256, 250), (180, 180, 180), -1)
            cv2.putText(image, "SMOKE", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError("Could not write VLM smoke image")
            analysis_prompt = storyboard_analysis_prompt(dataset="smoke", storyboard_ids=["SMOKE"])
            analyses = vlm.analyze_storyboards(
                prompt=analysis_prompt,
                storyboards=[("SMOKE", image_path)],
                seed=int(config.get("seed", 42)),
            )
            analyses = order_storyboard_analyses(analyses, ["SMOKE"])
            proposals = vlm.propose(
                prompt=proposal_prompt(
                    dataset="smoke",
                    storyboard_context=[analyses[0] | {"activity_label": "smoke_test"}],
                    selected=[],
                    proposal_count=1,
                ),
                proposal_count=1,
                seed=int(config.get("seed", 42)),
            )
        print(
            json.dumps(
                {
                    "pe_shape": list(encoded.shape),
                    "pe_norms": np.linalg.norm(encoded, axis=1).round(5).tolist(),
                    "vlm_storyboard_analysis": analyses,
                    "vlm_proposals": proposals,
                },
                indent=2,
            )
        )
        return
    outputs = []
    for dataset_config in dataset_configs:
        outputs.append(
            str(
                run_dataset(
                    dataset_config=dataset_config,
                    global_config=config,
                    run_dir=run_dir,
                    encoder=encoder,
                    vlm=vlm,
                )
            )
        )
    _atomic_json(run_dir / "completed.json", {"outputs": outputs, "completed_at": time.time()})
    print(json.dumps({"run_dir": str(run_dir), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
