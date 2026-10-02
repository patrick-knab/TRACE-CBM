from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from html import escape as html_escape
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.graph_concept_ui import (
    CheckpointRecord,
    discover_checkpoints,
    forward_outputs,
    forward_outputs_batched_interventions,
    graph_branch_options,
    graph_edges_for_concepts,
    graph_layer_options,
    graph_matrix,
    graph_temporal_vector,
    has_learned_threshold_calibrator,
    learned_threshold_intervention_value,
    load_workspace,
    prediction_option_rows,
    prediction_tables,
    prediction_concept_contributors,
    selected_class_probability,
    split_names,
    video_options,
)
from utils.paths import dataset_root as configured_dataset_root
from utils.intervention_notebook import select_instance


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PLAYBACK_BUFFER_SECONDS = 30.0
BUFFER_REFRESH_MARGIN_SECONDS = 5.0
VIDEO_HEIGHT = 480
POSITION_STATE_KEY = "video_position_seconds"
POSITION_SLIDER_KEY = "video_position_slider"
PENDING_SLIDER_POSITION_KEY = "video_pending_slider_position"
PLAYING_STATE_KEY = "video_playing"
PLAYBACK_STARTED_STATE_KEY = "video_playback_started"
SOURCE_STATE_KEY = "video_source_key"
BUFFER_START_STATE_KEY = "video_buffer_start_seconds"
VIDEO_START_OFFSET_STATE_KEY = "video_start_offset_seconds"
VIDEO_FILE_START_STATE_KEY = "video_file_start_seconds"
VIDEO_RENDER_TOKEN_KEY = "video_render_token"
LAST_RENDERED_TOKEN_KEY = "video_last_rendered_token"
LAST_RENDERED_PLAYING_KEY = "video_last_playing_state"
PLAYBACK_REFRESH_SECONDS = 0.5
ACTIVE_INTERVENTION_KEY = "active_concept_intervention"
LAST_RENDERED_SOURCE_KEY = "video_last_rendered_source"
SELECTED_PREDICTION_ROW_KEY = "selected_prediction_row_id"
TARGET_HORIZON_KEY = "intervention_target_horizon"
TARGET_LABEL_KEY = "intervention_target_label"
TARGET_LABEL_CONTEXT_KEY = "intervention_target_label_context"
ACTIVITY_FEEDBACK_LABEL_KEY = "activity_feedback_source_label"
ACTIVITY_FEEDBACK_LABEL_CONTEXT_KEY = "activity_feedback_source_label_context"
ACTIVITY_FEEDBACK_ROLLOUT_HORIZON = 3
INTERVENTION_MODE_KEY = "concept_intervention_mode"
GRAPH_SELECTED_CONCEPTS_KEY = "scene_graph_selected_concepts"
GRAPH_INTERVENTION_THRESHOLD_KEY = "scene_graph_intervention_threshold"
LIVE_PREDICTION_CACHE_KEY = "live_prediction_cache"
VIDEO_EVENT_PROCESSED_KEY = "video_event_processed"
COUNTERFACTUAL_RANKING_CACHE_KEY = "counterfactual_concept_ranking_cache"
GRAPH_CHANGE_EPSILON = 0.0005
GRAPH_CHANGED_NODE_LIMIT = 5
GRAPH_MAX_CONCEPTS = 10
GRAPH_DEFAULT_CONCEPTS = 6
GRAPH_EDGE_MIN_PENWIDTH = 1.1
GRAPH_EDGE_MAX_PENWIDTH = 4.2
GRAPH_EDGE_REFERENCE_WEIGHT = 0.05
DATASET_ROOT = configured_dataset_root()

VIDEO_EVENT_BRIDGE = st.components.v2.component(
    "concept_forecasting_video_events",
    html="<span aria-hidden=\"true\"></span>",
    css=":host { display: block; height: 0; overflow: hidden; }",
    js="""
    export default function(component) {
      const { data, setTriggerValue } = component;
      let video = null;
      let retry = null;
      let observer = null;
      let lastUserInteractionAt = 0;
      let localPlaying = Boolean(data.playing);

      const markUserInteraction = () => {
        lastUserInteractionAt = performance.now();
      };

      const wasRecentlyUserInitiated = () => (
        performance.now() - lastUserInteractionAt < 2000
      );

      const absolutePosition = () => {
        const current = video && Number.isFinite(video.currentTime) ? video.currentTime : 0;
        return Math.max(0, Number(data.offsetSeconds || 0) + current);
      };

      const emit = (kind, extra = {}) => {
        setTriggerValue('event', {
          kind,
          position: absolutePosition(),
          sourceKey: String(data.sourceKey || ''),
          id: `${Date.now()}-${Math.random()}`,
          ...extra,
        });
      };

      const onPause = () => {
        if (data.playing && video && !video.ended && wasRecentlyUserInitiated()) {
          localPlaying = false;
          emit('pause');
          lastUserInteractionAt = 0;
        }
      };
      const onPlay = () => {
        if (!data.playing && wasRecentlyUserInitiated()) {
          localPlaying = true;
          emit('play');
          lastUserInteractionAt = 0;
        }
      };
      const onEnded = () => {
        localPlaying = false;
        emit('ended');
      };
      const onSeeked = () => {
        const position = absolutePosition();
        if (
          wasRecentlyUserInitiated()
          && Math.abs(position - Number(data.startPosition || 0)) > 0.35
        ) {
          emit('seek', { paused: Boolean(video && video.paused) });
          lastUserInteractionAt = 0;
        }
      };

      const synchronizePlayback = () => {
        if (!video) return;
        if (localPlaying && video.paused && !video.ended) {
          video.play().catch(() => {});
        } else if (!localPlaying && !video.paused) {
          video.pause();
        }
      };

      const detach = () => {
        if (!video) return;
        video.removeEventListener('pause', onPause);
        video.removeEventListener('play', onPlay);
        video.removeEventListener('ended', onEnded);
        video.removeEventListener('seeked', onSeeked);
        video.removeEventListener('pointerdown', markUserInteraction);
        video.removeEventListener('pointerup', markUserInteraction);
        video.removeEventListener('keydown', markUserInteraction);
        video.removeEventListener('keyup', markUserInteraction);
        video = null;
        lastUserInteractionAt = 0;
      };

      const attach = () => {
        const videos = Array.from(document.querySelectorAll('video'))
          .filter((candidate) => candidate.offsetParent !== null);
        const candidate = videos.length ? videos[videos.length - 1] : null;
        if (!candidate) return;
        if (candidate === video) {
          synchronizePlayback();
          return;
        }
        detach();
        video = candidate;
        video.addEventListener('pause', onPause);
        video.addEventListener('play', onPlay);
        video.addEventListener('ended', onEnded);
        video.addEventListener('seeked', onSeeked);
        video.addEventListener('pointerdown', markUserInteraction);
        video.addEventListener('pointerup', markUserInteraction);
        video.addEventListener('keydown', markUserInteraction);
        video.addEventListener('keyup', markUserInteraction);
        synchronizePlayback();
      };

      observer = new MutationObserver(attach);
      observer.observe(document.body, { childList: true, subtree: true });
      retry = window.setInterval(attach, 250);
      attach();

      return () => {
        window.clearInterval(retry);
        observer.disconnect();
        detach();
      };
    }
    """,
)


def resolve_dataset_root(value: str | Path | None = None) -> Path:
    requested = Path(value).expanduser() if value is not None else None
    placeholders = {
        Path("/absolute/path/to/datasets"),
        Path("/path/to/datasets"),
        Path("data/datasets"),
    }
    if requested is not None and requested not in placeholders:
        return requested

    candidates = []
    configured = os.environ.get("TRACE_DATASET_ROOT")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path not in placeholders:
            candidates.append(configured_path)
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "datasets",
            PROJECT_ROOT.parent.parent / "Data" / "Datasets",
        ]
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "runs" / "train_models")),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--device", default=os.environ.get("UI_DEVICE", "cpu"))
    args = parser.parse_known_args()[0]
    args.dataset_root = resolve_dataset_root(args.dataset_root)
    return args


ARGS = parse_args()


@st.cache_data(show_spinner=False)
def cached_discover(model_root: str) -> tuple[list[CheckpointRecord], list[str]]:
    return discover_checkpoints(model_root)


@st.cache_resource(show_spinner="Loading checkpoint and dataset...")
def cached_workspace(checkpoint_path: str, device: str, dataset_root: str):
    return load_workspace(checkpoint_path, device=device, dataset_root=dataset_root)


@st.cache_data(show_spinner=False)
def cached_image_frame_paths(video_path: str) -> tuple[str, ...]:
    path = Path(video_path)
    if not path.is_dir():
        return ()
    return tuple(
        str(child)
        for child in sorted(path.iterdir())
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )


@st.cache_data(show_spinner="Preparing video preview...")
def cached_frame_clip(
    video_path: str,
    start_seconds: float,
    buffer_seconds: float = PLAYBACK_BUFFER_SECONDS,
    fps: float = 12.0,
    max_height: int = VIDEO_HEIGHT,
) -> bytes | None:
    path = Path(video_path)
    if not path.is_dir():
        return None
    frames = cached_image_frame_paths(str(path))
    if not frames:
        return None

    source_fps = max(float(fps), 1.0)
    start_index = int(max(0.0, float(start_seconds)) * source_fps)
    stop_index = min(len(frames), start_index + int(round(float(buffer_seconds) * source_fps)))
    selected = frames[start_index:stop_index]
    if not selected:
        selected = frames[-1:]

    first = cv2.imread(str(selected[0]))
    if first is None:
        return None
    height, width = first.shape[:2]
    if height > int(max_height):
        scale = float(max_height) / float(height)
        width = int(round(width * scale))
        height = int(round(height * scale))
    width = max(2, width - width % 2)
    height = max(2, height - height % 2)
    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = Path(handle.name)
    handle.close()
    encoder = None
    frames_written = 0
    try:
        encoder = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                f"{source_fps:g}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libopenh264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert encoder.stdin is not None
        for frame_path in selected:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
            frames_written += 1
        encoder.stdin.close()
        assert encoder.stderr is not None
        error = encoder.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = encoder.wait()
        if return_code != 0 or frames_written == 0:
            raise RuntimeError(error or "ffmpeg produced no playable frames")
        return output_path.read_bytes()
    except (BrokenPipeError, OSError, RuntimeError) as exc:
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        st.warning(f"Could not encode a browser-compatible video preview: {exc}")
        return None
    finally:
        output_path.unlink(missing_ok=True)


def canonical_dataset_name(value: object) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "breakfast": "breakfast",
        "gtea": "gtea_gaze",
        "gtea_gaze": "gtea_gaze",
        "mpii": "mpii_cooking_2",
        "mpii_cooking": "mpii_cooking_2",
        "mpii_cooking_2": "mpii_cooking_2",
        "barista": "barista",
        "epic": "epic_kitchens_100",
        "epic_kitchens": "epic_kitchens_100",
        "epic_kitchens_100": "epic_kitchens_100",
    }
    return aliases.get(key, key)


def resolve_video_path(
    raw_path: object,
    *,
    dataset: object,
    video_id: object,
    args: Mapping[str, Any] | None = None,
    dataset_root: str | Path = DATASET_ROOT,
) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.exists():
        return path

    args = dict(args or {})
    candidates: list[Path] = []
    data_root_value = args.get("data_root")
    if data_root_value:
        data_root = Path(str(data_root_value)).expanduser()
        candidates.append(data_root / path)
        if path.parts and path.parts[0] == "..":
            candidates.append(data_root / Path(*path.parts[1:]))
    embedding_path_value = args.get("embedding_path")
    if embedding_path_value:
        embedding_path = Path(str(embedding_path_value)).expanduser()
        candidates.append(embedding_path.parent / path)
        candidates.append(embedding_path.parent.parent / path)
    candidates.append(PROJECT_ROOT / path)

    dataset_key = canonical_dataset_name(dataset)
    root = Path(dataset_root).expanduser()
    dataset_roots = {
        "breakfast": (root / "Breakfast/Video_data",),
        "gtea_gaze": (
            root / "GTEA_Gaze/Image_data",
            root / "GTEA_Gaze/cropped_clips_mp4",
        ),
        "mpii_cooking_2": (root / "MPII_Cooking_2/Image_data",),
        "barista": (root / "Barista/Image_data",),
        "epic_kitchens_100": (root / "EPIC-KITCHENS-100/EPIC-KITCHENS",),
    }.get(dataset_key, ())
    path_parts = list(path.parts)
    if "Video_data" in path_parts:
        video_data_index = path_parts.index("Video_data")
        suffix = Path(*path_parts[video_data_index + 1 :])
        for root in dataset_roots:
            if root.name == "Video_data":
                candidates.append(root / suffix)
    if "Image_data" in path_parts:
        image_data_index = path_parts.index("Image_data")
        suffix = Path(*path_parts[image_data_index + 1 :])
        for root in dataset_roots:
            if root.name == "Image_data":
                candidates.append(root / suffix)

    video_id_text = str(video_id or "").strip()
    if video_id_text:
        for root in dataset_roots:
            candidates.append(root / video_id_text)
            if path.suffix:
                candidates.append(root / path.name)
            if dataset_key == "epic_kitchens_100":
                participant = video_id_text.split("_", 1)[0]
                candidates.append(root / participant / "videos" / f"{video_id_text}.MP4")
                candidates.append(root / participant / "videos" / f"{video_id_text}.mp4")

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.expanduser().resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.exists():
            return normalized
    return path


def video_timing_metadata(
    workspace,
    raw_path: object,
    resolved_path: Path,
    video_id: object,
) -> dict[str, object]:
    metadata = workspace.preprocessed_data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return {}
    candidates = [str(raw_path), str(resolved_path), str(video_id)]
    candidates.extend([Path(str(raw_path)).name, resolved_path.name])
    return {
        "window_spans": lookup_video_metadata(metadata.get("video_window_spans"), candidates),
        "video_meta": lookup_video_metadata(metadata.get("video_meta"), candidates),
    }


def lookup_video_metadata(values: object, candidates: list[str]) -> object | None:
    if not isinstance(values, Mapping):
        return None
    for candidate in candidates:
        if candidate in values:
            return values[candidate]
    names = {Path(candidate).name for candidate in candidates if candidate}
    ids = {str(candidate) for candidate in candidates if candidate}
    for key, value in values.items():
        key_text = str(key)
        if key_text in ids or Path(key_text).name in names:
            return value
    return None


def main() -> None:
    st.set_page_config(page_title="TRACE Intervention Viewer", layout="wide")
    inject_page_style()
    st.title("TRACE Intervention Viewer")

    record, device = checkpoint_picker()
    if record is None:
        return

    try:
        workspace = cached_workspace(str(record.path), device, str(ARGS.dataset_root))
    except Exception as exc:  # noqa: BLE001 - keep checkpoint failures visible in the app.
        st.error(f"Could not load checkpoint: {exc}")
        return

    split, video_index = instance_picker(workspace)
    raw_path = workspace.preprocessed_data[split]["video_paths"][video_index]
    video_id = workspace.preprocessed_data[split]["video_ids"][video_index]
    path = resolve_video_path(
        raw_path,
        dataset=record.dataset,
        video_id=video_id,
        args=record.args,
        dataset_root=ARGS.dataset_root,
    )
    timing = video_timing_metadata(workspace, raw_path, path, video_id)
    source = video_source_info(
        path,
        int(workspace.standardized_splits[split]["lengths"][video_index]),
        window_spans=timing.get("window_spans"),
        video_meta=timing.get("video_meta"),
    )
    ensure_video_state(f"{split}:{video_index}:{path}", source)
    apply_pending_slider_position()

    playing = bool(st.session_state.get(PLAYING_STATE_KEY, False))
    if not playing:
        apply_slider_seek_if_needed(source)
    position_seconds = current_position_seconds(source)
    timestep = source_position_to_timestep(position_seconds, source)

    instance = None
    intervention = None
    outputs = None
    baseline_outputs = None
    prediction_rows: list[dict[str, object]] = []
    selected_row = None
    if not playing:
        instance = select_instance(workspace, split=split, video_index=video_index, timestep=timestep)
        instance["video_path"] = str(path)
        intervention = active_intervention_for_instance(instance)
        outputs = forward_outputs(workspace, instance, intervention=intervention)
        baseline_outputs = forward_outputs(workspace, instance)
        prediction_rows = prediction_rows_for_instance(workspace, instance, outputs)
        ensure_selected_prediction(prediction_rows)

    video_col, label_col = st.columns([1.15, 0.85], gap="large")
    with video_col:
        video_slot = st.empty()
        position_seconds = video_position_picker(source)
        render_video_pane(
            video_slot,
            path,
            source,
            position_seconds,
            playing=playing,
        )
        if not playing:
            render_playback_status(source, position_seconds, source_position_to_timestep(position_seconds, source))
    with label_col:
        if playing:
            render_playing_prediction_pane(workspace, split, video_index, source)
        else:
            selected_row = render_prediction_pane(workspace, instance, prediction_rows)

    if playing:
        return

    st.divider()
    render_checkpoint_representation_status(record, workspace)
    selected_row = render_forecast_timeline(prediction_rows, selected_row)
    target_row = render_target_controls(workspace, instance, selected_row)
    if target_row is not None:
        render_selected_prediction_panel(
            workspace,
            instance,
            baseline_outputs,
            outputs,
            target_row,
            intervention,
        )
    render_intervention_cart(workspace, instance, baseline_outputs, outputs, intervention)


@st.fragment(run_every=PLAYBACK_REFRESH_SECONDS)
def render_playing_prediction_pane(
    workspace,
    split: str,
    video_index: int,
    source: Mapping[str, Any],
) -> None:
    position_seconds = current_position_seconds(source)
    duration = float(source["duration_seconds"])
    if position_seconds >= duration:
        pause_playback(source, position_seconds=duration)
        st.rerun()

    if str(source.get("kind")) == "frame_folder" and video_buffer_needs_refresh(position_seconds):
        sync_playback_position(source, position_seconds=position_seconds)
        reset_video_buffer(position_seconds, source, force_render=True)
        st.rerun()

    timestep = source_position_to_timestep(position_seconds, source)
    cache_key = f"{st.session_state.get(SOURCE_STATE_KEY)}:{timestep}"
    cached = st.session_state.get(LIVE_PREDICTION_CACHE_KEY)
    if not isinstance(cached, dict) or cached.get("key") != cache_key:
        instance = select_instance(workspace, split=split, video_index=video_index, timestep=timestep)
        outputs = forward_outputs(workspace, instance)
        rows = prediction_rows_for_instance(workspace, instance, outputs)
        cached = {"key": cache_key, "rows": rows}
        st.session_state[LIVE_PREDICTION_CACHE_KEY] = cached
    render_live_prediction_summary(cached.get("rows", []), source, position_seconds, timestep)


def render_live_prediction_summary(
    prediction_rows: list[dict[str, object]],
    source: Mapping[str, Any],
    position_seconds: float,
    timestep: int,
) -> None:
    st.markdown("#### Live predictions")
    st.caption(
        f"{format_seconds(position_seconds)} / {format_seconds(source['duration_seconds'])}"
        f"  |  window {int(timestep) + 1}/{int(source['num_windows'])}"
    )
    if not prediction_rows:
        st.info("No predictions available for this window.")
        return

    current = next((row for row in prediction_rows if str(row.get("target")) == "activity"), prediction_rows[0])
    summary_cols = st.columns(3)
    summary_cols[0].metric("Prediction", str(current.get("prediction", "n/a")))
    summary_cols[1].metric("Ground truth", str(current.get("ground_truth", "n/a")))
    summary_cols[2].metric("Confidence", str(current.get("probability", "n/a")))
    st.dataframe(
        pd.DataFrame(prediction_rows),
        width="stretch",
        hide_index=True,
        column_order=["horizon", "prediction", "probability", "ground_truth"],
    )


def render_checkpoint_representation_status(record, workspace) -> None:
    model_hparams = dict(record.args.get("model_hparams") or {})
    state_activation = str(
        getattr(workspace.model, "st_state_activation", None)
        or model_hparams.get("st_state_activation")
        or "identity"
    )
    prediction_transform = str(
        getattr(workspace.model, "st_prediction_transform", None)
        or model_hparams.get("st_prediction_transform")
        or "identity"
    )
    if state_activation == "bounded_logit":
        st.success(
            "Selected checkpoint uses bounded graph concept states: "
            f"`st_state_activation={state_activation}`. "
            f"Classifier input transform: `{prediction_transform}`. "
            "`shared/window/forecast_refined_concepts` should be in `[0, 1]`; "
            "`activity_repr` and `forecast_repr` may be outside `[0, 1]`."
        )
    else:
        st.warning(
            "Selected checkpoint uses legacy unbounded graph concept states. "
            f"`st_state_activation={state_activation}`. "
            "Scene graph node scores may be negative or larger than 1."
        )
    feedback_mode = str(
        getattr(workspace.model, "st_activity_feedback_mode", "none")
    )
    if feedback_mode == "sparse_label_to_concept":
        st.success(
            "Activity-belief feedback is enabled: categorical activity beliefs feed "
            "the next concept rollout through sparse signed label-to-concept edges."
        )
    else:
        st.caption("Activity-belief feedback is disabled for this checkpoint.")


def inject_page_style() -> None:
    st.markdown(
        f"""
        <style>
        video {{
            max-height: {VIDEO_HEIGHT}px;
            object-fit: contain;
            background: #111;
        }}
        .block-container {{
            padding-top: 1.6rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def checkpoint_picker() -> tuple[CheckpointRecord | None, str]:
    with st.sidebar.expander("Checkpoint and model", expanded=False):
        model_root = st.text_input("Model root", value=str(ARGS.model_root))
        device = st.text_input("Device", value=str(ARGS.device))
        if st.button("Refresh checkpoints"):
            cached_discover.clear()

        records, warnings = cached_discover(model_root)
        if warnings:
            with st.expander(f"Discovery warnings ({len(warnings)})", expanded=False):
                for warning in warnings[:30]:
                    st.warning(warning)
        if not records:
            st.error(f"No checkpoints found under {model_root}")
            return None, device

        datasets = ["All"] + sorted({record.dataset for record in records})
        dataset = st.selectbox("Dataset", datasets)
        filtered = [record for record in records if dataset == "All" or record.dataset == dataset]

        backbones = ["All"] + sorted({record.backbone for record in filtered})
        backbone = st.selectbox("Backbone", backbones)
        filtered = [record for record in filtered if backbone == "All" or record.backbone == backbone]

        methods = ["All"] + sorted({record.base_method for record in filtered})
        method = st.selectbox("Model type", methods)
        filtered = [record for record in filtered if method == "All" or record.base_method == method]

        labels = [record.label for record in filtered]
        selected = st.selectbox("Trained model", labels)
        return filtered[labels.index(selected)], device


def instance_picker(workspace) -> tuple[str, int]:
    st.sidebar.header("Test video")
    splits = split_names(workspace)
    split = st.sidebar.selectbox("Split", splits, index=splits.index("test") if "test" in splits else 0)

    videos = video_options(workspace, split)
    if not videos:
        st.sidebar.error(f"No videos available for split {split!r}.")
        st.stop()
    video_label = st.sidebar.selectbox("Video", [label for label, _ in videos])
    video_index = dict(videos)[video_label]
    return split, video_index


def ensure_video_state(source_key: str, source: Mapping[str, Any]) -> None:
    duration = float(source["duration_seconds"])
    if st.session_state.get(SOURCE_STATE_KEY) == source_key:
        return
    st.session_state[SOURCE_STATE_KEY] = source_key
    st.session_state[POSITION_STATE_KEY] = 0.0
    st.session_state[POSITION_SLIDER_KEY] = 0.0
    st.session_state.pop(PENDING_SLIDER_POSITION_KEY, None)
    st.session_state[PLAYING_STATE_KEY] = False
    st.session_state[PLAYBACK_STARTED_STATE_KEY] = time.monotonic()
    st.session_state[BUFFER_START_STATE_KEY] = 0.0
    st.session_state[VIDEO_START_OFFSET_STATE_KEY] = 0.0
    st.session_state[VIDEO_FILE_START_STATE_KEY] = 0.0
    st.session_state[VIDEO_RENDER_TOKEN_KEY] = 0
    st.session_state.pop(LAST_RENDERED_SOURCE_KEY, None)
    st.session_state.pop(LAST_RENDERED_TOKEN_KEY, None)
    st.session_state.pop(LAST_RENDERED_PLAYING_KEY, None)
    st.session_state.pop(SELECTED_PREDICTION_ROW_KEY, None)
    st.session_state.pop(LIVE_PREDICTION_CACHE_KEY, None)
    st.session_state.pop(ACTIVE_INTERVENTION_KEY, None)
    st.session_state.pop(COUNTERFACTUAL_RANKING_CACHE_KEY, None)
    st.session_state["video_duration_seconds"] = duration


def current_position_seconds(source: Mapping[str, Any], *, now: float | None = None) -> float:
    duration = float(source["duration_seconds"])
    position = float(st.session_state.get(POSITION_STATE_KEY, 0.0))
    if st.session_state.get(PLAYING_STATE_KEY, False):
        current = time.monotonic() if now is None else float(now)
        started = float(st.session_state.get(PLAYBACK_STARTED_STATE_KEY, current))
        position += max(0.0, current - started)
    return min(duration, max(0.0, position))


def sync_playback_position(
    source: Mapping[str, Any],
    *,
    position_seconds: float | None = None,
    now: float | None = None,
) -> float:
    current = time.monotonic() if now is None else float(now)
    position = current_position_seconds(source, now=current) if position_seconds is None else float(position_seconds)
    duration = float(source["duration_seconds"])
    position = min(duration, max(0.0, position))
    st.session_state[POSITION_STATE_KEY] = position
    st.session_state[PENDING_SLIDER_POSITION_KEY] = position
    st.session_state[PLAYBACK_STARTED_STATE_KEY] = current
    return position


def apply_pending_slider_position() -> None:
    if PENDING_SLIDER_POSITION_KEY not in st.session_state:
        return
    st.session_state[POSITION_SLIDER_KEY] = float(st.session_state.pop(PENDING_SLIDER_POSITION_KEY))


def pause_playback(source: Mapping[str, Any], *, position_seconds: float | None = None) -> None:
    position = sync_playback_position(source, position_seconds=position_seconds)
    st.session_state[PLAYING_STATE_KEY] = False
    reset_video_buffer(position, source, force_render=True)


def resume_playback(source: Mapping[str, Any], *, position_seconds: float | None = None) -> None:
    duration = float(source["duration_seconds"])
    position = float(st.session_state.get(POSITION_STATE_KEY, 0.0)) if position_seconds is None else float(position_seconds)
    position = min(duration, max(0.0, position))
    if position >= duration:
        position = 0.0
    st.session_state[POSITION_STATE_KEY] = position
    st.session_state[PENDING_SLIDER_POSITION_KEY] = position
    st.session_state[PLAYBACK_STARTED_STATE_KEY] = time.monotonic()
    st.session_state[PLAYING_STATE_KEY] = True
    st.session_state.pop(ACTIVE_INTERVENTION_KEY, None)
    st.session_state.pop(LIVE_PREDICTION_CACHE_KEY, None)
    reset_video_buffer(position, source, force_render=True)


def request_video_render() -> None:
    st.session_state[VIDEO_RENDER_TOKEN_KEY] = int(st.session_state.get(VIDEO_RENDER_TOKEN_KEY, 0)) + 1


def apply_slider_seek_if_needed(source: Mapping[str, Any]) -> None:
    if POSITION_SLIDER_KEY not in st.session_state:
        return
    duration = float(source["duration_seconds"])
    slider_position = min(duration, max(0.0, float(st.session_state[POSITION_SLIDER_KEY])))
    previous = current_position_seconds(source)
    if abs(slider_position - previous) <= 1e-6:
        return
    st.session_state[POSITION_STATE_KEY] = slider_position
    st.session_state[POSITION_SLIDER_KEY] = slider_position
    st.session_state[PLAYBACK_STARTED_STATE_KEY] = time.monotonic()
    reset_video_buffer(slider_position, source, force_render=True)


def video_position_picker(source: Mapping[str, Any]) -> float:
    duration = float(source["duration_seconds"])
    playing = bool(st.session_state.get(PLAYING_STATE_KEY, False))
    if POSITION_SLIDER_KEY not in st.session_state:
        st.session_state[POSITION_SLIDER_KEY] = float(st.session_state.get(POSITION_STATE_KEY, 0.0))

    controls = st.columns([0.22, 0.22, 0.56])
    controls[0].button(
        "Pause" if playing else "Play",
        width="stretch",
        on_click=toggle_playback,
    )
    controls[1].button("Restart", width="stretch", on_click=restart_playback)

    if playing:
        return current_position_seconds(source)

    slider_value = st.slider(
        "Video position",
        min_value=0.0,
        max_value=max(duration, 0.01),
        step=max(0.01, duration / 1000.0),
        format="%.2f s",
        key=POSITION_SLIDER_KEY,
    )
    position_seconds = min(duration, max(0.0, float(slider_value)))
    previous = float(st.session_state.get(POSITION_STATE_KEY, 0.0))
    if abs(position_seconds - previous) > 1e-6:
        st.session_state[POSITION_STATE_KEY] = position_seconds
        st.session_state[PLAYBACK_STARTED_STATE_KEY] = time.monotonic()
        reset_video_buffer(position_seconds, source, force_render=True)
        st.rerun()

    return float(st.session_state.get(POSITION_STATE_KEY, position_seconds))


def render_playback_status(source: Mapping[str, Any], position_seconds: float, timestep: int) -> None:
    duration = float(source["duration_seconds"])
    frame_count = int(source.get("frame_count", 0))
    if frame_count > 0:
        frame_start, frame_end = timestep_frame_range(
            frame_count,
            int(source["num_windows"]),
            timestep,
            window_spans=source.get("window_spans"),
            fps=float(source.get("fps", 1.0)),
        )
        frame_text = f"{frame_start}-{frame_end}"
    else:
        frame_text = "n/a"
    status_cols = st.columns(4)
    status_cols[0].metric("Position", f"{format_seconds(position_seconds)} / {format_seconds(duration)}")
    status_cols[1].metric("Timestep", f"{int(timestep) + 1} / {int(source['num_windows'])}")
    status_cols[2].metric("Frames", frame_text)
    status_cols[3].metric("State", "playing" if st.session_state.get(PLAYING_STATE_KEY, False) else "paused")


def toggle_playback() -> None:
    duration = float(st.session_state.get("video_duration_seconds", 0.0))
    source = {"duration_seconds": duration}
    was_playing = bool(st.session_state.get(PLAYING_STATE_KEY, False))
    if was_playing:
        pause_playback(source)
    else:
        resume_playback(source)
    request_video_render()


def restart_playback() -> None:
    st.session_state[POSITION_STATE_KEY] = 0.0
    st.session_state[POSITION_SLIDER_KEY] = 0.0
    st.session_state[PLAYING_STATE_KEY] = False
    st.session_state[PLAYBACK_STARTED_STATE_KEY] = time.monotonic()
    st.session_state[BUFFER_START_STATE_KEY] = 0.0
    st.session_state[VIDEO_START_OFFSET_STATE_KEY] = 0.0
    st.session_state[VIDEO_FILE_START_STATE_KEY] = 0.0
    st.session_state.pop(ACTIVE_INTERVENTION_KEY, None)
    st.session_state.pop(LIVE_PREDICTION_CACHE_KEY, None)
    request_video_render()


def reset_video_buffer(
    position_seconds: float,
    source: Mapping[str, Any],
    *,
    force_render: bool = False,
) -> None:
    previous_start = float(st.session_state.get(BUFFER_START_STATE_KEY, 0.0))
    buffer_start = smooth_buffer_start_seconds(position_seconds, float(source["duration_seconds"]))
    buffer_changed = abs(buffer_start - previous_start) > 1e-6
    st.session_state[BUFFER_START_STATE_KEY] = buffer_start
    if force_render or buffer_changed:
        st.session_state[VIDEO_START_OFFSET_STATE_KEY] = max(0.0, float(position_seconds) - buffer_start)
        st.session_state[VIDEO_FILE_START_STATE_KEY] = max(0.0, float(position_seconds))
        request_video_render()


def video_buffer_needs_refresh(position_seconds: float) -> bool:
    buffer_start = float(st.session_state.get(BUFFER_START_STATE_KEY, 0.0))
    buffer_end = buffer_start + PLAYBACK_BUFFER_SECONDS
    position_seconds = float(position_seconds)
    return (
        position_seconds < buffer_start
        or position_seconds > buffer_end - BUFFER_REFRESH_MARGIN_SECONDS
    )


def render_video_pane(
    video_slot,
    path: Path,
    source: Mapping[str, Any],
    position_seconds: float,
    playing: bool,
) -> None:
    if not path.exists():
        video_slot.warning(f"Video path does not exist: {path}")
        return

    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        last_source = st.session_state.get(LAST_RENDERED_SOURCE_KEY)
        render_token = int(st.session_state.get(VIDEO_RENDER_TOKEN_KEY, 0))
        last_token = st.session_state.get(LAST_RENDERED_TOKEN_KEY)
        last_playing = st.session_state.get(LAST_RENDERED_PLAYING_KEY)
        should_update = last_source != str(path) or last_token != render_token or last_playing != playing
        if should_update:
            st.session_state[VIDEO_FILE_START_STATE_KEY] = float(position_seconds)
            st.session_state[LAST_RENDERED_SOURCE_KEY] = str(path)
            st.session_state[LAST_RENDERED_TOKEN_KEY] = render_token
            st.session_state[LAST_RENDERED_PLAYING_KEY] = playing
        video_slot.video(
            str(path),
            start_time=float(st.session_state.get(VIDEO_FILE_START_STATE_KEY, position_seconds)),
            autoplay=False,
            muted=True,
        )
        render_video_event_bridge(
            source,
            start_position=float(st.session_state.get(VIDEO_FILE_START_STATE_KEY, position_seconds)),
            offset_seconds=0.0,
            playing=playing,
        )
        return

    if path.is_dir():
        buffer_start = float(st.session_state.get(BUFFER_START_STATE_KEY, 0.0))
        render_token = int(st.session_state.get(VIDEO_RENDER_TOKEN_KEY, 0))
        last_source = st.session_state.get(LAST_RENDERED_SOURCE_KEY)
        last_playing = st.session_state.get(LAST_RENDERED_PLAYING_KEY)
        current_source = f"{path}:{buffer_start:.3f}:{render_token}"
        if last_source != current_source or last_playing != playing:
            st.session_state[LAST_RENDERED_SOURCE_KEY] = current_source
            st.session_state[LAST_RENDERED_PLAYING_KEY] = playing
            st.session_state[LAST_RENDERED_TOKEN_KEY] = render_token
        clip = cached_frame_clip(
            str(path),
            start_seconds=buffer_start,
            buffer_seconds=min(
                PLAYBACK_BUFFER_SECONDS,
                max(0.01, float(source["duration_seconds"]) - buffer_start),
            ),
            fps=float(source["fps"]),
        )
        if clip:
            video_slot.video(
                clip,
                format="video/mp4",
                start_time=float(st.session_state.get(VIDEO_START_OFFSET_STATE_KEY, 0.0)),
                autoplay=False,
                muted=True,
            )
            render_video_event_bridge(
                source,
                start_position=position_seconds,
                offset_seconds=buffer_start,
                playing=playing,
            )
        else:
            video_slot.warning("Could not build a playable preview from this frame folder.")
        return

    video_slot.info(f"Unsupported video source: {path}")


def render_video_event_bridge(
    source: Mapping[str, Any],
    *,
    start_position: float,
    offset_seconds: float,
    playing: bool,
) -> None:
    result = VIDEO_EVENT_BRIDGE(
        key="video_event_bridge",
        data={
            "sourceKey": str(st.session_state.get(SOURCE_STATE_KEY, "")),
            "startPosition": float(start_position),
            "offsetSeconds": float(offset_seconds),
            "playing": bool(playing),
        },
        on_event_change=lambda: None,
        height=0,
    )
    event = getattr(result, "event", None)
    if isinstance(event, Mapping):
        handle_video_event(event, source)


def handle_video_event(event: Mapping[str, object], source: Mapping[str, Any]) -> None:
    event_id = str(event.get("id", ""))
    if not event_id or event_id == st.session_state.get(VIDEO_EVENT_PROCESSED_KEY):
        return
    if str(event.get("sourceKey", "")) != str(st.session_state.get(SOURCE_STATE_KEY, "")):
        return
    st.session_state[VIDEO_EVENT_PROCESSED_KEY] = event_id
    duration = float(source["duration_seconds"])
    try:
        position = min(duration, max(0.0, float(event.get("position", 0.0))))
    except (TypeError, ValueError):
        return

    kind = str(event.get("kind", ""))
    if kind == "pause":
        pause_playback(source, position_seconds=position)
    elif kind == "play":
        resume_playback(source, position_seconds=position)
    elif kind == "seek":
        sync_playback_position(source, position_seconds=position)
        st.session_state[PLAYING_STATE_KEY] = not bool(event.get("paused", False))
        if st.session_state[PLAYING_STATE_KEY]:
            st.session_state.pop(ACTIVE_INTERVENTION_KEY, None)
        reset_video_buffer(position, source, force_render=True)
    elif kind == "ended":
        pause_playback(source, position_seconds=duration)
    else:
        return
    st.session_state.pop(LIVE_PREDICTION_CACHE_KEY, None)
    request_video_render()
    st.rerun()


def active_intervention_for_instance(instance: Mapping[str, Any]) -> dict[str, object] | None:
    intervention = st.session_state.get(ACTIVE_INTERVENTION_KEY)
    if not isinstance(intervention, dict):
        return None
    mode = str(st.session_state.get(INTERVENTION_MODE_KEY, intervention.get("mode", "input")))
    expected = (str(instance["split"]), int(instance["video_index"]), int(instance["timestep"]))
    found = (
        str(intervention.get("split")),
        int(intervention.get("video_index", -1)),
        int(intervention.get("timestep", -1)),
    )
    if found != expected:
        return None
    
    # Get the time dimension size from the concepts array
    concepts_array = instance.get("concepts")
    if isinstance(concepts_array, (list, tuple)):
        concepts_time_size = len(concepts_array)
    else:
        # Assume it's a numpy array or similar
        try:
            import numpy as np
            concepts_array = np.asarray(concepts_array)
            concepts_time_size = concepts_array.shape[0] if concepts_array.ndim >= 1 else 0
        except Exception:
            concepts_time_size = 0
    
    # Validate that all intervention time_idx values are in bounds for this instance
    items = []
    for item in intervention.get("items", []):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("item_type", item.get("type", "concept"))).lower()
        if item_type == "edge":
            items.append(dict(item))
            continue
        if item_type == "activity":
            if all(key in item for key in ("step", "class_idx", "probability")):
                items.append(
                    {
                        **dict(item),
                        "item_type": "activity",
                        "step": int(item["step"]),
                        "class_idx": int(item["class_idx"]),
                        "probability": float(item["probability"]),
                    }
                )
            continue
        if "rollout_step" in item and "concept_idx" in item and "value" in item:
            rollout_step = int(item["rollout_step"])
            if rollout_step >= 1:
                items.append(
                    {
                        "item_type": "concept",
                        "rollout_step": rollout_step,
                        "concept_idx": int(item["concept_idx"]),
                        "value": float(item["value"]),
                        "delta": None,
                    }
                )
            continue
        if "time_idx" not in item or "concept_idx" not in item or "value" not in item:
            continue
        if 0 <= int(item["time_idx"]) < concepts_time_size:
            items.append(
                {
                    "time_idx": int(item["time_idx"]),
                    "concept_idx": int(item["concept_idx"]),
                    "value": float(item["value"]),
                    "delta": None,
                }
            )
    if items:
        return {"items": items, "mode": mode}
    if "time_idx" not in intervention or "concept_idx" not in intervention:
        return None
    
    # Validate single intervention time_idx
    time_idx = int(intervention["time_idx"])
    if time_idx < 0 or time_idx >= concepts_time_size:
        return None
    
    return {
        "time_idx": time_idx,
        "concept_idx": int(intervention["concept_idx"]),
        "value": float(intervention["value"]),
        "delta": None,
        "mode": mode,
    }


def set_active_intervention(
    workspace,
    instance: Mapping[str, Any],
    row: Mapping[str, Any],
    value: float,
    setting: str,
    target_row: Mapping[str, object],
    baseline_outputs: Mapping[str, object],
) -> None:
    existing = st.session_state.get(ACTIVE_INTERVENTION_KEY)
    base = {
        "split": str(instance["split"]),
        "video_index": int(instance["video_index"]),
        "timestep": int(instance["timestep"]),
        "mode": str(st.session_state.get(INTERVENTION_MODE_KEY, "input")),
    }
    items = []
    if isinstance(existing, dict):
        existing_key = (
            str(existing.get("split")),
            int(existing.get("video_index", -1)),
            int(existing.get("timestep", -1)),
        )
        current_key = (base["split"], base["video_index"], base["timestep"])
        if existing_key == current_key:
            items = [dict(item) for item in existing.get("items", []) if isinstance(item, dict)]
            if not items and "time_idx" in existing and "concept_idx" in existing:
                items = [dict(existing)]

    concept_idx = int(row["concept_idx"])
    rollout_step = row.get("rollout_step")
    if rollout_step is not None:
        rollout_step = int(rollout_step)
        if rollout_step < 1:
            st.warning("Future concept interventions require rollout step t+1 or later.")
            return
        items = [
            item for item in items
            if not (
                int(item.get("rollout_step", -1)) == rollout_step
                and int(item.get("concept_idx", -1)) == concept_idx
            )
        ]
        location = {
            "item_type": "concept",
            "rollout_step": rollout_step,
        }
    else:
        time_idx = int(row["history_t"])

        concepts_array = np.asarray(instance.get("concepts"))
        concepts_time_size = concepts_array.shape[0] if concepts_array.ndim >= 1 else 0
        if time_idx < 0 or time_idx >= concepts_time_size:
            st.warning(f"Cannot intervene on concept at history index {time_idx}: out of bounds (valid range: 0-{concepts_time_size-1})")
            return

        items = [
            item for item in items
            if not (int(item.get("time_idx", -1)) == time_idx and int(item.get("concept_idx", -1)) == concept_idx)
        ]
        location = {
            "time_idx": time_idx,
            "original_t": row.get("original_t"),
        }
    items.append({
        **location,
        "concept_idx": int(row["concept_idx"]),
        "concept": str(row["concept"]),
        "value": float(value),
        "setting": str(setting),
        "target": str(target_row["target"]),
        "horizon_idx": target_row.get("horizon_idx"),
        "horizon": str(target_row["horizon"]),
        "class_idx": int(target_row["class_idx"]),
        "target_label": label_name(workspace, target_row.get("class_idx")) or str(target_row.get("target_label", "")),
        "before_probability": selected_class_probability(
            workspace,
            baseline_outputs,
            str(target_row["target"]),
            int(target_row["class_idx"]),
            None if target_row.get("horizon_idx") is None or pd.isna(target_row.get("horizon_idx")) else int(target_row["horizon_idx"]),
        ),
    })
    st.session_state[ACTIVE_INTERVENTION_KEY] = {**base, "items": items}


def set_active_edge_intervention(
    instance: Mapping[str, Any],
    row: Mapping[str, object],
    setting: str,
    *,
    edge_scale: float | None = None,
    edge_value: float | None = None,
) -> None:
    if (edge_scale is None) == (edge_value is None):
        raise ValueError("Provide exactly one of edge_scale or edge_value.")
    existing = st.session_state.get(ACTIVE_INTERVENTION_KEY)
    base = {
        "split": str(instance["split"]),
        "video_index": int(instance["video_index"]),
        "timestep": int(instance["timestep"]),
        "mode": str(st.session_state.get(INTERVENTION_MODE_KEY, "input")),
    }
    items = []
    if isinstance(existing, dict):
        existing_key = (
            str(existing.get("split")),
            int(existing.get("video_index", -1)),
            int(existing.get("timestep", -1)),
        )
        current_key = (base["split"], base["video_index"], base["timestep"])
        if existing_key == current_key:
            items = [dict(item) for item in existing.get("items", []) if isinstance(item, dict)]
            if not items and "time_idx" in existing and "concept_idx" in existing:
                items = [dict(existing)]

    edge_kind = str(row.get("kind", "spatial"))
    # Automatically displayed rows may aggregate several branches. Preserve the
    # historical quick-edit behavior (all branches); the manual editor supplies
    # an explicit intervention_branch when the user chooses one branch.
    branch = str(row.get("intervention_branch", "all"))
    layer_index = row.get("layer_index")
    source_idx = int(row["source_idx"])
    target_idx = int(row["target_idx"])
    items = [
        item for item in items
        if not (
            str(item.get("item_type", item.get("type", ""))).lower() == "edge"
            and str(item.get("edge_kind", "")) == edge_kind
            and str(item.get("branch", "all")) == branch
            and item.get("layer_index") == layer_index
            and int(item.get("source_idx", -1)) == source_idx
            and int(item.get("target_idx", -1)) == target_idx
        )
    ]
    item = {
        "item_type": "edge",
        "edge_kind": edge_kind,
        "branch": branch,
        "source_idx": source_idx,
        "target_idx": target_idx,
        "source": str(row.get("source", source_idx)),
        "target": str(row.get("target", target_idx)),
        "setting": str(setting),
        "base_weight": float(row.get("weight", 0.0)),
    }
    if layer_index is not None:
        item["layer_index"] = int(layer_index)
    for key in ("edge_time_idx", "source_time_idx", "target_time_idx", "original_t"):
        if row.get(key) is not None:
            item[key] = row[key]
    if edge_scale is not None:
        item["edge_scale"] = float(edge_scale)
    else:
        item["edge_value"] = float(edge_value)
    items.append(item)
    st.session_state[ACTIVE_INTERVENTION_KEY] = {**base, "items": items}


def set_active_activity_intervention(
    workspace,
    instance: Mapping[str, Any],
    step: int,
    class_idx: int,
    probability: float,
    baseline_outputs: Mapping[str, object],
) -> None:
    existing = st.session_state.get(ACTIVE_INTERVENTION_KEY)
    base = {
        "split": str(instance["split"]),
        "video_index": int(instance["video_index"]),
        "timestep": int(instance["timestep"]),
        "mode": str(st.session_state.get(INTERVENTION_MODE_KEY, "input")),
    }
    items = []
    if isinstance(existing, dict):
        existing_key = (
            str(existing.get("split")),
            int(existing.get("video_index", -1)),
            int(existing.get("timestep", -1)),
        )
        current_key = (base["split"], base["video_index"], base["timestep"])
        if existing_key == current_key:
            items = [dict(item) for item in existing.get("items", []) if isinstance(item, dict)]

    step = int(step)
    class_idx = int(class_idx)
    items = [
        item
        for item in items
        if not (
            str(item.get("item_type", item.get("type", "concept"))).lower() == "activity"
            and int(item.get("step", -1)) == step
        )
    ]
    target = "activity" if step <= 0 else "forecast"
    horizon = None if step <= 0 else step
    step_label = "t" if step == 0 else f"t{step:+d}"
    items.append(
        {
            "item_type": "activity",
            "step": step,
            "class_idx": class_idx,
            "probability": float(probability),
            "activity": label_name(workspace, class_idx) or str(class_idx),
            "setting": f"p={float(probability):.2f}",
            "target": target,
            "horizon_idx": horizon,
            "horizon": step_label,
            "target_label": label_name(workspace, class_idx) or str(class_idx),
            "before_probability": selected_class_probability(
                workspace,
                baseline_outputs,
                target,
                class_idx,
                horizon,
            ),
            "belief_before_probability": activity_belief_probability(
                baseline_outputs,
                step,
                class_idx,
            ),
        }
    )
    st.session_state[ACTIVE_INTERVENTION_KEY] = {**base, "items": items}


def activity_belief_probability(
    outputs: Mapping[str, object],
    step: int,
    class_idx: int,
) -> float | None:
    if int(step) < 0:
        by_step = outputs.get("effective_activity_probs_by_history_step")
        if not isinstance(by_step, Mapping):
            return None
        probabilities = by_step.get(int(step))
        if not torch.is_tensor(probabilities):
            return None
        try:
            return float(probabilities[0, int(class_idx)].detach().cpu().item())
        except (IndexError, RuntimeError):
            return None
    by_step = outputs.get("effective_activity_probs_by_step")
    if not isinstance(by_step, Mapping):
        return None
    probabilities = by_step.get(int(step))
    if not torch.is_tensor(probabilities):
        return None
    try:
        return float(probabilities[0, -1, int(class_idx)].detach().cpu().item())
    except (IndexError, RuntimeError):
        return None


def remove_active_intervention_item(index: int) -> None:
    existing = st.session_state.get(ACTIVE_INTERVENTION_KEY)
    if not isinstance(existing, dict):
        return
    items = [dict(item) for item in existing.get("items", []) if isinstance(item, dict)]
    if 0 <= int(index) < len(items):
        items.pop(int(index))
    if items:
        st.session_state[ACTIVE_INTERVENTION_KEY] = {**existing, "items": items}
    else:
        st.session_state.pop(ACTIVE_INTERVENTION_KEY, None)


def clear_active_intervention(row: Mapping[str, Any] | None = None) -> None:
    if row is None:
        st.session_state.pop(ACTIVE_INTERVENTION_KEY, None)
        return
    existing = st.session_state.get(ACTIVE_INTERVENTION_KEY)
    if not isinstance(existing, dict):
        return
    time_idx = int(row["history_t"])
    concept_idx = int(row["concept_idx"])
    items = [
        item for item in existing.get("items", [])
        if isinstance(item, dict)
        and not (int(item.get("time_idx", -1)) == time_idx and int(item.get("concept_idx", -1)) == concept_idx)
    ]
    if items:
        st.session_state[ACTIVE_INTERVENTION_KEY] = {**existing, "items": items}
    else:
        st.session_state.pop(ACTIVE_INTERVENTION_KEY, None)


def render_intervention_cart(
    workspace,
    instance: Mapping[str, Any],
    baseline_outputs: Mapping[str, object],
    outputs: Mapping[str, object],
    intervention: Mapping[str, object] | None,
) -> None:
    items = active_intervention_items_for_instance(instance)
    st.markdown("#### Interventions")
    if not items:
        st.caption("No active interventions for this scene.")
        return
    if isinstance(intervention, Mapping):
        st.caption(f"Mode: {str(intervention.get('mode', 'input'))}")

    summary_rows = combined_intervention_target_rows(workspace, baseline_outputs, outputs, items)
    if summary_rows:
        st.caption("Combined effect across all active interventions")
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    render_rollout_prediction_changes(workspace, baseline_outputs, outputs)

    rows = []
    for index, item in enumerate(items):
        item_type = str(item.get("item_type", item.get("type", "concept"))).lower()
        if item_type == "edge":
            branch = str(item.get("branch", "all"))
            layer = item.get("layer_index")
            scope = branch if layer is None else f"{branch}.{int(layer)}"
            occurrence = item.get("original_t", item.get("edge_time_idx"))
            rows.append(
                {
                    "#": index + 1,
                    "concept": f"{item.get('source')} -> {item.get('target')}",
                    "t": "shared" if occurrence is None else f"t={occurrence}",
                    "set": item.get(
                        "setting",
                        format_optional(item.get("edge_value", item.get("edge_scale"))),
                    ),
                    "target": f"{scope} / {str(item.get('edge_kind', 'edge')).replace('_', ' ')}",
                    "p base": "",
                    "p alone": "",
                    "delta alone": "",
                }
            )
            continue
        if item_type == "activity":
            target = str(item.get("target", "forecast"))
            horizon = item.get("horizon_idx")
            horizon_int = None if horizon is None or pd.isna(horizon) else int(horizon)
            class_idx = int(item["class_idx"])
            single_outputs = forward_outputs(
                workspace,
                instance,
                intervention={"items": [item]},
            )
            alone = selected_class_probability(
                workspace,
                single_outputs,
                target,
                class_idx,
                horizon_int,
            )
            before = item.get("before_probability")
            rows.append(
                {
                    "#": index + 1,
                    "concept": f"activity: {item.get('activity', class_idx)}",
                    "t": item.get("horizon", f"step {item.get('step')}"),
                    "set": item.get("setting", f"p={float(item['probability']):.2f}"),
                    "target": "activity belief",
                    "p base": format_probability(before),
                    "p alone": format_probability(alone),
                    "delta alone": format_probability_delta(before, alone),
                }
            )
            continue
        target = str(item.get("target", "forecast"))
        horizon = item.get("horizon_idx")
        horizon_int = None if horizon is None or pd.isna(horizon) else int(horizon)
        class_idx = int(item.get("class_idx", -1))
        single_outputs = forward_outputs(
            workspace,
            instance,
            intervention={"items": [item]},
        )
        alone = (
            selected_class_probability(workspace, single_outputs, target, class_idx, horizon_int)
            if class_idx >= 0
            else None
        )
        before = item.get("before_probability")
        rows.append(
            {
                "#": index + 1,
                "concept": item.get("concept", item.get("concept_idx")),
                "t": (
                    f"t+{int(item['rollout_step'])}"
                    if item.get("rollout_step") is not None
                    else graph_time_option_label(instance, int(item["time_idx"]))
                ),
                "set": item.get("setting", format_optional(item.get("value"))),
                "target": f"{item.get('horizon', '?')} -> {item.get('target_label', label_name(workspace, class_idx) or class_idx)}",
                "p base": format_probability(before),
                "p alone": format_probability(alone),
                "delta alone": format_probability_delta(before, alone),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if intervention is not None:
        cols = st.columns([0.55, 0.45])
        remove_label = cols[0].selectbox(
            "Remove item",
            [str(row["#"]) for row in rows],
            format_func=lambda value: f"{value}: {rows[int(value) - 1]['concept']}",
        )
        cols[0].button(
            "Remove selected",
            width="stretch",
            on_click=remove_active_intervention_item,
            args=(int(remove_label) - 1,),
        )
        cols[1].button("Reset all", on_click=clear_active_intervention, width="stretch")


def combined_intervention_target_rows(
    workspace,
    baseline_outputs: Mapping[str, object],
    outputs: Mapping[str, object],
    items: list[Mapping[str, object]],
) -> list[dict[str, str]]:
    seen: set[tuple[str, int | None, int]] = set()
    rows = []
    for item in items:
        target = str(item.get("target", "forecast"))
        horizon = item.get("horizon_idx")
        horizon_int = None if horizon is None or pd.isna(horizon) else int(horizon)
        class_idx = int(item.get("class_idx", -1))
        if class_idx < 0:
            continue
        key = (target, horizon_int, class_idx)
        if key in seen:
            continue
        seen.add(key)
        before = selected_class_probability(workspace, baseline_outputs, target, class_idx, horizon_int)
        after = selected_class_probability(workspace, outputs, target, class_idx, horizon_int)
        rows.append(
            {
                "target": f"{item.get('horizon', '?')} -> {item.get('target_label', label_name(workspace, class_idx) or class_idx)}",
                "p base": format_probability(before),
                "p combined": format_probability(after),
                "delta combined": format_probability_delta(before, after),
            }
        )
    return rows


def prediction_probabilities_at_step(
    outputs: Mapping[str, object],
    step: int,
) -> np.ndarray | None:
    """Return the model's local prediction, excluding any forced feedback belief."""

    step = int(step)
    if step == 0:
        logits = outputs.get("activity_logits")
    else:
        logits = None
        for key in ("autoregressive_logits_by_step", "forecast_logits_by_horizon"):
            by_step = outputs.get(key)
            if isinstance(by_step, Mapping):
                logits = by_step.get(step, by_step.get(str(step)))
                if torch.is_tensor(logits):
                    break
    if not torch.is_tensor(logits):
        return None
    selected = logits[0, -1, :] if logits.ndim == 3 else logits[0, :]
    return torch.softmax(selected, dim=-1).detach().cpu().numpy()


def rollout_prediction_change_rows(
    workspace,
    baseline_outputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_horizon = min(
        ACTIVITY_FEEDBACK_ROLLOUT_HORIZON,
        max(int(value) for value in workspace.forecast_horizons),
    )
    for step in range(max_horizon + 1):
        before = prediction_probabilities_at_step(baseline_outputs, step)
        after = prediction_probabilities_at_step(outputs, step)
        if before is None or after is None:
            continue
        before_idx = int(before.argmax())
        after_idx = int(after.argmax())
        delta = after - before
        largest_idx = int(np.abs(delta).argmax())
        rows.append(
            {
                "time": "t" if step == 0 else f"t+{step}",
                "prediction before": label_name(workspace, before_idx) or str(before_idx),
                "p before": float(before[before_idx]),
                "prediction after": label_name(workspace, after_idx) or str(after_idx),
                "p after": float(after[after_idx]),
                "flipped": "yes" if before_idx != after_idx else "no",
                "total variation": float(0.5 * np.abs(delta).sum()),
                "largest local change": (
                    f"{label_name(workspace, largest_idx) or largest_idx} "
                    f"({float(delta[largest_idx]):+.3f})"
                ),
            }
        )
    return rows


def local_prediction_change_rows(
    workspace,
    baseline_outputs: Mapping[str, object],
    outputs: Mapping[str, object],
    top_k: int = 3,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_horizon = min(
        ACTIVITY_FEEDBACK_ROLLOUT_HORIZON,
        max(int(value) for value in workspace.forecast_horizons),
    )
    for step in range(max_horizon + 1):
        before = prediction_probabilities_at_step(baseline_outputs, step)
        after = prediction_probabilities_at_step(outputs, step)
        if before is None or after is None:
            continue
        delta = after - before
        for index in np.argsort(-np.abs(delta))[: min(int(top_k), delta.size)]:
            rows.append(
                {
                    "time": "t" if step == 0 else f"t+{step}",
                    "activity": label_name(workspace, int(index)) or str(int(index)),
                    "p before": float(before[index]),
                    "p after": float(after[index]),
                    "delta": float(delta[index]),
                }
            )
    return rows


def render_rollout_prediction_changes(
    workspace,
    baseline_outputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> None:
    rows = rollout_prediction_change_rows(workspace, baseline_outputs, outputs)
    if not rows:
        return
    st.markdown("##### Forecast response across the autoregressive rollout")
    st.caption(
        "These are the model's local predictions before and after intervention at every "
        "rollout step. A forced label belief is an input to the next step, so an intervention "
        "at t+n should first change predictions after t+n."
    )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    local_rows = local_prediction_change_rows(
        workspace,
        baseline_outputs,
        outputs,
    )
    if local_rows:
        with st.expander("Largest local class-probability changes", expanded=True):
            st.dataframe(pd.DataFrame(local_rows), width="stretch", hide_index=True)


def active_intervention_items_for_instance(instance: Mapping[str, Any]) -> list[dict[str, object]]:
    intervention = st.session_state.get(ACTIVE_INTERVENTION_KEY)
    if not isinstance(intervention, dict):
        return []
    expected = (str(instance["split"]), int(instance["video_index"]), int(instance["timestep"]))
    found = (
        str(intervention.get("split")),
        int(intervention.get("video_index", -1)),
        int(intervention.get("timestep", -1)),
    )
    if found != expected:
        return []
    return [dict(item) for item in intervention.get("items", []) if isinstance(item, dict)]


def prediction_rows_for_instance(
    workspace,
    instance: Mapping[str, Any],
    outputs: Mapping[str, object],
) -> list[dict[str, object]]:
    current_label = label_name(workspace, instance.get("current_label"))
    current_rows, forecast_rows = prediction_tables(workspace, instance, outputs, top_k=1)
    prediction_rows: list[dict[str, object]] = []
    if current_rows:
        row = current_rows[0]
        prediction_rows.append(
            {
                "row_id": "activity:t",
                "horizon": "t",
                "target": "activity",
                "horizon_idx": None,
                "class_idx": int(row["class_idx"]),
                "prediction": row["class"],
                "probability": f"{float(row['probability']):.3f}",
                "ground_truth": current_label or "n/a",
            }
        )
    for horizon, rows in sorted(forecast_rows.items()):
        if not rows:
            continue
        row = rows[0]
        prediction_rows.append(
            {
                "row_id": f"forecast:{int(horizon)}",
                "horizon": f"t+{int(horizon)}",
                "target": "forecast",
                "horizon_idx": int(horizon),
                "class_idx": int(row["class_idx"]),
                "prediction": row["class"],
                "probability": f"{float(row['probability']):.3f}",
                "ground_truth": row.get("true") or "n/a",
            }
        )
    return prediction_rows


def ensure_selected_prediction(prediction_rows: list[dict[str, object]]) -> None:
    valid = {str(row["row_id"]) for row in prediction_rows}
    selected = str(st.session_state.get(SELECTED_PREDICTION_ROW_KEY, ""))
    if selected not in valid:
        st.session_state[SELECTED_PREDICTION_ROW_KEY] = str(prediction_rows[0]["row_id"]) if prediction_rows else ""


def selected_prediction_from_state(prediction_rows: list[dict[str, object]]) -> dict[str, object] | None:
    selected = str(st.session_state.get(SELECTED_PREDICTION_ROW_KEY, ""))
    for row in prediction_rows:
        if str(row["row_id"]) == selected:
            return row
    return prediction_rows[0] if prediction_rows else None


def select_prediction_row(row_id: str) -> None:
    st.session_state[SELECTED_PREDICTION_ROW_KEY] = str(row_id)


def render_target_controls(
    workspace,
    instance: Mapping[str, Any],
    selected_row: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if selected_row is None:
        return None

    horizon_options = target_horizon_options(workspace, instance)
    if not horizon_options:
        return dict(selected_row)
    selected_horizon = str(selected_row.get("row_id", horizon_options[0]["row_id"]))
    valid_horizons = [str(row["row_id"]) for row in horizon_options]
    if st.session_state.get(TARGET_HORIZON_KEY) not in valid_horizons:
        st.session_state[TARGET_HORIZON_KEY] = selected_horizon if selected_horizon in valid_horizons else valid_horizons[0]

    st.markdown("#### Intervention target")
    cols = st.columns([0.28, 0.72])
    target_row_id = cols[0].selectbox(
        "Horizon",
        valid_horizons,
        format_func=lambda row_id: target_horizon_label(horizon_options, row_id),
        key=TARGET_HORIZON_KEY,
    )
    target_base = next((dict(row) for row in horizon_options if str(row["row_id"]) == str(target_row_id)), dict(selected_row))
    ground_truth_class_idx = target_base.get("class_idx")
    if ground_truth_class_idx is None:
        cols[1].selectbox("Activity label", ["unavailable"], disabled=True)
        st.warning("No ground-truth label is available at the selected horizon.")
        return None
    ground_truth_class_idx = int(ground_truth_class_idx)
    label_options = [str(name) for name in workspace.activity_names]
    ground_truth_label = label_name(workspace, ground_truth_class_idx) or str(ground_truth_class_idx)
    context = (
        str(instance.get("split")),
        int(instance.get("video_index", -1)),
        int(instance.get("timestep", -1)),
        str(target_row_id),
        ground_truth_class_idx,
    )
    if st.session_state.get(TARGET_LABEL_CONTEXT_KEY) != context:
        st.session_state[TARGET_LABEL_KEY] = ground_truth_label
        st.session_state[TARGET_LABEL_CONTEXT_KEY] = context
    if st.session_state.get(TARGET_LABEL_KEY) not in label_options:
        st.session_state[TARGET_LABEL_KEY] = ground_truth_label
    label = cols[1].selectbox(
        "Activity label",
        label_options,
        key=TARGET_LABEL_KEY,
        help="Defaults to the ground-truth label for this horizon; change it to inspect or intervene on another label.",
    )
    class_idx = int(label_options.index(str(label)))
    target_base.update(
        {
            "class_idx": class_idx,
            "prediction": str(label),
            "target_label": str(label),
            "probability": None,
            "ground_truth": target_base.get("ground_truth", "n/a"),
        }
    )
    return target_base


def target_horizon_options(workspace, instance: Mapping[str, Any]) -> list[dict[str, object]]:
    options: list[dict[str, object]] = [
        {
            "row_id": "activity:t",
            "horizon": "t",
            "target": "activity",
            "horizon_idx": None,
            "class_idx": instance.get("current_label"),
        }
    ]
    future_labels = instance.get("future_labels", {})
    for horizon in workspace.forecast_horizons:
        options.append(
            {
                "row_id": f"forecast:{int(horizon)}",
                "horizon": f"t+{int(horizon)}",
                "target": "forecast",
                "horizon_idx": int(horizon),
                "class_idx": future_labels.get(int(horizon)) if isinstance(future_labels, Mapping) else None,
            }
        )
    return options


def target_horizon_label(options: list[dict[str, object]], row_id: str) -> str:
    for row in options:
        if str(row["row_id"]) == str(row_id):
            return str(row["horizon"])
    return str(row_id)


def render_prediction_pane(
    workspace,
    instance: Mapping[str, Any],
    prediction_rows: list[dict[str, object]],
) -> dict[str, object] | None:
    st.markdown("#### Current label")
    current_label = label_name(workspace, instance.get("current_label"))
    st.metric("Current window", current_label or "unknown")

    st.markdown("#### Prediction")
    if not prediction_rows:
        st.info("No predictions available for this timestep.")
        return None

    prediction_df = pd.DataFrame(prediction_rows)
    st.dataframe(
        prediction_df,
        width="stretch",
        hide_index=True,
        column_order=["horizon", "prediction", "probability", "ground_truth"],
    )
    row_ids = [str(row["row_id"]) for row in prediction_rows]
    selected = selected_prediction_from_state(prediction_rows)
    selected_id = str(selected["row_id"]) if selected is not None else row_ids[0]
    st.selectbox(
        "Explain row",
        row_ids,
        index=row_ids.index(selected_id) if selected_id in row_ids else 0,
        format_func=lambda row_id: prediction_label(prediction_rows, row_id),
        key=SELECTED_PREDICTION_ROW_KEY,
    )
    return selected_prediction_from_state(prediction_rows)


def render_forecast_timeline(
    prediction_rows: list[dict[str, object]],
    selected_row: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if not prediction_rows:
        return None
    st.markdown("#### Forecast timeline")
    selected_id = str(selected_row["row_id"]) if selected_row is not None else str(prediction_rows[0]["row_id"])
    cols = st.columns(len(prediction_rows))
    for col, row in zip(cols, prediction_rows):
        row_id = str(row["row_id"])
        label = str(row["horizon"])
        button_label = f"{label}  {row['probability']}"
        col.button(
            button_label,
            key=f"timeline_{row_id}",
            width="stretch",
            type="primary" if row_id == selected_id else "secondary",
            on_click=select_prediction_row,
            args=(row_id,),
        )
        col.caption(
            f"pred: {row['prediction']}\n\n"
            f"true: {row['ground_truth']}"
        )
    return selected_prediction_from_state(prediction_rows)


def prediction_label(prediction_rows: list[dict[str, object]], row_id: str) -> str:
    for row in prediction_rows:
        if str(row["row_id"]) == str(row_id):
            return f"{row['horizon']} -> {row['prediction']}"
    return str(row_id)


def selected_prediction_row(
    prediction_rows: list[dict[str, object]],
    selection: object,
) -> dict[str, object]:
    rows = getattr(getattr(selection, "selection", None), "rows", None)
    if rows:
        index = int(rows[0])
        if 0 <= index < len(prediction_rows):
            return prediction_rows[index]
    return prediction_rows[0]


def render_selected_prediction_panel(
    workspace,
    instance: Mapping[str, Any],
    baseline_outputs: Mapping[str, object],
    outputs: Mapping[str, object],
    selected_row: Mapping[str, object],
    intervention: Mapping[str, object] | None,
) -> None:
    target = str(selected_row["target"])
    horizon = selected_row.get("horizon_idx")
    horizon_int = None if horizon is None or pd.isna(horizon) else int(horizon)
    class_idx = int(selected_row["class_idx"])
    st.markdown("#### Selected target")
    st.caption(f"{selected_row['horizon']} -> {selected_row['prediction']}")
    render_prediction_options(workspace, outputs, target, horizon_int, class_idx)
    render_activity_feedback_controls(
        workspace,
        instance,
        baseline_outputs,
        outputs,
        selected_row,
    )

    valid_history = valid_history_time_indices(instance)
    current_time_idx = max(valid_history) if valid_history else None
    importance_top_k = min(10, len(workspace.concept_names))
    rows = prediction_concept_contributors(
        workspace,
        instance,
        outputs,
        target=target,
        class_idx=class_idx,
        horizon=horizon_int,
        top_k=importance_top_k,
        history_time_idx=current_time_idx,
    )
    if not rows:
        st.info("No concept-level classifier explanation is available for this checkpoint.")
        return

    predicted_options = prediction_option_rows(workspace, outputs, target, horizon=horizon_int, top_k=1)
    predicted_class_idx = int(predicted_options[0]["class_idx"]) if predicted_options else class_idx
    predicted_rows = rows
    if predicted_class_idx != class_idx:
        predicted_rows = prediction_concept_contributors(
            workspace,
            instance,
            outputs,
            target=target,
            class_idx=predicted_class_idx,
            horizon=horizon_int,
            top_k=importance_top_k,
            history_time_idx=current_time_idx,
        )

    selected_activity_label = label_name(workspace, class_idx) or str(class_idx)
    selected_window_label = str(selected_row["horizon"])
    st.markdown(f"##### Top {importance_top_k} linear-head concept contributions")
    st.caption(
        f"Activity: {selected_activity_label} · window: {selected_window_label}. "
        "Signed contribution = transformed concept activation × final linear-head weight; "
        "concepts are ranked by absolute contribution and the head bias is excluded."
    )
    if predicted_class_idx != class_idx and predicted_rows:
        predicted_label = label_name(workspace, predicted_class_idx) or str(predicted_class_idx)
        predicted_col, truth_col = st.columns(2)
        with predicted_col:
            st.markdown(f"**Predicted: {predicted_label}**")
            render_importance_chart(predicted_rows)
        with truth_col:
            st.markdown(f"**Selected: {selected_activity_label}**")
            render_importance_chart(rows)
    else:
        render_importance_chart(rows)
    if intervention is not None:
        render_intervention_prediction_delta(
            workspace,
            baseline_outputs,
            outputs,
            target,
            class_idx,
            horizon_int,
        )

    baseline_rows = rows
    if intervention is not None:
        baseline_rows = prediction_concept_contributors(
            workspace,
            instance,
            baseline_outputs,
            target=target,
            class_idx=class_idx,
            horizon=horizon_int,
            top_k=importance_top_k,
            history_time_idx=current_time_idx,
        )

    graph_intervention_items = active_intervention_items_for_instance(instance)
    render_scene_graph_overview(
        workspace,
        instance,
        baseline_outputs,
        outputs,
        baseline_rows,
        rows,
        graph_intervention_items,
        target,
    )

    if not has_learned_threshold_calibrator(workspace):
        st.info("Binary true/false intervention needs a learned-threshold calibrator.")
        return

    intervention_mode = st.selectbox(
        "Intervention mode",
        ["input", "persistent"],
        key=INTERVENTION_MODE_KEY,
        format_func=lambda value: {
            "input": "Input evidence (one-time)",
            "persistent": "Persistent node clamp",
        }[str(value)],
        help=(
            "Input changes the raw standardized concept evidence before calibration. "
            "Persistent starts with that same edit and then re-clamps the calibrated node "
            "after every graph refinement and rollout step."
        ),
    )
    if intervention_mode == "input":
        st.caption(
            "Input evidence is injected once before calibration. The graph may attenuate, "
            "amplify, or overwrite that evidence during refinement."
        )
    else:
        st.caption(
            "Persistent clamp keeps the chosen node fixed after every graph/forecast update; "
            "other nodes may still respond. This is the stronger model-level do-style test, "
            "not a claim of real-world causality."
        )
    quick_time_idx = int(
        st.selectbox(
            "Counterfactual intervention node time",
            valid_history,
            index=len(valid_history) - 1,
            format_func=lambda value: graph_time_option_label(instance, int(value)),
            key="counterfactual_node_intervention_time",
            help=(
                "Each concept is tested alone at this node using both state=0.1 and state=0.9. "
                "The ranking uses the actual model probability response."
            ),
        )
    )
    counterfactual_rows = cached_counterfactual_concept_rows(
        workspace,
        instance,
        baseline_outputs,
        target,
        horizon_int,
        class_idx,
        quick_time_idx,
        intervention_mode,
        top_k=importance_top_k,
    )
    render_counterfactual_importance_chart(
        counterfactual_rows,
        selected_activity_label,
        graph_time_option_label(instance, quick_time_idx).split(" (frame", 1)[0],
    )
    with st.expander("Quick interventions (optional)", expanded=False):
        quick_rows = counterfactual_rows[:5]
        st.caption("Shortcuts for the five concepts with the largest tested probability effect")
        for row in quick_rows:
            intervention_row = {
                **dict(row),
                "history_t": quick_time_idx,
                "original_t": scene_graph_original_timestep(instance, quick_time_idx),
            }
            cols = st.columns([0.56, 0.14, 0.14, 0.16])
            relative_time = graph_time_option_label(instance, quick_time_idx).split(" (frame", 1)[0]
            cols[0].markdown(
                f"`{relative_time}` {row['concept']}  \n"
                f"best tested effect: {float(row['recommended_delta']):+.3f}"
            )
            true_value = float(row["high_value"])
            false_value = float(row["low_value"])
            if cols[1].button("Set true", key=f"set_true_{row['concept_idx']}"):
                set_active_intervention(workspace, instance, intervention_row, true_value, "true", selected_row, baseline_outputs)
                st.rerun()
            if cols[2].button("Set false", key=f"set_false_{row['concept_idx']}"):
                set_active_intervention(workspace, instance, intervention_row, false_value, "false", selected_row, baseline_outputs)
                st.rerun()
            active_here = intervention_contains(intervention, intervention_row)
            if cols[3].button("Reset", key=f"clear_{row['concept_idx']}", disabled=not active_here):
                clear_active_intervention(intervention_row)
                st.rerun()

    render_manual_node_intervention_controls(
        workspace,
        instance,
        selected_row,
        baseline_outputs,
    )

    if intervention is not None:
        st.button("Reset all interventions", on_click=clear_active_intervention, width="stretch")


def render_manual_node_intervention_controls(
    workspace,
    instance: Mapping[str, Any],
    selected_row: Mapping[str, object],
    baseline_outputs: Mapping[str, object],
) -> None:
    """Allow a concept-time intervention outside the five automatic contributors."""

    history_time_options = valid_history_time_indices(instance)
    if not history_time_options or not workspace.concept_names:
        return
    max_horizon = max(int(value) for value in workspace.forecast_horizons)
    time_options = [f"history:{time_idx}" for time_idx in history_time_options]
    time_options.extend(f"rollout:{step}" for step in range(1, max_horizon + 1))

    def time_option_label(value: str) -> str:
        kind, raw_time = str(value).split(":", 1)
        if kind == "rollout":
            return f"t+{int(raw_time)} (forecast state)"
        return graph_time_option_label(instance, int(raw_time))

    with st.expander("Intervene on any concept node", expanded=False):
        st.caption(
            "Choose an observed node from t-k through t, or a future autoregressive state at t+n. "
            "Future edits are applied directly to that rollout concept state."
        )
        cols = st.columns([0.28, 0.48, 0.24])
        selected_time = str(
            cols[0].selectbox(
                "Node time",
                time_options,
                index=len(history_time_options) - 1,
                format_func=time_option_label,
                key="manual_node_intervention_time",
            )
        )
        concept_idx = int(
            cols[1].selectbox(
                "Concept label",
                list(range(len(workspace.concept_names))),
                format_func=lambda value: graph_concept_label(workspace, int(value)),
                key="manual_node_intervention_concept",
            )
        )
        state = str(
            cols[2].selectbox(
                "State",
                ["true", "false"],
                format_func=lambda value: "True (0.9)" if value == "true" else "False (0.1)",
                key="manual_node_intervention_state",
            )
        )
        target_probability = 0.9 if state == "true" else 0.1
        raw_value = learned_threshold_intervention_value(workspace, concept_idx, target_probability)
        if st.button("Add concept-node intervention", width="stretch", key="manual_node_intervention_add"):
            time_kind, raw_time = selected_time.split(":", 1)
            if time_kind == "rollout":
                row = {
                    "rollout_step": int(raw_time),
                    "concept_idx": concept_idx,
                    "concept": graph_concept_label(workspace, concept_idx),
                }
                set_active_intervention(
                    workspace,
                    instance,
                    row,
                    target_probability,
                    state,
                    selected_row,
                    baseline_outputs,
                )
                st.rerun()
            elif raw_value is not None:
                history_t = int(raw_time)
                row = {
                    "history_t": history_t,
                    "original_t": scene_graph_original_timestep(instance, history_t),
                    "concept_idx": concept_idx,
                    "concept": graph_concept_label(workspace, concept_idx),
                }
                set_active_intervention(
                    workspace,
                    instance,
                    row,
                    raw_value,
                    state,
                    selected_row,
                    baseline_outputs,
                )
                st.rerun()


def render_activity_feedback_controls(
    workspace,
    instance: Mapping[str, Any],
    baseline_outputs: Mapping[str, object],
    outputs: Mapping[str, object],
    selected_row: Mapping[str, object],
) -> None:
    model = workspace.model
    enabled_fn = getattr(model, "_activity_feedback_enabled", None)
    if not callable(enabled_fn) or not bool(enabled_fn()):
        return
    max_horizon = max(int(value) for value in workspace.forecast_horizons)
    rollout_horizon = min(ACTIVITY_FEEDBACK_ROLLOUT_HORIZON, max_horizon)

    st.markdown("##### Label-belief intervention")
    history_steps_fn = getattr(model, "_activity_feedback_history_steps", None)
    history_steps = int(history_steps_fn()) if callable(history_steps_fn) else 0
    if history_steps > 0:
        render_historical_activity_feedback_controls(
            workspace,
            instance,
            baseline_outputs,
            outputs,
            selected_row,
            history_steps,
        )
    with st.expander("Intervene at t or a future rollout step", expanded=True):
        st.caption(
            "Choose the label belief that is fed into the next autoregressive step. "
            f"This view rolls out through t+{rollout_horizon}. Valid source times are t through "
            f"t+{rollout_horizon - 1}; the final t+{rollout_horizon} "
            "belief has no later prediction inside this rollout."
        )
        cols = st.columns([0.24, 0.39, 0.22, 0.15])
        step = int(
            cols[0].selectbox(
                "Source time",
                list(range(rollout_horizon)),
                format_func=lambda value: "t" if int(value) == 0 else f"t+{int(value)}",
                key="activity_feedback_source_step",
            )
        )
        ground_truth_class_idx = activity_ground_truth_class_idx(instance, step)
        label_options = [str(name) for name in workspace.activity_names]
        if ground_truth_class_idx is None:
            cols[1].selectbox("Activity label", ["ground truth unavailable"], disabled=True)
            st.warning(f"No ground-truth activity label is available at {'t' if step == 0 else f't+{step}'}.")
            return
        ground_truth_label = label_name(workspace, ground_truth_class_idx) or str(ground_truth_class_idx)
        label_context = (
            str(instance.get("split")),
            int(instance.get("video_index", -1)),
            int(instance.get("timestep", -1)),
            step,
            ground_truth_class_idx,
        )
        if st.session_state.get(ACTIVITY_FEEDBACK_LABEL_CONTEXT_KEY) != label_context:
            st.session_state[ACTIVITY_FEEDBACK_LABEL_KEY] = ground_truth_label
            st.session_state[ACTIVITY_FEEDBACK_LABEL_CONTEXT_KEY] = label_context
        if st.session_state.get(ACTIVITY_FEEDBACK_LABEL_KEY) not in label_options:
            st.session_state[ACTIVITY_FEEDBACK_LABEL_KEY] = ground_truth_label
        class_label = str(
            cols[1].selectbox(
                "Activity label",
                label_options,
                key=ACTIVITY_FEEDBACK_LABEL_KEY,
                help=(
                    "Defaults to the ground-truth activity at the selected source time. "
                    "Choose another label to override it."
                ),
            )
        )
        class_idx = int(label_options.index(class_label))
        probability = float(
            cols[2].slider(
                "Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
                key="activity_feedback_source_probability",
            )
        )
        if cols[3].button("Apply", key="activity_feedback_source_apply", width="stretch"):
            set_active_activity_intervention(
                workspace,
                instance,
                step,
                class_idx,
                probability,
                baseline_outputs,
            )
            st.rerun()

        active_steps = sorted(
            {
                int(item.get("step", 0))
                for item in active_intervention_items_for_instance(instance)
                if str(item.get("item_type", item.get("type", "concept"))).lower() == "activity"
                and int(item.get("step", 0)) >= 0
            }
        )
        for active_step in active_steps:
            render_activity_feedback_effects(
                workspace,
                baseline_outputs,
                outputs,
                active_step,
            )


def activity_ground_truth_class_idx(
    instance: Mapping[str, Any],
    step: int,
) -> int | None:
    if int(step) == 0:
        value = instance.get("current_label")
    else:
        future_labels = instance.get("future_labels", {})
        if not isinstance(future_labels, Mapping):
            return None
        value = future_labels.get(int(step), future_labels.get(str(int(step))))
    return None if value is None else int(value)


def render_historical_activity_feedback_controls(
    workspace,
    instance: Mapping[str, Any],
    baseline_outputs: Mapping[str, object],
    outputs: Mapping[str, object],
    selected_row: Mapping[str, object],
    history_steps: int,
) -> None:
    with st.expander("Past activity beliefs → current t", expanded=False):
        st.caption(
            "Force a categorical belief at an observed past step. Its learned lag-gated "
            "label→concept message is injected into the current concept state before classifying t."
        )
        cols = st.columns([0.24, 0.39, 0.22, 0.15])
        history_step = int(
            cols[0].selectbox(
                "Belief time",
                [-lag for lag in range(1, int(history_steps) + 1)],
                format_func=lambda value: f"t{int(value):+d}",
                key="history_activity_feedback_step",
            )
        )
        class_idx = int(selected_row.get("class_idx", 0))
        class_label = label_name(workspace, class_idx) or str(class_idx)
        cols[1].metric("Selected activity label", class_label)
        probability = float(
            cols[2].slider(
                "Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
                key="history_activity_feedback_probability",
            )
        )
        if cols[3].button("Apply", key="history_activity_feedback_apply", width="stretch"):
            set_active_activity_intervention(
                workspace,
                instance,
                history_step,
                class_idx,
                probability,
                baseline_outputs,
            )
            st.rerun()

        active_steps = sorted(
            {
                int(item.get("step", 0))
                for item in active_intervention_items_for_instance(instance)
                if str(item.get("item_type", item.get("type", "concept"))).lower() == "activity"
                and int(item.get("step", 0)) < 0
            }
        )
        if active_steps:
            st.caption("Current-t effects from active historical belief interventions")
            for active_step in active_steps:
                render_activity_feedback_effects(
                    workspace,
                    baseline_outputs,
                    outputs,
                    active_step,
                )


def render_activity_feedback_effects(
    workspace,
    baseline_outputs: Mapping[str, object],
    outputs: Mapping[str, object],
    step: int,
) -> None:
    if int(step) < 0:
        render_historical_activity_feedback_effects(
            workspace,
            baseline_outputs,
            outputs,
            int(step),
        )
        return
    before_by_step = baseline_outputs.get("effective_activity_probs_by_step")
    after_by_step = outputs.get("effective_activity_probs_by_step")
    if not isinstance(before_by_step, Mapping) or not isinstance(after_by_step, Mapping):
        return
    before = before_by_step.get(int(step))
    after = after_by_step.get(int(step))
    if not torch.is_tensor(before) or not torch.is_tensor(after):
        return
    before_np = before[0, -1, :].detach().cpu().numpy()
    after_np = after[0, -1, :].detach().cpu().numpy()
    label_delta = after_np - before_np
    top_labels = np.argsort(-np.abs(label_delta))[: min(6, len(workspace.activity_names))]
    st.caption("Forced versus original categorical belief")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "activity": label_name(workspace, int(index)) or int(index),
                    "original p": float(before_np[index]),
                    "effective p": float(after_np[index]),
                    "delta": float(label_delta[index]),
                }
                for index in top_labels
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    matrix_fn = getattr(workspace.model, "effective_activity_feedback_matrix", None)
    if not callable(matrix_fn):
        return
    matrix = matrix_fn().detach().cpu().numpy()
    message_delta = label_delta @ matrix
    top_concepts = np.argsort(-np.abs(message_delta))[: min(10, len(workspace.concept_names))]
    st.caption("Top signed label-to-concept message changes")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "concept": workspace.concept_names[int(index)],
                    "message delta": float(message_delta[index]),
                    "direction": "increase" if float(message_delta[index]) >= 0.0 else "decrease",
                }
                for index in top_concepts
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    before_concepts = baseline_outputs.get("predicted_concepts_by_step")
    after_concepts = outputs.get("predicted_concepts_by_step")
    if not isinstance(before_concepts, Mapping) or not isinstance(after_concepts, Mapping):
        return
    future_rows = []
    common_horizons = set(int(value) for value in before_concepts) & set(
        int(value) for value in after_concepts
    )
    for horizon in sorted(common_horizons):
        if horizon <= int(step) or horizon > ACTIVITY_FEEDBACK_ROLLOUT_HORIZON:
            continue
        delta = (
            after_concepts[horizon][0, -1, :] - before_concepts[horizon][0, -1, :]
        ).detach().cpu().numpy()
        for index in np.argsort(-np.abs(delta))[:3]:
            future_rows.append(
                {
                    "horizon": f"t+{horizon}",
                    "concept": workspace.concept_names[int(index)],
                    "state delta": float(delta[index]),
                }
            )
    if future_rows:
        st.caption("Largest propagated future-concept changes")
        st.dataframe(pd.DataFrame(future_rows), width="stretch", hide_index=True)


def render_historical_activity_feedback_effects(
    workspace,
    baseline_outputs: Mapping[str, object],
    outputs: Mapping[str, object],
    step: int,
) -> None:
    before_by_step = baseline_outputs.get("effective_activity_probs_by_history_step")
    after_by_step = outputs.get("effective_activity_probs_by_history_step")
    if not isinstance(before_by_step, Mapping) or not isinstance(after_by_step, Mapping):
        return
    before = before_by_step.get(int(step))
    after = after_by_step.get(int(step))
    if not torch.is_tensor(before) or not torch.is_tensor(after):
        return
    before_np = before[0].detach().cpu().numpy()
    after_np = after[0].detach().cpu().numpy()
    label_delta = after_np - before_np
    top_labels = np.argsort(-np.abs(label_delta))[: min(6, len(workspace.activity_names))]
    st.markdown(f"**t{int(step):+d} intervention**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "activity": label_name(workspace, int(index)) or int(index),
                    "original p": float(before_np[index]),
                    "effective p": float(after_np[index]),
                    "delta": float(label_delta[index]),
                }
                for index in top_labels
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    before_messages = baseline_outputs.get("activity_feedback_messages_by_history_step")
    after_messages = outputs.get("activity_feedback_messages_by_history_step")
    if isinstance(before_messages, Mapping) and isinstance(after_messages, Mapping):
        before_message = before_messages.get(int(step))
        after_message = after_messages.get(int(step))
        if torch.is_tensor(before_message) and torch.is_tensor(after_message):
            message_delta = (after_message[0] - before_message[0]).detach().cpu().numpy()
            top_message = np.argsort(-np.abs(message_delta))[: min(8, len(workspace.concept_names))]
            st.caption("Largest signed lag-gated label→concept message changes")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "concept": workspace.concept_names[int(index)],
                            "message delta": float(message_delta[index]),
                            "direction": "increase" if float(message_delta[index]) >= 0.0 else "decrease",
                        }
                        for index in top_message
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    before_current = baseline_outputs.get("post_history_feedback_current_concepts")
    after_current = outputs.get("post_history_feedback_current_concepts")
    if torch.is_tensor(before_current) and torch.is_tensor(after_current):
        concept_delta = (after_current[0] - before_current[0]).detach().cpu().numpy()
        top_concepts = np.argsort(-np.abs(concept_delta))[: min(8, len(workspace.concept_names))]
        st.caption("Largest current concept-state changes at t")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "concept": workspace.concept_names[int(index)],
                        "state delta at t": float(concept_delta[index]),
                    }
                    for index in top_concepts
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    before_logits = baseline_outputs.get("activity_logits")
    after_logits = outputs.get("activity_logits")
    if torch.is_tensor(before_logits) and torch.is_tensor(after_logits):
        before_probs = torch.softmax(before_logits[0, -1, :], dim=-1).detach().cpu().numpy()
        after_probs = torch.softmax(after_logits[0, -1, :], dim=-1).detach().cpu().numpy()
        current_delta = after_probs - before_probs
        top_current = np.argsort(-np.abs(current_delta))[: min(8, len(workspace.activity_names))]
        st.caption("Current activity probability changes at t")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "activity": label_name(workspace, int(index)) or int(index),
                        "p original": float(before_probs[index]),
                        "p intervened": float(after_probs[index]),
                        "delta": float(current_delta[index]),
                    }
                    for index in top_current
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def render_importance_chart(rows: list[Mapping[str, object]]) -> None:
    chart_rows = []
    for row in rows:
        contribution = float(row.get("contribution", 0.0))
        chart_rows.append(
            {
                "label": str(row.get("concept", "concept")),
                "concept": str(row.get("concept", "concept")),
                "contribution": contribution,
                "abs_contribution": abs(contribution),
                "activation": row.get("classifier_input", row.get("standardized_input")),
                "direction": "supports" if contribution >= 0.0 else "opposes",
            }
        )
    if not chart_rows:
        return

    frame = pd.DataFrame(chart_rows)
    max_abs = max(float(frame["abs_contribution"].max()), 1e-6)
    contribution_scale = alt.Scale(domain=[-1.08 * max_abs, 1.08 * max_abs])
    base = alt.Chart(frame).encode(
        x=alt.X("contribution:Q", title="Signed contribution", scale=contribution_scale),
        y=alt.Y(
            "label:N",
            title=None,
            sort=alt.SortField(field="abs_contribution", order="descending"),
            axis=alt.Axis(labelLimit=280),
        ),
        tooltip=[
            alt.Tooltip("concept:N", title="Concept"),
            alt.Tooltip("contribution:Q", title="Contribution", format="+.4f"),
            alt.Tooltip("activation:Q", title="Head input", format="+.4f"),
            alt.Tooltip("direction:N", title="Direction"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.condition(
            alt.datum.contribution >= 0,
            alt.value("#2f855a"),
            alt.value("#c2414b"),
        )
    )
    positive_labels = base.transform_filter(alt.datum.contribution >= 0).mark_text(
        align="left",
        baseline="middle",
        dx=5,
        color="#d1d5db",
    ).encode(text=alt.Text("contribution:Q", format="+.3f"))
    negative_labels = base.transform_filter(alt.datum.contribution < 0).mark_text(
        align="right",
        baseline="middle",
        dx=-5,
        color="#d1d5db",
    ).encode(text=alt.Text("contribution:Q", format="+.3f"))
    zero = alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule(color="#6b7280", strokeWidth=1).encode(
        x=alt.X("zero:Q", scale=contribution_scale)
    )
    chart = (zero + bars + positive_labels + negative_labels).properties(
        height=max(190, 38 * len(chart_rows))
    )
    st.altair_chart(chart, width="stretch")


def prediction_probability_matrix(
    workspace,
    outputs: Mapping[str, object],
    target: str,
    horizon: int | None,
) -> np.ndarray | None:
    step = 0 if str(target) == "activity" else int(horizon or workspace.horizon)
    by_step = outputs.get("effective_activity_probs_by_step")
    probabilities = (
        by_step.get(step, by_step.get(str(step)))
        if isinstance(by_step, Mapping)
        else None
    )
    if torch.is_tensor(probabilities):
        selected = probabilities[:, -1, :] if probabilities.ndim == 3 else probabilities
        return selected.detach().cpu().numpy()

    if str(target) == "activity":
        logits = outputs.get("activity_logits")
    else:
        logits = None
        for key in ("autoregressive_logits_by_step", "forecast_logits_by_horizon"):
            by_horizon = outputs.get(key)
            if isinstance(by_horizon, Mapping):
                logits = by_horizon.get(step, by_horizon.get(str(step)))
                if torch.is_tensor(logits):
                    break
    if not torch.is_tensor(logits):
        return None
    selected_logits = logits[:, -1, :] if logits.ndim == 3 else logits
    return torch.softmax(selected_logits, dim=-1).detach().cpu().numpy()


def counterfactual_concept_rows(
    workspace,
    instance: Mapping[str, Any],
    baseline_outputs: Mapping[str, object],
    target: str,
    horizon: int | None,
    class_idx: int,
    time_idx: int,
    intervention_mode: str,
    top_k: int = 10,
) -> list[dict[str, object]]:
    baseline_matrix = prediction_probability_matrix(
        workspace,
        baseline_outputs,
        target,
        horizon,
    )
    if baseline_matrix is None or baseline_matrix.shape[0] < 1:
        return []
    target_idx = int(class_idx)
    if target_idx < 0 or target_idx >= baseline_matrix.shape[1]:
        return []

    interventions: list[dict[str, object]] = []
    metadata: list[tuple[int, str, float, float]] = []
    for concept_idx, concept_name in enumerate(workspace.concept_names):
        for setting, state in (("false", 0.1), ("true", 0.9)):
            raw_value = learned_threshold_intervention_value(workspace, concept_idx, state)
            if raw_value is None:
                continue
            interventions.append(
                {
                    "mode": str(intervention_mode),
                    "items": [
                        {
                            "item_type": "concept",
                            "time_idx": int(time_idx),
                            "concept_idx": int(concept_idx),
                            "value": float(raw_value),
                        }
                    ],
                }
            )
            metadata.append((int(concept_idx), str(setting), float(state), float(raw_value)))
    if not interventions:
        return []

    counterfactual_outputs = forward_outputs_batched_interventions(
        workspace,
        instance,
        interventions,
    )
    probability_matrix = prediction_probability_matrix(
        workspace,
        counterfactual_outputs,
        target,
        horizon,
    )
    if probability_matrix is None or probability_matrix.shape[0] != len(metadata):
        return []

    baseline_probabilities = baseline_matrix[0]
    baseline_target = float(baseline_probabilities[target_idx])
    other_indices = [idx for idx in range(baseline_probabilities.size) if idx != target_idx]
    baseline_other = (
        float(np.max(baseline_probabilities[other_indices]))
        if other_indices
        else 0.0
    )
    baseline_margin = baseline_target - baseline_other
    by_concept: dict[int, dict[str, object]] = {}
    for batch_idx, (concept_idx, setting, state, raw_value) in enumerate(metadata):
        probabilities = probability_matrix[batch_idx]
        target_probability = float(probabilities[target_idx])
        strongest_other = (
            float(np.max(probabilities[other_indices]))
            if other_indices
            else 0.0
        )
        row = by_concept.setdefault(
            int(concept_idx),
            {
                "concept_idx": int(concept_idx),
                "concept": str(workspace.concept_names[int(concept_idx)]),
                "baseline_probability": baseline_target,
                "baseline_margin": baseline_margin,
            },
        )
        row[f"{setting}_state"] = state
        row[f"{setting}_value"] = raw_value
        row[f"{setting}_probability"] = target_probability
        row[f"{setting}_delta"] = target_probability - baseline_target
        row[f"{setting}_margin_delta"] = (
            target_probability - strongest_other - baseline_margin
        )

    rows: list[dict[str, object]] = []
    for row in by_concept.values():
        if "false_delta" not in row or "true_delta" not in row:
            continue
        recommended = (
            "true"
            if abs(float(row["true_delta"])) >= abs(float(row["false_delta"]))
            else "false"
        )
        row["recommended"] = recommended
        row["recommended_delta"] = float(row[f"{recommended}_delta"])
        row["recommended_margin_delta"] = float(row[f"{recommended}_margin_delta"])
        row["abs_effect"] = abs(float(row["recommended_delta"]))
        row["low_value"] = float(row["false_value"])
        row["high_value"] = float(row["true_value"])
        rows.append(row)
    rows.sort(key=lambda row: float(row["abs_effect"]), reverse=True)
    return rows[: max(int(top_k), 0)]


def cached_counterfactual_concept_rows(
    workspace,
    instance: Mapping[str, Any],
    baseline_outputs: Mapping[str, object],
    target: str,
    horizon: int | None,
    class_idx: int,
    time_idx: int,
    intervention_mode: str,
    top_k: int,
) -> list[dict[str, object]]:
    context = (
        id(workspace.model),
        str(instance.get("split")),
        int(instance.get("video_index", -1)),
        int(instance.get("timestep", -1)),
        str(target),
        None if horizon is None else int(horizon),
        int(class_idx),
        int(time_idx),
        str(intervention_mode),
        int(top_k),
    )
    cached = st.session_state.get(COUNTERFACTUAL_RANKING_CACHE_KEY)
    if isinstance(cached, dict) and cached.get("context") == context:
        return [dict(row) for row in cached.get("rows", [])]
    with st.spinner("Testing low/high interventions for every concept..."):
        rows = counterfactual_concept_rows(
            workspace,
            instance,
            baseline_outputs,
            target,
            horizon,
            class_idx,
            time_idx,
            intervention_mode,
            top_k=top_k,
        )
    st.session_state[COUNTERFACTUAL_RANKING_CACHE_KEY] = {
        "context": context,
        "rows": [dict(row) for row in rows],
    }
    return rows


def render_counterfactual_importance_chart(
    rows: list[Mapping[str, object]],
    activity_label: str,
    time_label: str,
) -> None:
    st.markdown("##### Top concepts by counterfactual intervention effect")
    st.caption(
        f"Activity: {activity_label} · intervention node: {time_label}. "
        "Every concept is tested alone at state 0.1 and 0.9. Ranking uses the larger "
        "absolute change in the selected activity probability after the full model forward pass."
    )
    if not rows:
        st.info("No counterfactual concept ranking is available for this checkpoint.")
        return
    frame = pd.DataFrame(
        [
            {
                "concept": str(row["concept"]),
                "effect": float(row["recommended_delta"]),
                "abs_effect": float(row["abs_effect"]),
                "recommended": f"set {row['recommended']}",
                "p baseline": float(row["baseline_probability"]),
                "p false": float(row["false_probability"]),
                "delta false": float(row["false_delta"]),
                "p true": float(row["true_probability"]),
                "delta true": float(row["true_delta"]),
                "margin delta": float(row["recommended_margin_delta"]),
            }
            for row in rows
        ]
    )
    max_abs = max(float(frame["abs_effect"].max()), 1e-6)
    scale = alt.Scale(domain=[-1.08 * max_abs, 1.08 * max_abs])
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("effect:Q", title="Selected-activity probability change", scale=scale),
            y=alt.Y(
                "concept:N",
                title=None,
                sort=alt.SortField(field="abs_effect", order="descending"),
                axis=alt.Axis(labelLimit=280),
            ),
            color=alt.condition(
                alt.datum.effect >= 0,
                alt.value("#2f855a"),
                alt.value("#c2414b"),
            ),
            tooltip=[
                alt.Tooltip("concept:N", title="Concept"),
                alt.Tooltip("recommended:N", title="Strongest setting"),
                alt.Tooltip("p baseline:Q", title="Baseline p", format=".4f"),
                alt.Tooltip("p false:Q", title="p at state 0.1", format=".4f"),
                alt.Tooltip("delta false:Q", title="Delta at 0.1", format="+.4f"),
                alt.Tooltip("p true:Q", title="p at state 0.9", format=".4f"),
                alt.Tooltip("delta true:Q", title="Delta at 0.9", format="+.4f"),
                alt.Tooltip("margin delta:Q", title="Decision-margin delta", format="+.4f"),
            ],
        )
        .properties(height=max(190, 38 * len(rows)))
    )
    st.altair_chart(chart, width="stretch")


def render_prediction_options(
    workspace,
    outputs: Mapping[str, object],
    target: str,
    horizon: int | None,
    selected_class_idx: int,
) -> None:
    options = prediction_option_rows(workspace, outputs, target, horizon=horizon, top_k=5)
    alternatives = [row for row in options if int(row["class_idx"]) != int(selected_class_idx)][:3]
    if not alternatives:
        return
    st.caption("Other likely activities for this row")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "rank": row["rank"],
                    "activity": row["class"],
                    "probability": f"{float(row['probability']):.3f}",
                }
                for row in alternatives
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def render_scene_graph_overview(
    workspace,
    instance: Mapping[str, Any],
    baseline_outputs: Mapping[str, object],
    outputs: Mapping[str, object],
    baseline_driver_rows: list[Mapping[str, Any]],
    driver_rows: list[Mapping[str, Any]],
    intervention_items: list[Mapping[str, object]],
    target: str,
) -> None:
    st.markdown("##### Manual intervention graph")
    with st.expander("? How to read graph colors and scores", expanded=False):
        st.markdown(
            "- Node scores use `pre->post`: the value before graph propagation, then the graph-refined value after message passing.\n"
            "- Purple nodes are forecast rollout states at `t+n`. Their `pre->post` score shows the preceding rollout state and the resulting future concept state after graph transition and any activity-belief feedback.\n"
            "- For bounded checkpoints, graph-refined concept scores are probability-like values in [0, 1], and 0.5 is the visual activation threshold.\n"
            "- For older unbounded checkpoints, positive values mean the refined concept state is active in the positive direction; negative values mean it moved below zero.\n"
            "- Green nodes are active; red nodes only appear for older unbounded negative refined values.\n"
            "- Gray nodes are near zero or unchanged context nodes.\n"
            "- Edge thickness grows with learned edge magnitude; green edges are positive and red edges are negative.\n"
            "- Cross-temporal graph edges are adjacent `t -> t+1`; longer-range effects can appear indirectly through repeated graph/autoregressive steps.\n"
            "- Orange marks only the exact concept-time node that you directly intervened on.\n"
            "- Cyan in the after-intervention graph marks non-intervened nodes whose refined value changed compared with the original graph; the node label shows the delta.\n"
            "- Orange edges in the after-intervention graph are edited edges; their labels show the temporary effective value.\n"
            "- Label-belief interventions automatically seed the graph with the concepts whose rollout states changed most.\n"
            f"- The view contains at most {GRAPH_MAX_CONCEPTS} concepts, prioritizing manual choices, intervention endpoints, and the strongest distinct changes.\n"
            "- Edge interventions are temporary forward-pass edits; they do not mutate the checkpoint.\n"
            "- The intervention visibility threshold slider controls which propagated deltas are shown. Bounded graph states can change only slightly when values are already close to 0 or 1."
        )

    available = graph_branch_options(workspace)
    if not available:
        st.caption("This checkpoint does not expose learned graph edges.")
        return

    forecast_steps = scene_graph_forecast_steps(outputs, max_steps=3)
    branches: list[str] = []
    for branch in ("shared", "window" if target == "activity" else "forecast"):
        if branch in available and branch not in branches:
            branches.append(branch)
    if forecast_steps and "forecast" in available and "forecast" not in branches:
        branches.append("forecast")
    if not branches and "legacy" in available:
        branches = ["legacy"]

    manual_indices = st.multiselect(
        "Concepts to include",
        list(range(len(workspace.concept_names))),
        default=[],
        format_func=lambda idx: graph_concept_label(workspace, int(idx)),
        key=GRAPH_SELECTED_CONCEPTS_KEY,
        max_selections=5,
        placeholder="Add a concept to create the graph",
    )
    manual_anchor_indices = {int(index) for index in manual_indices}
    intervention_anchor_indices = scene_graph_intervention_concepts(intervention_items)

    # Intervention threshold slider for showing intervened nodes
    intervention_threshold = st.slider(
        "Intervention visibility threshold",
        min_value=0.0,
        max_value=0.1,
        value=0.01,
        step=0.001,
        key=GRAPH_INTERVENTION_THRESHOLD_KEY,
        help="Show intervened nodes only when their value change exceeds this threshold",
    )

    time_indices = scene_graph_time_indices(
        instance,
        list(baseline_driver_rows) + list(driver_rows),
        intervention_items,
        max_steps=3,
    )
    forecast_steps = scene_graph_forecast_steps(outputs, max_steps=3)
    valid_history = valid_history_time_indices(instance)
    if valid_history:
        current_time = max(valid_history)
        time_indices.extend(current_time + int(step) for step in forecast_steps)
        time_indices = sorted(set(time_indices))

    changed_node_keys: list[tuple[int, int]] = []
    if intervention_items:
        changed_node_keys = scene_graph_changed_node_keys(
            outputs,
            baseline_outputs,
            instance,
            time_indices,
            target,
            exclude_keys=scene_graph_intervention_node_keys(instance, intervention_items),
            limit=GRAPH_CHANGED_NODE_LIMIT,
            threshold=intervention_threshold,
        )
    affected_indices = [int(concept_idx) for concept_idx, _ in changed_node_keys]
    direct_indices = list(dict.fromkeys(
        [int(index) for index in manual_indices] + list(intervention_anchor_indices)
    ))
    anchor_indices = set(direct_indices + affected_indices)

    if not anchor_indices:
        st.caption(
            "No graph yet. Add a concept above, or add a node, edge, or label-belief "
            "intervention. The strongest affected concepts will then be included automatically."
        )
        render_edge_intervention_controls(workspace, instance, [], branches)
        return

    graph_rows = scene_graph_rows(workspace, anchor_indices, branches, edge_limit=3)
    graph_rows = scene_graph_add_intervened_edges(graph_rows, intervention_items)
    if not graph_rows:
        st.caption("No strong learned edges touch the current driver concepts.")
        render_edge_intervention_controls(workspace, instance, [], branches)
        return
    intervention_node_keys = scene_graph_intervention_node_keys(instance, intervention_items)
    visible_limit = min(
        GRAPH_MAX_CONCEPTS,
        max(
            GRAPH_DEFAULT_CONCEPTS,
            len(direct_indices) + min(3, len(affected_indices)),
        ),
    )
    visible_indices = scene_graph_visible_concepts(
        [],
        intervention_items,
        graph_rows,
        max_concepts=visible_limit,
        seed_indices=direct_indices + affected_indices,
    )
    if len(set(direct_indices)) > GRAPH_MAX_CONCEPTS:
        st.warning(
            f"The graph shows at most {GRAPH_MAX_CONCEPTS} concepts. "
            "Some direct selections/intervention endpoints are hidden; remove older edits to inspect them."
        )

    if intervention_items:
        intervened_graph_rows = scene_graph_apply_edge_interventions(graph_rows, intervention_items)
        before_col, after_col = st.columns(2, gap="medium")
        with before_col:
            render_scene_graph_panel(
                "Original",
                workspace,
                instance,
                baseline_outputs,
                graph_rows,
                visible_indices,
                time_indices,
                baseline_driver_rows,
                [],
                set(),
                anchor_indices,
                target,
                reference_outputs=None,
                forced_node_keys=changed_node_keys,
                change_threshold=intervention_threshold,
                height=500,
            )
        with after_col:
            render_scene_graph_panel(
                "After intervention",
                workspace,
                instance,
                outputs,
                intervened_graph_rows,
                visible_indices,
                time_indices,
                driver_rows,
                intervention_items,
                intervention_node_keys,
                anchor_indices,
                target,
                reference_outputs=baseline_outputs,
                forced_node_keys=changed_node_keys,
                change_threshold=intervention_threshold,
                height=500,
            )
    else:
        render_scene_graph_panel(
            "",
            workspace,
            instance,
            outputs,
            graph_rows,
            visible_indices,
            time_indices,
            driver_rows,
            [],
            set(),
            anchor_indices,
            target,
            reference_outputs=None,
            forced_node_keys=set(),
            change_threshold=GRAPH_CHANGE_EPSILON,
            height=520,
        )
    render_edge_intervention_controls(workspace, instance, graph_rows, branches)


def render_scene_graph_panel(
    title: str,
    workspace,
    instance: Mapping[str, Any],
    outputs: Mapping[str, object],
    graph_rows: list[Mapping[str, object]],
    visible_indices: list[int],
    time_indices: list[int],
    driver_rows: list[Mapping[str, Any]],
    intervention_items: list[Mapping[str, object]],
    intervention_node_keys: set[tuple[int, int]],
    anchor_indices: set[int],
    target: str,
    reference_outputs: Mapping[str, object] | None,
    forced_node_keys: set[tuple[int, int]] | list[tuple[int, int]],
    change_threshold: float,
    height: int,
) -> None:
    if title:
        st.caption(title)
    visible_node_keys = scene_graph_visible_node_keys(
        outputs,
        instance,
        driver_rows,
        intervention_items,
        graph_rows,
        visible_indices,
        time_indices,
        anchor_indices,
        target,
        reference_outputs,
        change_threshold=change_threshold,
    )
    visible_node_keys.update((int(concept_idx), int(time_idx)) for concept_idx, time_idx in forced_node_keys)
    render_zoomable_graphviz(
        spatiotemporal_scene_graph_dot(
            workspace,
            instance,
            outputs,
            graph_rows,
            visible_indices,
            time_indices,
            visible_node_keys,
            intervention_node_keys,
            scene_graph_driver_time_scores(driver_rows),
            target,
            reference_outputs,
            change_threshold=change_threshold,
        ),
        height=height,
    )


def render_edge_intervention_controls(
    workspace,
    instance: Mapping[str, Any],
    graph_rows: list[Mapping[str, object]],
    branches: list[str],
) -> None:
    with st.expander("Edge interventions", expanded=False):
        st.caption(
            "Temporary inference-time edits; checkpoint weights are unchanged. Graph edge parameters "
            "are shared across time, so t identifies an occurrence to inspect, while the edit applies "
            "to that branch/layer relation at every eligible t."
        )
        if graph_rows:
            st.markdown("**Quick edit: currently shown edges**")
            edge_options = list(range(min(12, len(graph_rows))))
            selected = st.selectbox(
                "Shown edge",
                edge_options,
                format_func=lambda idx: edge_intervention_label(graph_rows[int(idx)]),
                key="scene_graph_edge_intervention_edge",
            )
            row = graph_rows[int(selected)]
            cols = st.columns(4)
            actions = [
                ("Zero", 0.0, "zero"),
                ("Half", 0.5, "half"),
                ("Double", 2.0, "double"),
                ("Invert", -1.0, "invert"),
            ]
            for col, (label, scale, setting) in zip(cols, actions):
                key = (
                    f"edge_{setting}_{int(row['source_idx'])}_{int(row['target_idx'])}_"
                    f"{row.get('kind', 'spatial')}"
                )
                if col.button(label, key=key):
                    set_active_edge_intervention(instance, row, setting, edge_scale=scale)
                    st.rerun()

        st.divider()
        st.markdown("**Edit any learned edge**")
        if not branches:
            st.caption("No editable graph branch is available.")
            return
        branch = str(
            st.selectbox(
                "Graph branch",
                branches,
                key="manual_edge_branch",
                help="shared feeds both tasks; window is current activity; forecast is future rollout.",
            )
        )
        layer_options = graph_layer_options(workspace, branch)
        layer_option = str(
            st.selectbox(
                "Graph layer",
                layer_options,
                format_func=lambda value: "All layers (mean weight shown)" if value == "mean" else str(value),
                key="manual_edge_layer",
            )
        )
        edge_kind = str(
            st.selectbox(
                "Edge type",
                ["spatial", "temporal", "cross_temporal"],
                format_func=lambda value: {
                    "spatial": "Same-time: source(t) -> target(t)",
                    "temporal": "Same-concept: concept(t-1) -> concept(t)",
                    "cross_temporal": "Cross-time: source(t-1) -> target(t)",
                }[str(value)],
                key="manual_edge_kind",
            )
        )
        time_options = valid_history_time_indices(instance)
        if edge_kind != "spatial" and len(time_options) > 1:
            time_options = time_options[1:]
        if not time_options:
            st.caption("This instance has no valid time occurrence for the selected edge type.")
            return
        target_time_idx = int(
            st.selectbox(
                "t occurrence",
                time_options,
                index=len(time_options) - 1,
                format_func=lambda value: graph_time_option_label(instance, int(value)),
                key="manual_edge_time",
            )
        )
        source_idx = int(
            st.selectbox(
                "Source concept",
                list(range(len(workspace.concept_names))),
                format_func=lambda value: graph_concept_label(workspace, int(value)),
                key="manual_edge_source",
            )
        )
        if edge_kind == "temporal":
            target_idx = source_idx
            st.caption(f"Target concept: {graph_concept_label(workspace, target_idx)} (same-concept temporal edge)")
        else:
            target_idx = int(
                st.selectbox(
                    "Target concept",
                    list(range(len(workspace.concept_names))),
                    format_func=lambda value: graph_concept_label(workspace, int(value)),
                    key="manual_edge_target",
                )
            )
        base_weight = manual_edge_weight(
            workspace,
            branch,
            layer_option,
            edge_kind,
            source_idx,
            target_idx,
        )
        st.caption(f"Effective learned weight: {base_weight:+.6f}")
        edit_mode = str(
            st.selectbox(
                "Edit",
                ["value", "scale"],
                format_func=lambda value: "Set exact edge value" if value == "value" else "Scale learned value",
                key="manual_edge_edit_mode",
            )
        )
        default_edit = (base_weight if abs(base_weight) > 1e-8 else 0.1) if edit_mode == "value" else 0.0
        edit_value = float(
            st.number_input(
                "New value" if edit_mode == "value" else "Scale factor",
                value=float(default_edit),
                step=0.05,
                format="%.6f",
                key=f"manual_edge_edit_{edit_mode}",
            )
        )
        layer_index = None if layer_option == "mean" else int(layer_option.rsplit(".", 1)[-1])
        source_time_idx = target_time_idx if edge_kind == "spatial" else target_time_idx - 1
        manual_row = {
            "branch": branch,
            "intervention_branch": branch,
            "layer_index": layer_index,
            "kind": edge_kind,
            "source_idx": source_idx,
            "target_idx": target_idx,
            "source": graph_concept_label(workspace, source_idx),
            "target": graph_concept_label(workspace, target_idx),
            "weight": base_weight,
            "edge_time_idx": target_time_idx,
            "source_time_idx": source_time_idx,
            "target_time_idx": target_time_idx,
            "original_t": scene_graph_original_timestep(instance, target_time_idx),
        }
        if st.button("Add edge intervention", width="stretch", key="manual_edge_add"):
            if edit_mode == "value":
                set_active_edge_intervention(
                    instance,
                    manual_row,
                    f"set {edit_value:+.3f}",
                    edge_value=edit_value,
                )
            else:
                set_active_edge_intervention(
                    instance,
                    manual_row,
                    f"scale x{edit_value:+.3f}",
                    edge_scale=edit_value,
                )
            st.rerun()


def edge_intervention_label(row: Mapping[str, object]) -> str:
    kind = str(row.get("kind", "spatial")).replace("_", " ")
    return (
        f"{kind}: {row.get('source')} -> {row.get('target')} "
        f"({float(row.get('weight', 0.0)):+.3f})"
    )


def manual_edge_weight(
    workspace,
    branch: str,
    layer_option: str,
    edge_kind: str,
    source_idx: int,
    target_idx: int,
) -> float:
    if edge_kind == "temporal":
        vector = graph_temporal_vector(workspace, branch, layer_option)
        if 0 <= int(source_idx) < int(vector.size):
            return float(vector[int(source_idx)])
        return 0.0
    matrix_kind = "cross temporal" if edge_kind == "cross_temporal" else "spatial"
    matrix = graph_matrix(workspace, branch, layer_option, edge_kind=matrix_kind)
    if (
        matrix.ndim == 2
        and 0 <= int(source_idx) < matrix.shape[0]
        and 0 <= int(target_idx) < matrix.shape[1]
    ):
        return float(matrix[int(source_idx), int(target_idx)])
    return 0.0


def render_zoomable_graphviz(dot_source: str, height: int) -> None:
    try:
        completed = subprocess.run(
            ["dot", "-Tsvg"],
            input=dot_source,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        st.graphviz_chart(dot_source, width="stretch", height=height)
        return

    svg = completed.stdout
    html = f"""
    <html>
      <head>
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            overflow: hidden;
          }}
          .graph-shell {{
            height: {int(height)}px;
            width: 100%;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 6px;
            overflow: hidden;
          }}
          .graph-toolbar {{
            box-sizing: border-box;
            height: 38px;
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 6px;
            padding: 5px 8px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.45);
          }}
          .graph-toolbar button {{
            box-sizing: border-box;
            min-width: 30px;
            height: 28px;
            padding: 0 8px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 4px;
            background: rgba(30, 41, 59, 0.9);
            color: #e5e7eb;
            font: 600 12px/1 Helvetica, Arial, sans-serif;
            cursor: pointer;
          }}
          .graph-toolbar button:hover {{
            background: rgba(51, 65, 85, 0.95);
          }}
          .graph-viewport {{
            height: {max(120, int(height) - 38)}px;
            width: 100%;
            overflow: auto;
            background: transparent;
            cursor: grab;
            user-select: none;
          }}
          .graph-viewport.dragging {{
            cursor: grabbing;
          }}
          .graph-content {{
            position: relative;
            display: inline-block;
            min-width: 100%;
            min-height: 100%;
          }}
          .graph-scaled {{
            position: absolute;
            left: 0;
            top: 0;
            display: inline-block;
            transform: scale(1);
            transform-origin: 0 0;
          }}
          .graph-scaled svg {{
            max-width: none !important;
            display: block;
          }}
        </style>
      </head>
      <body>
        <div class="graph-shell">
          <div class="graph-toolbar">
            <button id="zoom-out" title="Zoom out" aria-label="Zoom out">-</button>
            <button id="zoom-in" title="Zoom in" aria-label="Zoom in">+</button>
            <button id="actual-size" title="Actual size" aria-label="Actual size">1:1</button>
            <button id="fit-graph" title="Fit graph" aria-label="Fit graph">Fit</button>
          </div>
          <div class="graph-viewport">
            <div class="graph-content">
              <div class="graph-scaled">{svg}</div>
            </div>
          </div>
        </div>
        <script>
          const viewport = document.querySelector('.graph-viewport');
          const content = document.querySelector('.graph-content');
          const scaled = document.querySelector('.graph-scaled');
          const svg = scaled.querySelector('svg');
          let scale = 1.0;
          const minScale = 0.20;
          const maxScale = 4.0;
          const viewBox = svg.viewBox && svg.viewBox.baseVal;
          const baseWidth = Math.max(1, viewBox && viewBox.width ? viewBox.width : svg.getBoundingClientRect().width);
          const baseHeight = Math.max(1, viewBox && viewBox.height ? viewBox.height : svg.getBoundingClientRect().height);
          svg.style.width = `${{baseWidth}}px`;
          svg.style.height = `${{baseHeight}}px`;

          function layout() {{
            const scaledWidth = baseWidth * scale;
            const scaledHeight = baseHeight * scale;
            const left = Math.max(12, (viewport.clientWidth - scaledWidth) / 2);
            const top = Math.max(12, (viewport.clientHeight - scaledHeight) / 2);
            scaled.style.left = `${{left}}px`;
            scaled.style.top = `${{top}}px`;
            scaled.style.transform = `scale(${{scale}})`;
            content.style.width = `${{Math.ceil(Math.max(viewport.clientWidth, scaledWidth + 24))}}px`;
            content.style.height = `${{Math.ceil(Math.max(viewport.clientHeight, scaledHeight + 24))}}px`;
          }}

          function applyScale(nextScale, originX, originY) {{
            const previous = scale;
            scale = Math.max(minScale, Math.min(maxScale, nextScale));
            const beforeLeft = viewport.scrollLeft + originX;
            const beforeTop = viewport.scrollTop + originY;
            const ratio = scale / previous;
            layout();
            viewport.scrollLeft = beforeLeft * ratio - originX;
            viewport.scrollTop = beforeTop * ratio - originY;
          }}

          function fitGraph() {{
            const widthScale = (viewport.clientWidth - 28) / baseWidth;
            const heightScale = (viewport.clientHeight - 28) / baseHeight;
            scale = Math.max(minScale, Math.min(1.35, widthScale, heightScale));
            layout();
            viewport.scrollLeft = 0;
            viewport.scrollTop = 0;
          }}

          requestAnimationFrame(fitGraph);
          document.getElementById('fit-graph').addEventListener('click', fitGraph);
          document.getElementById('actual-size').addEventListener('click', () => {{
            applyScale(1.0, viewport.clientWidth / 2, viewport.clientHeight / 2);
          }});
          document.getElementById('zoom-in').addEventListener('click', () => {{
            applyScale(scale * 1.18, viewport.clientWidth / 2, viewport.clientHeight / 2);
          }});
          document.getElementById('zoom-out').addEventListener('click', () => {{
            applyScale(scale / 1.18, viewport.clientWidth / 2, viewport.clientHeight / 2);
          }});

          viewport.addEventListener('wheel', (event) => {{
            event.preventDefault();
            const rect = viewport.getBoundingClientRect();
            const originX = event.clientX - rect.left;
            const originY = event.clientY - rect.top;
            const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
            applyScale(scale * factor, originX, originY);
          }}, {{ passive: false }});

          let dragging = false;
          let dragStartX = 0;
          let dragStartY = 0;
          let scrollStartLeft = 0;
          let scrollStartTop = 0;

          viewport.addEventListener('pointerdown', (event) => {{
            dragging = true;
            viewport.classList.add('dragging');
            viewport.setPointerCapture(event.pointerId);
            dragStartX = event.clientX;
            dragStartY = event.clientY;
            scrollStartLeft = viewport.scrollLeft;
            scrollStartTop = viewport.scrollTop;
          }});

          viewport.addEventListener('pointermove', (event) => {{
            if (!dragging) return;
            viewport.scrollLeft = scrollStartLeft - (event.clientX - dragStartX);
            viewport.scrollTop = scrollStartTop - (event.clientY - dragStartY);
          }});

          viewport.addEventListener('pointerup', (event) => {{
            dragging = false;
            viewport.classList.remove('dragging');
            viewport.releasePointerCapture(event.pointerId);
          }});

          viewport.addEventListener('dblclick', (event) => {{
            fitGraph();
          }});

          const resizeObserver = new ResizeObserver(() => fitGraph());
          resizeObserver.observe(viewport);
        </script>
      </body>
    </html>
    """
    st.iframe(
        "data:text/html;charset=utf-8," + quote(html),
        width="stretch",
        height=int(height) + 8,
    )


def scene_graph_rows(
    workspace,
    anchor_indices: set[int],
    branches: list[str],
    edge_limit: int,
) -> list[dict[str, object]]:
    rows = []
    candidate_limit = max(int(edge_limit), len(workspace.concept_names) ** 2)
    for branch in branches:
        layer_options = graph_layer_options(workspace, branch)
        layer = "mean" if "mean" in layer_options else layer_options[0]
        spatial_rows = graph_edges_for_concepts(
            workspace,
            sorted(anchor_indices),
            branch=branch,
            layer_option=layer,
            top_k=candidate_limit,
        )
        for row in scene_graph_prioritized_rows(spatial_rows, anchor_indices, edge_limit):
            rows.append(
                {
                    "branch": branch,
                    "kind": "spatial",
                    "source_idx": int(row["source_idx"]),
                    "target_idx": int(row["target_idx"]),
                    "source": row["source"],
                    "target": row["target"],
                    "weight": float(row["weight"]),
                    "touches": row.get("touches_driver_as", ""),
                }
            )
        cross_temporal_rows = graph_cross_temporal_rows(
            workspace,
            anchor_indices,
            branch,
            layer,
            top_k=candidate_limit,
        )
        for row in scene_graph_prioritized_rows(cross_temporal_rows, anchor_indices, edge_limit):
            rows.append(row)
    return aggregate_scene_graph_rows(rows)


def scene_graph_prioritized_rows(
    rows: list[Mapping[str, object]],
    anchor_indices: set[int],
    limit: int,
) -> list[Mapping[str, object]]:
    selected = {int(index) for index in anchor_indices}
    return sorted(
        rows,
        key=lambda row: (
            int(row["source_idx"]) in selected and int(row["target_idx"]) in selected,
            abs(float(row.get("weight", 0.0))),
        ),
        reverse=True,
    )[: max(int(limit), 0)]


def aggregate_scene_graph_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row.get("kind", "spatial")), int(row["source_idx"]), int(row["target_idx"]))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, object]] = []
    for (kind, source_idx, target_idx), group in grouped.items():
        weights = [float(row["weight"]) for row in group]
        branches = sorted({str(row.get("branch", "")) for row in group if row.get("branch")})
        touches = sorted({str(row.get("touches", "")) for row in group if row.get("touches")})
        first = group[0]
        aggregated.append(
            {
                "branch": "+".join(branches),
                "kind": kind,
                "source_idx": source_idx,
                "target_idx": target_idx,
                "source": first["source"],
                "target": first["target"],
                "weight": float(sum(weights) / max(len(weights), 1)),
                "touches": "+".join(touches),
                "edge_count": len(group),
            }
        )
    aggregated.sort(key=lambda row: abs(float(row["weight"])), reverse=True)
    return aggregated


def scene_graph_edge_item_matches(
    row: Mapping[str, object],
    item: Mapping[str, object],
) -> bool:
    if str(item.get("item_type", item.get("type", ""))).lower() != "edge":
        return False
    if (
        str(row.get("kind", "spatial")) != str(item.get("edge_kind", "spatial"))
        or int(row.get("source_idx", -1)) != int(item.get("source_idx", -2))
        or int(row.get("target_idx", -1)) != int(item.get("target_idx", -2))
    ):
        return False
    item_branch = str(item.get("branch", "all"))
    row_branches = {value for value in str(row.get("branch", "")).split("+") if value}
    return item_branch == "all" or item_branch in row_branches


def scene_graph_add_intervened_edges(
    graph_rows: list[Mapping[str, object]],
    intervention_items: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = [dict(row) for row in graph_rows]
    for item in intervention_items:
        if str(item.get("item_type", item.get("type", ""))).lower() != "edge":
            continue
        if any(scene_graph_edge_item_matches(row, item) for row in rows):
            continue
        branch = str(item.get("branch", "all"))
        rows.append(
            {
                "branch": "legacy" if branch == "all" else branch,
                "kind": str(item.get("edge_kind", "spatial")),
                "source_idx": int(item["source_idx"]),
                "target_idx": int(item["target_idx"]),
                "source": str(item.get("source", item["source_idx"])),
                "target": str(item.get("target", item["target_idx"])),
                "weight": float(item.get("base_weight", 0.0)),
                "touches": "intervention",
                "edge_count": 1,
            }
        )
    return rows


def scene_graph_apply_edge_interventions(
    graph_rows: list[Mapping[str, object]],
    intervention_items: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = [dict(row) for row in graph_rows]
    for row in rows:
        matching = [
            item
            for item in intervention_items
            if scene_graph_edge_item_matches(row, item)
        ]
        if not matching:
            continue
        item = matching[-1]
        if item.get("edge_value") is not None:
            row["weight"] = float(item["edge_value"])
        elif item.get("edge_scale") is not None:
            row["weight"] = float(row.get("weight", 0.0)) * float(item["edge_scale"])
        row["intervened"] = True
    return rows


def graph_cross_temporal_rows(
    workspace,
    anchor_indices: set[int],
    branch: str,
    layer: str,
    top_k: int,
) -> list[dict[str, object]]:
    matrix = graph_matrix(workspace, branch, layer, edge_kind="cross temporal")
    if matrix.size == 0:
        return []
    selected = {int(index) for index in anchor_indices}
    rows = []
    for source, target in sorted(
        np.argwhere(np.abs(matrix) > 0.0),
        key=lambda pair: abs(float(matrix[int(pair[0]), int(pair[1])])),
        reverse=True,
    ):
        source_idx = int(source)
        target_idx = int(target)
        if source_idx == target_idx:
            continue
        if source_idx not in selected and target_idx not in selected:
            continue
        rows.append(
            {
                "branch": branch,
                "kind": "cross_temporal",
                "source_idx": source_idx,
                "target_idx": target_idx,
                "source": graph_concept_label(workspace, source_idx),
                "target": graph_concept_label(workspace, target_idx),
                "weight": float(matrix[source_idx, target_idx]),
                "touches": _scene_graph_edge_touch_label(source_idx, target_idx, selected),
            }
        )
        if len(rows) >= int(top_k):
            break
    return rows


def _scene_graph_edge_touch_label(source_idx: int, target_idx: int, selected: set[int]) -> str:
    if int(source_idx) in selected and int(target_idx) in selected:
        return "both"
    if int(source_idx) in selected:
        return "source"
    if int(target_idx) in selected:
        return "target"
    return ""


def graph_concept_candidates(
    graph_rows: list[Mapping[str, object]],
    anchor_indices: set[int],
) -> set[int]:
    candidates = set(anchor_indices)
    for row in graph_rows:
        candidates.add(int(row["source_idx"]))
        candidates.add(int(row["target_idx"]))
    return candidates


def graph_concept_label(workspace, concept_idx: int) -> str:
    if 0 <= int(concept_idx) < len(workspace.concept_names):
        return str(workspace.concept_names[int(concept_idx)])
    return str(concept_idx)


def scene_graph_time_labels(
    driver_rows: list[Mapping[str, Any]],
    intervention_items: list[Mapping[str, object]],
) -> dict[int, list[str]]:
    labels: dict[int, list[str]] = {}
    for row in driver_rows:
        concept_idx = row.get("concept_idx")
        if concept_idx is None:
            continue
        add_graph_time_label(
            labels,
            int(concept_idx),
            row.get("original_t"),
            row.get("soft_binary"),
            row.get("contribution"),
        )
    for item in intervention_items:
        concept_idx = item.get("concept_idx")
        if concept_idx is None:
            continue
        add_graph_time_label(labels, int(concept_idx), item.get("original_t"), item.get("setting"), None)
    return labels


def add_graph_time_label(
    labels: dict[int, list[str]],
    concept_idx: int,
    value: object,
    fires_value: object,
    contribution: object,
) -> None:
    if value is None:
        return
    if fires_value is None:
        fire_label = ""
    elif isinstance(fires_value, str):
        fire_label = f" {fires_value}"
    else:
        fire_label = " fires" if bool(fires_value) else " off"
    contribution_label = ""
    if contribution is not None:
        try:
            contribution_label = f" {float(contribution):+.2f}"
        except (TypeError, ValueError):
            contribution_label = ""
    label = f"t={value}{fire_label}{contribution_label}"
    current = labels.setdefault(concept_idx, [])
    if label not in current:
        current.append(label)


def scene_graph_driver_scores(driver_rows: list[Mapping[str, Any]]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for row in driver_rows:
        concept_idx = row.get("concept_idx")
        if concept_idx is None:
            continue
        try:
            scores[int(concept_idx)] = scores.get(int(concept_idx), 0.0) + float(row.get("contribution", 0.0))
        except (TypeError, ValueError):
            continue
    return scores


def scene_graph_driver_time_scores(driver_rows: list[Mapping[str, Any]]) -> dict[tuple[int, int], float]:
    scores: dict[tuple[int, int], float] = {}
    for row in driver_rows:
        concept_idx = row.get("concept_idx")
        history_t = row.get("history_t")
        if concept_idx is None or history_t is None:
            continue
        try:
            key = (int(concept_idx), int(history_t))
            scores[key] = scores.get(key, 0.0) + float(row.get("contribution", 0.0))
        except (TypeError, ValueError):
            continue
    return scores


def scene_graph_time_indices(
    instance: Mapping[str, Any],
    driver_rows: list[Mapping[str, Any]],
    intervention_items: list[Mapping[str, object]],
    max_steps: int = 4,
) -> list[int]:
    mask = list(instance.get("key_padding_mask", []))
    valid_indices = [index for index, is_pad in enumerate(mask) if not bool(is_pad)]
    if not valid_indices:
        return []
    selected = {
        int(row["history_t"])
        for row in driver_rows
        if row.get("history_t") is not None and int(row["history_t"]) in valid_indices
    }
    selected.update(
        int(item["time_idx"])
        for item in intervention_items
        if item.get("time_idx") is not None and int(item["time_idx"]) in valid_indices
    )
    current = max(valid_indices)
    while len(selected) < max_steps and current in valid_indices:
        selected.add(current)
        current -= 1
    return sorted(selected)[-int(max_steps):]


def scene_graph_forecast_steps(
    outputs: Mapping[str, object],
    max_steps: int = 3,
) -> list[int]:
    predicted = outputs.get("predicted_concepts_by_step")
    if not isinstance(predicted, Mapping):
        return []
    steps: list[int] = []
    for raw_step, states in predicted.items():
        try:
            step = int(raw_step)
        except (TypeError, ValueError):
            continue
        if step > 0 and hasattr(states, "detach"):
            steps.append(step)
    return sorted(set(steps))[: max(int(max_steps), 0)]


def scene_graph_forecast_step(
    instance: Mapping[str, Any],
    time_idx: int,
) -> int | None:
    valid = valid_history_time_indices(instance)
    if not valid:
        return None
    step = int(time_idx) - max(valid)
    return step if step > 0 else None


def scene_graph_rollout_concept_value(
    outputs: Mapping[str, object],
    mapping_key: str,
    step: int,
    concept_idx: int,
) -> float | None:
    values_by_step = outputs.get(mapping_key)
    if not isinstance(values_by_step, Mapping):
        return None
    states = values_by_step.get(int(step), values_by_step.get(str(int(step))))
    if not hasattr(states, "detach"):
        return None
    try:
        if states.ndim >= 3:
            value = states[0, -1, int(concept_idx)]
        elif states.ndim == 2:
            value = states[0, int(concept_idx)]
        elif states.ndim == 1:
            value = states[int(concept_idx)]
        else:
            return None
        return float(value.detach().cpu().item())
    except (IndexError, RuntimeError, TypeError, ValueError):
        return None


def scene_graph_visible_concepts(
    driver_rows: list[Mapping[str, Any]],
    intervention_items: list[Mapping[str, object]],
    graph_rows: list[Mapping[str, object]],
    max_concepts: int = 10,
    seed_indices: list[int] | set[int] | None = None,
) -> list[int]:
    visible: list[int] = []
    for concept_idx in seed_indices or []:
        add_visible_concept(visible, concept_idx, max_concepts)
    for row in sorted(driver_rows, key=lambda item: abs(float(item.get("contribution", 0.0))), reverse=True):
        add_visible_concept(visible, row.get("concept_idx"), max_concepts)
    for item in intervention_items:
        add_visible_concept(visible, item.get("concept_idx"), max_concepts)
    for row in graph_rows:
        add_visible_concept(visible, row.get("source_idx"), max_concepts)
        add_visible_concept(visible, row.get("target_idx"), max_concepts)
    return visible


def add_visible_concept(visible: list[int], concept_idx: object, max_concepts: int) -> None:
    if concept_idx is None or len(visible) >= int(max_concepts):
        return
    index = int(concept_idx)
    if index not in visible:
        visible.append(index)


def scene_graph_intervention_concepts(
    intervention_items: list[Mapping[str, object]],
) -> list[int]:
    concepts: list[int] = []
    for item in intervention_items:
        item_type = str(item.get("item_type", item.get("type", "concept"))).lower()
        candidates = (
            (item.get("source_idx"), item.get("target_idx"))
            if item_type == "edge"
            else (item.get("concept_idx"),)
        )
        for concept_idx in candidates:
            if concept_idx is not None and int(concept_idx) not in concepts:
                concepts.append(int(concept_idx))
    return concepts


def scene_graph_intervention_node_keys(
    instance: Mapping[str, Any],
    intervention_items: list[Mapping[str, object]],
) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    valid_history = valid_history_time_indices(instance)
    current_time = max(valid_history) if valid_history else None
    for item in intervention_items:
        if item.get("concept_idx") is None:
            continue
        if item.get("time_idx") is not None:
            keys.add((int(item["concept_idx"]), int(item["time_idx"])))
        elif item.get("rollout_step") is not None and current_time is not None:
            keys.add((int(item["concept_idx"]), current_time + int(item["rollout_step"])))
    return keys


def scene_graph_intervention_connected_node_keys(
    graph_rows: list[Mapping[str, object]],
    intervention_node_keys: set[tuple[int, int]],
    time_indices: list[int],
) -> set[tuple[int, int]]:
    if not intervention_node_keys:
        return set()
    time_set = {int(time_idx) for time_idx in time_indices}
    connected: set[tuple[int, int]] = set()
    connected.update((int(concept_idx), int(time_idx)) for concept_idx, time_idx in intervention_node_keys)

    for row in graph_rows:
        source_idx = int(row["source_idx"])
        target_idx = int(row["target_idx"])
        kind = str(row.get("kind", "spatial"))
        for concept_idx, time_idx in intervention_node_keys:
            concept = int(concept_idx)
            time = int(time_idx)
            if kind == "cross_temporal":
                if source_idx == concept and time + 1 in time_set:
                    connected.add((target_idx, time + 1))
                if target_idx == concept and time - 1 in time_set:
                    connected.add((source_idx, time - 1))
            else:
                if source_idx == concept and time in time_set:
                    connected.add((target_idx, time))
                if target_idx == concept and time in time_set:
                    connected.add((source_idx, time))

    # Same-concept temporal edges are drawn explicitly in the scene graph, so
    # allow immediate same-concept changes only where that visual edge exists.
    for concept_idx, time_idx in intervention_node_keys:
        concept = int(concept_idx)
        time = int(time_idx)
        if time - 1 in time_set:
            connected.add((concept, time - 1))
        if time + 1 in time_set:
            connected.add((concept, time + 1))
    return connected


def scene_graph_changed_node_keys(
    outputs: Mapping[str, object],
    reference_outputs: Mapping[str, object],
    instance: Mapping[str, Any],
    time_indices: list[int],
    target: str,
    allowed_concept_indices: set[int] | None = None,
    allowed_node_keys: set[tuple[int, int]] | None = None,
    exclude_keys: set[tuple[int, int]] | None = None,
    limit: int = GRAPH_CHANGED_NODE_LIMIT,
    threshold: float = GRAPH_CHANGE_EPSILON,
) -> list[tuple[int, int]]:
    excluded = set(exclude_keys or set())
    allowed = None if allowed_concept_indices is None else {int(index) for index in allowed_concept_indices}
    allowed_keys = None if allowed_node_keys is None else {
        (int(concept_idx), int(time_idx)) for concept_idx, time_idx in allowed_node_keys
    }
    candidates: list[tuple[float, int, int]] = []
    concept_count = len(instance.get("concept_names", []))
    if concept_count <= 0:
        history_tensor, _ = scene_graph_activation_tensor(outputs, target)
        concept_count = int(history_tensor.shape[-1]) if hasattr(history_tensor, "shape") else 0
    for time_idx in time_indices:
        time = int(time_idx)
        for concept_idx in range(concept_count):
            key = (int(concept_idx), time)
            if allowed_keys is not None and key not in allowed_keys:
                continue
            if allowed is not None and int(concept_idx) not in allowed:
                continue
            _, after = scene_graph_activation(outputs, instance, time, concept_idx, target)
            _, before = scene_graph_activation(reference_outputs, instance, time, concept_idx, target)
            if after is None or before is None:
                continue
            magnitude = abs(float(after) - float(before))
            if key in excluded or magnitude < threshold:
                continue
            candidates.append((magnitude, int(concept_idx), time))
    candidates.sort(reverse=True)
    selected: list[tuple[int, int]] = []
    seen_concepts: set[int] = set()
    for _, concept_idx, time_idx in candidates:
        if concept_idx in seen_concepts:
            continue
        selected.append((concept_idx, time_idx))
        seen_concepts.add(concept_idx)
        if len(selected) >= int(limit):
            break
    return selected


def scene_graph_visible_node_keys(
    outputs: Mapping[str, object],
    instance: Mapping[str, Any],
    driver_rows: list[Mapping[str, Any]],
    intervention_items: list[Mapping[str, object]],
    graph_rows: list[Mapping[str, object]],
    concept_indices: list[int],
    time_indices: list[int],
    anchor_indices: set[int],
    target: str,
    reference_outputs: Mapping[str, object] | None = None,
    change_threshold: float = GRAPH_CHANGE_EPSILON,
) -> set[tuple[int, int]]:
    valid_history = valid_history_time_indices(instance)
    current_time = max(valid_history) if valid_history else None
    concept_set = set(int(index) for index in concept_indices)
    time_set = set(int(index) for index in time_indices)
    keys: set[tuple[int, int]] = set()

    for row in driver_rows:
        add_scene_graph_key(keys, row.get("concept_idx"), row.get("history_t"), concept_set, time_set)
    for item in intervention_items:
        add_scene_graph_key(keys, item.get("concept_idx"), item.get("time_idx"), concept_set, time_set)

    important_concepts = {concept_idx for concept_idx, _ in keys}
    if current_time is not None:
        for concept_idx in important_concepts:
            keys.add((int(concept_idx), int(current_time)))

    neighbor_counts: dict[int, int] = {}
    for row in sorted(graph_rows, key=lambda item: abs(float(item.get("weight", 0.0))), reverse=True):
        for endpoint in ("source_idx", "target_idx"):
            concept_idx = int(row[endpoint])
            if concept_idx not in concept_set or concept_idx in important_concepts:
                continue
            if neighbor_counts.get(concept_idx, 0) >= 1 or current_time is None:
                continue
            fires, _ = scene_graph_activation(outputs, instance, current_time, concept_idx, target)
            if fires:
                keys.add((concept_idx, int(current_time)))
                neighbor_counts[concept_idx] = neighbor_counts.get(concept_idx, 0) + 1

    for concept_idx in list(important_concepts):
        for time_idx in time_indices:
            fires, _ = scene_graph_activation(outputs, instance, time_idx, concept_idx, target)
            if fires:
                keys.add((int(concept_idx), int(time_idx)))

    # Keep the structural edges readable: selected concepts appear across the
    # displayed history, and their strongest graph neighbors appear wherever
    # the selected edge needs an endpoint.
    selected_concepts = set(int(index) for index in anchor_indices)
    selected_concepts.update(
        int(item["concept_idx"])
        for item in intervention_items
        if item.get("concept_idx") is not None
    )
    for concept_idx in selected_concepts.intersection(concept_set):
        for time_idx in time_indices:
            keys.add((int(concept_idx), int(time_idx)))

    for row in graph_rows:
        source_idx = int(row["source_idx"])
        target_idx = int(row["target_idx"])
        if source_idx not in concept_set or target_idx not in concept_set:
            continue
        if source_idx not in selected_concepts and target_idx not in selected_concepts:
            continue
        kind = str(row.get("kind", "spatial"))
        if kind == "cross_temporal":
            for left, right in zip(time_indices, time_indices[1:]):
                keys.add((source_idx, int(left)))
                keys.add((target_idx, int(right)))
        else:
            for time_idx in time_indices:
                keys.add((source_idx, int(time_idx)))
                keys.add((target_idx, int(time_idx)))

    if reference_outputs is not None:
        for concept_idx in concept_indices:
            for time_idx in time_indices:
                _, value = scene_graph_activation(outputs, instance, time_idx, concept_idx, target)
                _, reference_value = scene_graph_activation(reference_outputs, instance, time_idx, concept_idx, target)
                if value is None or reference_value is None:
                    continue
                if abs(float(value) - float(reference_value)) >= float(change_threshold):
                    keys.add((int(concept_idx), int(time_idx)))

    if not keys and current_time is not None:
        for concept_idx in concept_indices[:5]:
            keys.add((int(concept_idx), int(current_time)))
    return keys


def add_scene_graph_key(
    keys: set[tuple[int, int]],
    concept_idx: object,
    time_idx: object,
    concept_set: set[int],
    time_set: set[int],
) -> None:
    if concept_idx is None or time_idx is None:
        return
    concept = int(concept_idx)
    time = int(time_idx)
    if concept in concept_set and time in time_set:
        keys.add((concept, time))


def spatiotemporal_scene_graph_dot(
    workspace,
    instance: Mapping[str, Any],
    outputs: Mapping[str, object],
    graph_rows: list[Mapping[str, object]],
    concept_indices: list[int],
    time_indices: list[int],
    visible_node_keys: set[tuple[int, int]],
    intervention_node_keys: set[tuple[int, int]],
    driver_time_scores: Mapping[tuple[int, int], float],
    target: str,
    reference_outputs: Mapping[str, object] | None,
    change_threshold: float = GRAPH_CHANGE_EPSILON,
) -> str:
    lines = [
        "digraph SpatioTemporalSceneGraph {",
        "  graph [rankdir=LR, bgcolor=\"transparent\", pad=\"0.10\", nodesep=\"0.18\", ranksep=\"0.52\", margin=\"0.02\", splines=polyline, ratio=compress, newrank=true];",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=11, fontcolor=\"#f8fafc\", color=\"#94a3b8\", penwidth=1.6, margin=\"0.10,0.07\"];",
        "  edge [fontname=\"Helvetica\", fontsize=9, arrowsize=0.60, penwidth=1.7, color=\"#94a3b8\"];",
    ]
    lines.extend(
        spatiotemporal_scene_graph_body_lines(
            workspace,
            instance,
            outputs,
            graph_rows,
            concept_indices,
            time_indices,
            visible_node_keys,
            intervention_node_keys,
            driver_time_scores,
            node_prefix="g",
            target=target,
            reference_outputs=reference_outputs,
            change_threshold=change_threshold,
        )
    )
    lines.append("}")
    return "\n".join(lines)


def comparison_spatiotemporal_scene_graph_dot(
    workspace,
    instance: Mapping[str, Any],
    graph_rows: list[Mapping[str, object]],
    concept_indices: list[int],
    time_indices: list[int],
    baseline_outputs: Mapping[str, object],
    baseline_node_keys: set[tuple[int, int]],
    baseline_intervention_indices: set[int],
    baseline_driver_time_scores: Mapping[tuple[int, int], float],
    intervention_outputs: Mapping[str, object],
    intervention_node_keys: set[tuple[int, int]],
    intervention_indices: set[int],
    intervention_driver_time_scores: Mapping[tuple[int, int], float],
    target: str,
) -> str:
    lines = [
        "digraph SpatioTemporalSceneGraphComparison {",
        "  graph [rankdir=LR, bgcolor=\"transparent\", pad=\"0.08\", nodesep=\"0.22\", ranksep=\"0.62\", margin=\"0.02\", splines=true, compound=true];",
        "  node [fontname=\"Helvetica\", fontsize=12, fontcolor=\"#f8fafc\", color=\"#94a3b8\", penwidth=1.6];",
        "  edge [fontname=\"Helvetica\", fontsize=10, arrowsize=0.65, penwidth=1.7, color=\"#94a3b8\"];",
        "  subgraph cluster_original {",
        "    label=\"Original\";",
        "    color=\"#334155\";",
        "    fontcolor=\"#f8fafc\";",
        "    style=\"rounded\";",
    ]
    lines.extend(
        spatiotemporal_scene_graph_body_lines(
            workspace,
            instance,
            baseline_outputs,
            graph_rows,
            concept_indices,
            time_indices,
            baseline_node_keys,
            baseline_intervention_indices,
            baseline_driver_time_scores,
            node_prefix="orig",
            target=target,
            reference_outputs=None,
        )
    )
    lines.extend(
        [
            "  }",
            "  subgraph cluster_intervention {",
            "    label=\"After intervention\";",
            "    color=\"#334155\";",
            "    fontcolor=\"#f8fafc\";",
            "    style=\"rounded\";",
        ]
    )
    lines.extend(
        spatiotemporal_scene_graph_body_lines(
            workspace,
            instance,
            intervention_outputs,
            graph_rows,
            concept_indices,
            time_indices,
            intervention_node_keys,
            intervention_indices,
            intervention_driver_time_scores,
            node_prefix="after",
            target=target,
            reference_outputs=baseline_outputs,
        )
    )
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def spatiotemporal_scene_graph_body_lines(
    workspace,
    instance: Mapping[str, Any],
    outputs: Mapping[str, object],
    graph_rows: list[Mapping[str, object]],
    concept_indices: list[int],
    time_indices: list[int],
    visible_node_keys: set[tuple[int, int]],
    intervention_node_keys: set[tuple[int, int]],
    driver_time_scores: Mapping[tuple[int, int], float],
    node_prefix: str,
    target: str,
    reference_outputs: Mapping[str, object] | None,
    change_threshold: float = GRAPH_CHANGE_EPSILON,
) -> list[str]:
    valid_history = valid_history_time_indices(instance)
    current_time = max(valid_history) if valid_history else None
    labelled_times = scene_graph_labelled_times(visible_node_keys)
    lines: list[str] = []

    for time_idx in time_indices:
        header_label = scene_graph_time_header_html(instance, time_idx, current_time)
        lines.append(
            f'  {scene_graph_header_id(node_prefix, time_idx)} [shape=plain, label=<{header_label}>];'
        )

    for concept_idx in concept_indices:
        concept_name = graph_concept_label(workspace, concept_idx)
        is_probability = scene_graph_activation_is_probability(outputs, target)
        for time_idx in time_indices:
            if (int(concept_idx), int(time_idx)) not in visible_node_keys:
                continue
            node_id = scene_graph_node_id(node_prefix, concept_idx, time_idx)
            fires, value = scene_graph_activation(outputs, instance, time_idx, concept_idx, target)
            pre_value = scene_graph_pre_graph_value(outputs, instance, time_idx, concept_idx)
            _, reference_value = scene_graph_activation(reference_outputs or {}, instance, time_idx, concept_idx, target)
            delta = None if reference_outputs is None or value is None or reference_value is None else value - reference_value
            fill, border = scene_graph_node_colors(
                concept_idx,
                time_idx,
                fires,
                intervention_node_keys,
                driver_time_scores,
                value,
                delta,
                is_probability=is_probability,
                is_forecast=scene_graph_forecast_step(instance, time_idx) is not None,
                change_threshold=change_threshold,
            )
            penwidth = 2.8 if time_idx == current_time else 1.6
            label = scene_graph_node_label(
                concept_name,
                pre_value,
                value,
                labelled_times.get(int(concept_idx)) == int(time_idx),
                delta,
                change_threshold=change_threshold,
            )
            lines.append(
                f'  {node_id} [label="{dot_label_escape(label)}", fillcolor="{fill}", color="{border}", '
                f'penwidth="{penwidth:.1f}", width="1.35", height="0.72"];'
            )

    for time_idx in time_indices:
        rank_nodes = " ".join(
            [scene_graph_header_id(node_prefix, time_idx)]
            + [
                scene_graph_node_id(node_prefix, concept_idx, time_idx)
                for concept_idx in concept_indices
                if (int(concept_idx), int(time_idx)) in visible_node_keys
            ]
        )
        lines.append(f"  {{ rank=same; {rank_nodes}; }}")

    for left, right in zip(time_indices, time_indices[1:]):
        lines.append(
            f"  {scene_graph_header_id(node_prefix, left)} -> {scene_graph_header_id(node_prefix, right)} "
            "[style=invis, weight=20];"
        )
    for concept_idx in concept_indices:
        for left, right in zip(time_indices, time_indices[1:]):
            if int(right) != int(left) + 1:
                continue
            if (int(concept_idx), int(left)) not in visible_node_keys or (int(concept_idx), int(right)) not in visible_node_keys:
                continue
            lines.append(
                f"  {scene_graph_node_id(node_prefix, concept_idx, left)} -> {scene_graph_node_id(node_prefix, concept_idx, right)} "
                '[color="#64748b", penwidth=1.4, arrowsize=0.55, weight=8];'
            )

    visible = set(concept_indices)
    for row in graph_rows:
        source_idx = int(row["source_idx"])
        target_idx = int(row["target_idx"])
        if source_idx not in visible or target_idx not in visible:
            continue
        weight = float(row["weight"])
        penwidth = scene_graph_edge_penwidth(weight)
        is_intervened_edge = bool(row.get("intervened", False))
        color = "#f59e0b" if is_intervened_edge else ("#86efac" if weight >= 0.0 else "#fca5a5")
        edge_label = f', label="{weight:+.2f}", fontcolor="#b45309"' if is_intervened_edge else ""
        kind = str(row.get("kind", "spatial"))
        if kind in {"cross_temporal", "temporal"}:
            for left, right in zip(time_indices, time_indices[1:]):
                if int(right) != int(left) + 1:
                    continue
                if kind == "temporal" and source_idx != target_idx:
                    continue
                if not scene_graph_edge_applies_at_time(row, right, current_time, target):
                    continue
                if (source_idx, int(left)) not in visible_node_keys or (target_idx, int(right)) not in visible_node_keys:
                    continue
                style = "solid" if kind == "temporal" else "dashed"
                lines.append(
                    f"  {scene_graph_node_id(node_prefix, source_idx, left)} -> {scene_graph_node_id(node_prefix, target_idx, right)} "
                    f'[constraint=false, color="{color}", style="{style}", penwidth="{penwidth:.2f}", arrowsize="0.50"{edge_label}];'
                )
        else:
            style = "solid" if str(row["branch"]) != "shared" else "dotted"
            for time_idx in time_indices:
                if not scene_graph_edge_applies_at_time(row, time_idx, current_time, target):
                    continue
                if (source_idx, int(time_idx)) not in visible_node_keys or (target_idx, int(time_idx)) not in visible_node_keys:
                    continue
                lines.append(
                    f"  {scene_graph_node_id(node_prefix, source_idx, time_idx)} -> {scene_graph_node_id(node_prefix, target_idx, time_idx)} "
                    f'[constraint=false, color="{color}", style="{style}", penwidth="{penwidth:.2f}", arrowsize="0.50"{edge_label}];'
                )

    return lines


def scene_graph_edge_applies_at_time(
    row: Mapping[str, object],
    time_idx: int,
    current_time: int | None,
    target: str,
) -> bool:
    branches = {value for value in str(row.get("branch", "")).split("+") if value}
    if not branches or "legacy" in branches or "shared" in branches:
        return True
    if current_time is not None and int(time_idx) > int(current_time):
        return "forecast" in branches
    return ("window" if str(target) == "activity" else "forecast") in branches


def scene_graph_edge_penwidth(weight: float) -> float:
    magnitude = abs(float(weight))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        return GRAPH_EDGE_MIN_PENWIDTH
    scaled = min(1.0, math.sqrt(magnitude / GRAPH_EDGE_REFERENCE_WEIGHT))
    return GRAPH_EDGE_MIN_PENWIDTH + scaled * (GRAPH_EDGE_MAX_PENWIDTH - GRAPH_EDGE_MIN_PENWIDTH)


def scene_graph_labelled_times(visible_node_keys: set[tuple[int, int]]) -> dict[int, int]:
    labelled: dict[int, int] = {}
    for concept_idx, time_idx in visible_node_keys:
        labelled[int(concept_idx)] = max(int(time_idx), labelled.get(int(concept_idx), int(time_idx)))
    return labelled


def scene_graph_node_colors(
    concept_idx: int,
    time_idx: int,
    fires: bool | None,
    intervention_node_keys: set[tuple[int, int]],
    driver_time_scores: Mapping[tuple[int, int], float],
    value: float | None,
    delta: float | None,
    is_probability: bool = False,
    is_forecast: bool = False,
    change_threshold: float = GRAPH_CHANGE_EPSILON,
) -> tuple[str, str]:
    del driver_time_scores
    if (int(concept_idx), int(time_idx)) in intervention_node_keys:
        return "#78350f", "#fbbf24"
    if is_forecast:
        if delta is not None and abs(float(delta)) >= float(change_threshold):
            return "#312e81", "#67e8f9"
        return ("#312e81", "#c4b5fd") if fires else ("#1e1b4b", "#8b5cf6")
    if delta is not None and abs(float(delta)) >= float(change_threshold):
        return "#164e63", "#67e8f9"
    if is_probability:
        if fires:
            return "#14532d", "#86efac"
        return "#1f2937", "#94a3b8"
    if value is not None and float(value) < -GRAPH_CHANGE_EPSILON:
        return "#7f1d1d", "#fca5a5"
    if value is not None and float(value) > GRAPH_CHANGE_EPSILON:
        return "#14532d", "#86efac"
    if fires:
        return "#14532d", "#86efac"
    return "#1f2937", "#94a3b8"


def scene_graph_activation(
    outputs: Mapping[str, object],
    instance: Mapping[str, Any],
    time_idx: int,
    concept_idx: int,
    target: str,
) -> tuple[bool | None, float | None]:
    forecast_step = scene_graph_forecast_step(instance, time_idx)
    if forecast_step is not None:
        value = scene_graph_rollout_concept_value(
            outputs,
            "predicted_concepts_by_step",
            forecast_step,
            concept_idx,
        )
        if value is None:
            return None, None
        is_probability = str(outputs.get("st_state_activation")) == "bounded_logit"
        return (value >= 0.5 if is_probability else value > 0.0), value

    tensor, is_calibrated = scene_graph_activation_tensor(outputs, target)
    if hasattr(tensor, "detach"):
        try:
            value = float(tensor[0, int(time_idx), int(concept_idx)].detach().cpu().item())
            return (value >= 0.5 if is_calibrated else value > 0.0), value
        except (IndexError, RuntimeError, TypeError, ValueError):
            pass
    try:
        value = float(instance["concepts"][int(time_idx)][int(concept_idx)])
    except (IndexError, KeyError, TypeError, ValueError):
        return None, None
    return value > 0.0, value


def scene_graph_pre_graph_value(
    outputs: Mapping[str, object],
    instance: Mapping[str, Any],
    time_idx: int,
    concept_idx: int,
) -> float | None:
    forecast_step = scene_graph_forecast_step(instance, time_idx)
    if forecast_step is not None:
        if forecast_step > 1:
            return scene_graph_rollout_concept_value(
                outputs,
                "predicted_concepts_by_step",
                forecast_step - 1,
                concept_idx,
            )
        tensor, _ = scene_graph_activation_tensor(outputs, "forecast")
        if hasattr(tensor, "detach"):
            try:
                return float(tensor[0, -1, int(concept_idx)].detach().cpu().item())
            except (IndexError, RuntimeError, TypeError, ValueError):
                return None
        return None

    for key in ("temporalized_concepts", "calibrated_concepts"):
        tensor = outputs.get(key)
        if hasattr(tensor, "detach"):
            try:
                return float(tensor[0, int(time_idx), int(concept_idx)].detach().cpu().item())
            except (IndexError, RuntimeError, TypeError, ValueError):
                pass
    try:
        return float(instance["concepts"][int(time_idx)][int(concept_idx)])
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def scene_graph_activation_is_probability(outputs: Mapping[str, object], target: str) -> bool:
    _, is_probability = scene_graph_activation_tensor(outputs, target)
    return bool(is_probability)


def scene_graph_activation_tensor(outputs: Mapping[str, object], target: str) -> tuple[object, bool]:
    bounded_graph_states = str(outputs.get("st_state_activation")) == "bounded_logit"
    if str(target) == "activity":
        for key in ("window_refined_concepts", "shared_refined_concepts", "concept_states", "calibrated_concepts"):
            tensor = outputs.get(key)
            if hasattr(tensor, "detach"):
                return tensor, key == "calibrated_concepts" or (
                    bounded_graph_states and key in {"window_refined_concepts", "shared_refined_concepts", "concept_states"}
                )
    else:
        for key in ("forecast_refined_concepts", "shared_refined_concepts", "concept_states", "calibrated_concepts"):
            tensor = outputs.get(key)
            if hasattr(tensor, "detach"):
                return tensor, key == "calibrated_concepts" or (
                    bounded_graph_states and key in {"forecast_refined_concepts", "shared_refined_concepts", "concept_states"}
                )
    return None, False


def scene_graph_node_label(
    concept_name: str,
    pre_value: float | None,
    value: float | None,
    show_name: bool,
    delta: float | None,
    change_threshold: float = GRAPH_CHANGE_EPSILON,
) -> str:
    concept_name = "\\n".join(textwrap.wrap(str(concept_name), width=18, break_long_words=True))
    delta_label = ""
    if delta is not None and abs(float(delta)) >= float(change_threshold):
        delta_label = f"\\nchange {float(delta):+.3f}"
    if value is None and pre_value is None:
        return "." if not show_name else concept_name
    if pre_value is not None and value is not None:
        value_label = f"{pre_value:.2f} -> {value:.2f}"
    elif value is not None:
        value_label = f"{value:.2f}"
    else:
        value_label = f"{pre_value:.2f} -> ?"
    if not show_name:
        return f"{value_label}{delta_label}"
    if value is None and pre_value is None:
        return concept_name
    return f"{concept_name}\\n{value_label}{delta_label}"


def scene_graph_time_header_html(instance: Mapping[str, Any], time_idx: int, current_time: int | None) -> str:
    original = scene_graph_original_timestep(instance, time_idx)
    if current_time is None:
        relative = f"window {time_idx + 1}"
    else:
        delta = int(time_idx) - int(current_time)
        if delta == 0:
            relative = "Current"
        else:
            sign = "+" if delta > 0 else "-"
            prefix = "Forecast " if delta > 0 else ""
            relative = f"{prefix}t {sign} {abs(delta)}"
    is_forecast = current_time is not None and int(time_idx) > int(current_time)
    color = "#c4b5fd" if is_forecast else "#f8fafc"
    rows = [
        f'<TR><TD><FONT POINT-SIZE="11" COLOR="{color}"><B>{html_escape(relative)}</B></FONT></TD></TR>'
    ]
    if original is not None:
        rows.append(
            f'<TR><TD><FONT POINT-SIZE="9" COLOR="#94a3b8">frame {int(original)}</FONT></TD></TR>'
        )
    return '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1">' + "".join(rows) + "</TABLE>"


def scene_graph_original_timestep(instance: Mapping[str, Any], history_t: int) -> int | None:
    try:
        if bool(instance["key_padding_mask"][int(history_t)]):
            return None
        return int(instance["history_start"]) + int(history_t) - int(instance["history_offset"])
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def valid_history_time_indices(instance: Mapping[str, Any]) -> list[int]:
    return [
        int(index)
        for index, is_pad in enumerate(instance.get("key_padding_mask", []))
        if not bool(is_pad)
    ]


def graph_time_option_label(instance: Mapping[str, Any], history_t: int) -> str:
    valid = valid_history_time_indices(instance)
    current = max(valid) if valid else int(history_t)
    delta = int(history_t) - int(current)
    relative = "t" if delta == 0 else f"t{delta:+d}"
    original = scene_graph_original_timestep(instance, int(history_t))
    return relative if original is None else f"{relative} (frame {original})"


def scene_graph_header_id(node_prefix: str, time_idx: int) -> str:
    return f"{dot_id_prefix(node_prefix)}_h{int(time_idx)}"


def scene_graph_node_id(node_prefix: str, concept_idx: int, time_idx: int) -> str:
    return f"{dot_id_prefix(node_prefix)}_n{int(concept_idx)}_t{int(time_idx)}"


def dot_id_prefix(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(value)) or "g"


def render_scene_graph_inspector(
    workspace,
    graph_rows: list[Mapping[str, object]],
    driver_rows: list[Mapping[str, Any]],
    intervention_items: list[Mapping[str, object]],
) -> None:
    visible_indices = sorted(graph_concept_candidates(graph_rows, set()), key=lambda idx: graph_concept_label(workspace, idx).lower())
    if not visible_indices:
        return
    selected = st.selectbox(
        "Inspect visible concept",
        visible_indices,
        format_func=lambda idx: graph_concept_label(workspace, int(idx)),
        key=GRAPH_INSPECT_CONCEPT_KEY,
    )
    selected_idx = int(selected)
    time_rows = scene_graph_concept_time_rows(selected_idx, driver_rows, intervention_items)
    touch_rows = [
        {
            "branch": row["branch"],
            "from": row["source"],
            "to": row["target"],
            "weight": f"{float(row['weight']):+.3f}",
        }
        for row in graph_rows
        if int(row["source_idx"]) == selected_idx or int(row["target_idx"]) == selected_idx
    ]
    col_time, col_edges = st.columns([0.42, 0.58])
    with col_time:
        st.caption("When this concept is active in the selected scene")
        if time_rows:
            st.dataframe(pd.DataFrame(time_rows), width="stretch", hide_index=True)
        else:
            st.caption("This visible neighbor is learned by the graph, but is not one of the current top driver/intervention concepts.")
    with col_edges:
        st.caption("Learned edges touching the selected concept")
        st.dataframe(pd.DataFrame(touch_rows), width="stretch", hide_index=True)


def scene_graph_concept_time_rows(
    concept_idx: int,
    driver_rows: list[Mapping[str, Any]],
    intervention_items: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in driver_rows:
        if int(row.get("concept_idx", -1)) != int(concept_idx):
            continue
        rows.append(
            {
                "t": row.get("original_t"),
                "fires": binary_label(row.get("soft_binary")),
                "activation": format_optional(row.get("standardized_input")),
                "contribution": format_optional(row.get("contribution")),
            }
        )
    for item in intervention_items:
        if int(item.get("concept_idx", -1)) != int(concept_idx):
            continue
        rows.append(
            {
                "t": item.get("original_t"),
                "fires": item.get("setting", "intervention"),
                "activation": format_optional(item.get("value")),
                "contribution": "intervention",
            }
        )
    return rows


def render_intervention_prediction_delta(
    workspace,
    baseline_outputs: Mapping[str, object],
    outputs: Mapping[str, object],
    target: str,
    class_idx: int,
    horizon: int | None,
) -> None:
    before_options = prediction_option_rows(workspace, baseline_outputs, target, horizon=horizon, top_k=1)
    after_options = prediction_option_rows(workspace, outputs, target, horizon=horizon, top_k=1)
    before_selected = selected_class_probability(workspace, baseline_outputs, target, class_idx, horizon)
    after_selected = selected_class_probability(workspace, outputs, target, class_idx, horizon)
    rows = []
    if before_options:
        rows.append({
            "state": "before",
            "predicted_activity": before_options[0]["class"],
            "predicted_probability": f"{float(before_options[0]['probability']):.3f}",
            "selected_label_probability": format_probability(before_selected),
        })
    if after_options:
        rows.append({
            "state": "after",
            "predicted_activity": after_options[0]["class"],
            "predicted_probability": f"{float(after_options[0]['probability']):.3f}",
            "selected_label_probability": format_probability(after_selected),
        })
    if rows:
        st.caption("Prediction before / after intervention")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if before_selected is not None and after_selected is not None:
        st.metric(
            "Selected-label probability after intervention",
            f"{after_selected:.3f}",
            delta=f"{after_selected - before_selected:+.3f}",
        )


def scene_graph_dot(
    rows: list[Mapping[str, object]],
    driver_indices: set[int],
    intervention_indices: set[int],
    time_labels: Mapping[int, list[str]],
    driver_scores: Mapping[int, float],
    edge_label_mode: str,
) -> str:
    node_labels: dict[int, str] = {}
    for row in rows:
        node_labels[int(row["source_idx"])] = str(row["source"])
        node_labels[int(row["target_idx"])] = str(row["target"])

    lines = [
        "digraph CurrentSceneGraph {",
        "  graph [rankdir=LR, bgcolor=\"transparent\", pad=\"0.12\", nodesep=\"0.32\", ranksep=\"0.48\", margin=\"0.02\", splines=true];",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=15, fontcolor=\"#f8fafc\", color=\"#94a3b8\", penwidth=1.8, margin=\"0.13,0.08\"];",
        "  edge [fontname=\"Helvetica\", fontsize=12, fontcolor=\"#f8fafc\", arrowsize=0.8, penwidth=2.0, color=\"#cbd5e1\"];",
    ]
    for node_idx, label in sorted(node_labels.items(), key=lambda item: item[1].lower()):
        if node_idx in intervention_indices:
            fill = "#78350f"
            color = "#fbbf24"
        elif node_idx in driver_indices:
            score = float(driver_scores.get(node_idx, 0.0))
            fill = "#14532d" if score >= 0.0 else "#7f1d1d"
            color = "#86efac" if score >= 0.0 else "#fca5a5"
        else:
            fill = "#1f2937"
            color = "#94a3b8"
        node_label = label
        if node_idx in time_labels:
            node_label = f"{label}\\n[{', '.join(time_labels[node_idx])}]"
        lines.append(
            f'  n{node_idx} [label="{dot_escape(node_label)}", fillcolor="{fill}", color="{color}"];'
        )

    for row in rows:
        weight = float(row["weight"])
        color = "#86efac" if weight >= 0 else "#fca5a5"
        style = "solid" if abs(weight) >= 0.05 else "dashed"
        if edge_label_mode == "off":
            label = ""
        elif edge_label_mode == "branch":
            label = str(row["branch"])
        else:
            label = f"{row['branch']} {weight:+.2f}"
        lines.append(
            f'  n{int(row["source_idx"])} -> n{int(row["target_idx"])} '
            f'[label="{dot_escape(label)}", color="{color}", fontcolor="#f8fafc", style="{style}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def dot_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def dot_label_escape(value: str) -> str:
    return "\\n".join(dot_escape(part) for part in str(value).split("\\n"))


def intervention_contains(intervention: Mapping[str, object] | None, row: Mapping[str, Any]) -> bool:
    if intervention is None:
        return False
    items = intervention.get("items") if isinstance(intervention, Mapping) else None
    if items is None:
        items = [intervention]
    for item in items:
        if (
            int(item.get("time_idx", -1)) == int(row["history_t"])
            and int(item.get("concept_idx", -1)) == int(row["concept_idx"])
        ):
            return True
    return False


def format_optional(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.3f}"
    except (TypeError, ValueError):
        return str(value)


def format_probability(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def format_probability_delta(before: object, after: object) -> str:
    if before is None or after is None:
        return "n/a"
    return f"{float(after) - float(before):+.3f}"


def binary_label(value: object) -> str:
    if value is None:
        return "n/a"
    return "true" if bool(value) else "false"


def video_source_info(
    path: Path,
    num_windows: int,
    *,
    window_spans: object = None,
    video_meta: object = None,
) -> dict[str, Any]:
    meta = dict(video_meta) if isinstance(video_meta, Mapping) else {}
    if path.is_dir():
        frame_count = len(image_frames(path))
        fps = float(meta.get("fps", 0.0) or 0.0) or 12.0
        kind = "frame_folder"
    elif path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        kind = "video_file"
    else:
        fps = float(meta.get("fps", 0.0) or 0.0) or 1.0
        frame_count = int(meta.get("frame_count", 0.0) or 0)
        kind = "unknown"
    duration = frame_count / fps if frame_count > 0 else float(num_windows)
    duration = max(float(duration), 0.01)
    spans = validated_window_spans(window_spans, int(num_windows), duration)
    if spans:
        duration = max(duration, float(spans[-1][1]))
    return {
        "kind": kind,
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": duration,
        "num_windows": int(num_windows),
        "window_spans": spans,
    }


def validated_window_spans(
    raw_spans: object,
    num_windows: int,
    duration_seconds: float,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw_spans, (list, tuple)) or len(raw_spans) < int(num_windows):
        return ()
    duration = max(float(duration_seconds), 0.01)
    spans: list[tuple[float, float]] = []
    previous_start = -math.inf
    for raw in raw_spans[: int(num_windows)]:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            return ()
        try:
            start = float(raw[0])
            end = float(raw[1])
        except (TypeError, ValueError):
            return ()
        if not math.isfinite(start) or not math.isfinite(end) or start < previous_start or end < start:
            return ()
        start = min(duration, max(0.0, start))
        end = min(duration, max(start, end))
        spans.append((start, end))
        previous_start = start
    return tuple(spans)


def position_to_timestep(
    position_seconds: float,
    duration_seconds: float,
    num_windows: int,
    window_spans: object = None,
) -> int:
    num_windows = max(1, int(num_windows))
    duration_seconds = max(float(duration_seconds), 1e-6)
    position = min(duration_seconds, max(0.0, float(position_seconds)))
    if isinstance(window_spans, (list, tuple)) and len(window_spans) >= num_windows:
        timestep = 0
        for index, span in enumerate(window_spans[:num_windows]):
            if float(span[0]) > position:
                break
            timestep = index
        return min(num_windows - 1, max(0, timestep))
    ratio = min(0.999999, max(0.0, position / duration_seconds))
    return min(num_windows - 1, max(0, int(ratio * num_windows)))


def timestep_to_position_seconds(
    timestep: int,
    duration_seconds: float,
    num_windows: int,
    window_spans: object = None,
) -> float:
    num_windows = max(1, int(num_windows))
    duration_seconds = max(float(duration_seconds), 0.01)
    timestep = min(num_windows - 1, max(0, int(timestep)))
    if isinstance(window_spans, (list, tuple)) and len(window_spans) >= num_windows:
        return min(duration_seconds, max(0.0, float(window_spans[timestep][0])))
    return min(duration_seconds, max(0.0, timestep * duration_seconds / float(num_windows)))


def source_position_to_timestep(position_seconds: float, source: Mapping[str, Any]) -> int:
    return position_to_timestep(
        position_seconds,
        source["duration_seconds"],
        source["num_windows"],
        source.get("window_spans"),
    )


def snap_position_to_timestep(position_seconds: float, source: Mapping[str, Any]) -> float:
    timestep = source_position_to_timestep(position_seconds, source)
    return timestep_to_position_seconds(
        timestep,
        source["duration_seconds"],
        source["num_windows"],
        source.get("window_spans"),
    )


def smooth_buffer_start_seconds(position_seconds: float, duration_seconds: float) -> float:
    duration_seconds = max(float(duration_seconds), 0.0)
    position_seconds = min(duration_seconds, max(0.0, float(position_seconds)))
    prebuffer_seconds = min(BUFFER_REFRESH_MARGIN_SECONDS, PLAYBACK_BUFFER_SECONDS / 3.0)
    start = max(0.0, position_seconds - prebuffer_seconds)
    return min(start, max(0.0, duration_seconds - PLAYBACK_BUFFER_SECONDS))


def image_frames(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [Path(frame_path) for frame_path in cached_image_frame_paths(str(path))]


def timestep_frame_range(
    frame_count: int,
    num_windows: int,
    timestep: int,
    *,
    window_spans: object = None,
    fps: float = 1.0,
) -> tuple[int, int]:
    frame_count = max(1, int(frame_count))
    num_windows = max(1, int(num_windows))
    timestep = min(max(0, int(timestep)), num_windows - 1)
    if isinstance(window_spans, (list, tuple)) and len(window_spans) >= num_windows:
        start = int(round(float(window_spans[timestep][0]) * max(float(fps), 1e-6)))
        end = int(round(float(window_spans[timestep][1]) * max(float(fps), 1e-6)))
    else:
        start = int(round(timestep * frame_count / num_windows))
        end = int(round((timestep + 1) * frame_count / num_windows)) - 1
    start = min(frame_count - 1, max(0, start))
    end = min(frame_count - 1, max(start, end))
    return start, end


def label_name(workspace, index: object) -> str | None:
    if index is None:
        return None
    idx = int(index)
    if 0 <= idx < len(workspace.activity_names):
        return workspace.activity_names[idx]
    return str(idx)


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remaining = seconds - 60 * minutes
    return f"{minutes:d}:{remaining:04.1f}"


if __name__ == "__main__":
    main()
