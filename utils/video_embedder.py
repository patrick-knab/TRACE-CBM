from __future__ import annotations

import os
import random
import gc
import numpy as np
import torch
import clip
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
)
from typing import List, Tuple, Optional, Dict
import cv2
from PIL import Image
import tqdm
import time


def init_repro(seed: int = 42, deterministic: bool = True):
    """Call this at the very top of your notebook/script BEFORE creating any model/processor/device context."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":16:8"  # deterministic cuBLAS on Ampere+, nice default
    )
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism knobs (do this before any CUDA ops)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # older torch may not support signature
            torch.set_deterministic(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Reduce threading nondeterminism
    torch.set_num_threads(1)

    return seed


def _cpu_tensor_or_none(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


class _PickleBackendsMixin:
    def attach_backends(
        self, *, model=None, tokenizer=None, clip_model=None, clip_tokenizer=None, device=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if getattr(self, "model", None) is not None:
            self.model = self.model.to(self.device).eval()
            if self.device.type == "cuda" and getattr(self, "model_name", "") in {
                "pe-l14",
                "pe-g14",
            }:
                self.model = self.model.half()
            elif self.device.type != "cuda":
                # Half precision kernels are not consistently supported on CPU.
                self.model = self.model.float()

    def __getstate__(self):
        s = self.__dict__.copy()
        # drop unpicklables
        for k in ("model", "tokenizer", "clip_model", "clip_tokenizer", "device"):
            s.pop(k, None)
        # ensure tensors are CPU-picklable
        for k in ("video_embeddings", "text_embeddings"):
            if k in s and s[k] is not None:
                if isinstance(s[k], dict):
                    s[k] = {kk: _cpu_tensor_or_none(vv) for kk, vv in s[k].items()}
                else:
                    s[k] = _cpu_tensor_or_none(s[k])
        return s

    def __setstate__(self, s):
        self.__dict__.update(s)
        # backends are reattached by caller after unpickle
        self.model = None
        self.tokenizer = None
        self.clip_model = None
        self.clip_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VideoEmbedder(_PickleBackendsMixin):
    def __init__(self, model_name, model, tokenizer, clip_model=None,
                 pe_video_batch_size: Optional[int] = None, pe_target_T: Optional[int] = None,
                 device=None):
        self.model_name = model_name.lower()
        self.model = model
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.pe_video_batch_size = pe_video_batch_size
        self.pe_target_T = pe_target_T

        self.dataset_name: Optional[str] = None
        self.video_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.labels: Optional[List[str]] = None
        self.video_window_spans: Dict[str, List[Tuple[float, float]]] = {}
        self.video_meta: Dict[str, Dict[str, float]] = {}

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(self.device).eval()
        if self.device.type == "cuda" and self.model_name in {"pe-l14", "pe-g14"}:
            self.model = self.model.half()
        elif self.device.type != "cuda":
            # Keep CPU inference in fp32 to avoid layernorm dtype errors with fp16 weights.
            self.model = self.model.float()
        torch.backends.cudnn.benchmark = True
        self.embed_dim = self._detect_embed_dim()

    # ------------------------- util
    @staticmethod
    def _extract_hf_image_feats(out) -> torch.Tensor:
        """
        Normalize HF model outputs to a single (B, D) tensor.

        Different Transformers versions/models return different output types:
        - CLIPModel: CLIPOutput with .image_embeds
        - SigLIP: BaseModelOutputWithPooling with .pooler_output (or model-specific embeds)
        - Some models may directly return a tensor/dict
        """
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict):
            if "image_embeds" in out and isinstance(out["image_embeds"], torch.Tensor):
                return out["image_embeds"]
            if "pooler_output" in out and isinstance(out["pooler_output"], torch.Tensor):
                return out["pooler_output"]
        if hasattr(out, "image_embeds") and isinstance(getattr(out, "image_embeds"), torch.Tensor):
            return out.image_embeds
        if hasattr(out, "pooler_output") and isinstance(getattr(out, "pooler_output"), torch.Tensor):
            return out.pooler_output
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            return out[0]
        raise TypeError(f"Unsupported HF output type for image feats: {type(out)}")

    def _hf_get_image_features(self, batch: dict) -> torch.Tensor:
        """
        Return a (B, D) image embedding tensor for HF CLIP/SigLIP style models.

        We prefer get_image_features() because some Transformers CLIPModel
        versions require input_ids in forward(), even for image-only calls.
        """
        if not hasattr(self.model, "get_image_features"):
            out = self.model(**batch)
            return self._extract_hf_image_feats(out)

        out = self.model.get_image_features(**batch)
        if isinstance(out, torch.Tensor):
            return out

        # Some versions return vision outputs (e.g., BaseModelOutputWithPooling).
        if hasattr(out, "pooler_output") and isinstance(getattr(out, "pooler_output"), torch.Tensor):
            pooled = out.pooler_output
            if hasattr(self.model, "visual_projection"):
                proj = getattr(self.model, "visual_projection")
                if isinstance(proj, torch.nn.Module):
                    # Only project if dimensions match; some variants already return projected features.
                    in_features = getattr(proj, "in_features", None)
                    out_features = getattr(proj, "out_features", None)
                    d = pooled.shape[-1]
                    if in_features is None:
                        return proj(pooled)
                    if d == in_features:
                        return proj(pooled)
                    if out_features is not None and d == out_features:
                        return pooled
                    return pooled
            return pooled

        return self._extract_hf_image_feats(out)

    @staticmethod
    def _normalize_video_feature_output(out) -> torch.Tensor:
        """Normalize model output object to a tensor of shape (B, D) or (B, T, D)."""
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict):
            for key in ("last_hidden_state", "pooler_output", "logits"):
                val = out.get(key, None)
                if isinstance(val, torch.Tensor):
                    return val
        if hasattr(out, "last_hidden_state") and isinstance(out.last_hidden_state, torch.Tensor):
            return out.last_hidden_state
        if hasattr(out, "pooler_output") and isinstance(out.pooler_output, torch.Tensor):
            return out.pooler_output
        if hasattr(out, "logits") and isinstance(out.logits, torch.Tensor):
            return out.logits
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            return out[0]
        raise TypeError(f"Unsupported video feature output type: {type(out)}")

    def _detect_embed_dim(self) -> int:
        def _cfg_get(root, *keys):
            cur = root
            for key in keys:
                if cur is None:
                    return None
                if isinstance(cur, dict):
                    cur = cur.get(key, None)
                else:
                    cur = getattr(cur, key, None)
            return cur

        with torch.inference_mode():
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.model_name in {"res50", "b32", "b16", "l14"} and hasattr(
                self.model, "encode_image"
            ):
                t = self.tokenizer(Image.fromarray(dummy)).unsqueeze(0).to(self.device)
                d = self.model.encode_image(t).shape[-1]
            elif self.model_name in {"clip", "b32", "b16", "l14", "siglip", "siglipl14", "siglip2"}:
                batch = self.tokenizer(images=dummy, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                feats = self._hf_get_image_features(batch)
                d = feats.shape[-1]
            elif self.model_name == "clip4clip":
                preprocess = Compose(
                    [
                        Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize(
                            (0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711),
                        ),
                    ]
                )
                t = preprocess(Image.fromarray(dummy)).unsqueeze(0).to(self.device)
                d = self.model(t)["image_embeds"].shape[-1]
            elif self.model_name == "pe-l14":
                target_t = int(self.pe_target_T) if self.pe_target_T is not None else 16
                dummy_img = Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))
                frame = self.tokenizer(dummy_img)
                clip = torch.stack([frame for _ in range(target_t)], dim=0)  # (T,C,H,W)
                clip = clip.unsqueeze(0).to(self.device)  # (B,T,C,H,W)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(self.device.type == "cuda"),
                ):
                    d = self.model.encode_video(clip).shape[-1]
            elif self.model_name == "pe-g14":
                target_t = int(self.pe_target_T) if self.pe_target_T is not None else 16
                dummy_img = Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))
                frame = self.tokenizer(dummy_img)
                clip = torch.stack([frame for _ in range(target_t)], dim=0)  # (T,C,H,W)
                clip = clip.unsqueeze(0).to(self.device)  # (B,T,C,H,W)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(self.device.type == "cuda"),
                ):
                    d = self.model.encode_video(clip).shape[-1]
            elif self.model_name == "internvideo2-clip-s":
                # Prefer config-derived dimension to avoid expensive/fragile probe forward.
                align_dim = _cfg_get(self.model.config, "model", "vision_encoder", "align_dim")
                if align_dim is not None:
                    d = int(align_dim)
                else:
                    # Fallback probe if config structure changes.
                    target_t = int(getattr(self.model.config, "num_frames", 8))
                    dummy_clip = torch.zeros((target_t, 3, 224, 224), dtype=torch.uint8)
                    clip = self.tokenizer(dummy_clip) if callable(self.tokenizer) else dummy_clip.float().div(255.0)
                    if not torch.is_tensor(clip):
                        raise TypeError(
                            f"InternVideo2 transform returned unsupported type: {type(clip)}"
                        )
                    clip = clip.unsqueeze(0).to(self.device)  # (B,T,C,H,W)
                    try:
                        feats = self.model.encode_vision(clip, test=True)
                    except TypeError:
                        feats = self.model.encode_vision(clip)
                    feats = self._normalize_video_feature_output(feats)
                    if feats.dim() >= 3:
                        feats = feats.mean(dim=1)
                    d = feats.shape[-1]
            else:
                raise ValueError(f"Unknown model_name {self.model_name}")
        return int(d)

    def _preprocess_video_pe(
            self,
            video: List[Image.Image],  # now expects a list of PIL Images
            num_frames: int = 4,
            transform: Optional[Compose] = None,
            return_first_frame_for_demo: bool = False
        ) -> Tuple[torch.Tensor, Optional[Image.Image]]:
        total_frames = len(video)
        # Uniformly sample frame indices
        frame_indices = [int(i * (total_frames / num_frames)) for i in range(num_frames)]
        frames = [video[i] for i in frame_indices]
        # Preprocess frames
        preprocessed_frames = [transform(frame) for frame in frames]

        first_frame = None
        if return_first_frame_for_demo:
            first_frame = frames[0]
        return torch.stack(preprocessed_frames, dim=0), first_frame
    @staticmethod
    def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _sample_indices(n: int, k: int, random: bool) -> List[int]:
        if n <= 0 or k <= 0:
            return []
        if random:
            k = min(k, n)
            return np.random.choice(n, size=k, replace=False).tolist()
        if k >= n:
            return list(range(n))
        step = (n - 1) / (k - 1) if k > 1 else 1e9
        return [int(round(i * step)) for i in range(k)]

    # ------------------------- encoders
    def _encode_images_hf(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """HF CLIP, SigLIP"""
        if not frames_bgr:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)
        images_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        batch = self.tokenizer(images=images_rgb, return_tensors="pt")
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(self.device.type == "cuda"),
        ):
            feats = self._hf_get_image_features(batch)
        return feats.float().detach().cpu()

    def _encode_images_pe(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """Encode frames using PE-L/14 video model.

        This implementation preserves alignment: it returns one embedding per
        input frame by forming a temporal clip centered on each anchor frame
        (padding at the edges). Clips are length T (default 16) and are
        batched efficiently on GPU.
        """

        if not frames_bgr:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)

        # Config
        clip_len = 16  # temporal length expected by the video model
        half_clip = clip_len // 2
        max_gpu_batch = 8  # number of clips to encode per forward pass

        # Preprocess all frames once (C,H,W tensors on CPU)
        pil_imgs = [self._bgr_to_pil(f) for f in frames_bgr]
        frames_tensor = [self.tokenizer(img) for img in pil_imgs]
        n = len(frames_tensor)

        def build_clip_around(idx: int) -> torch.Tensor:
            """Return a tensor of shape (T, C, H, W) for anchor frame idx."""
            start = idx - half_clip
            end = start + clip_len
            # Clamp and pad by edge repetition
            frames = []
            for t in range(start, end):
                clamped = min(max(t, 0), n - 1)
                frames.append(frames_tensor[clamped])
            return torch.stack(frames, dim=0)

        embs = []
        with torch.inference_mode():
            # Iterate in micro-batches of clips to control memory
            for s in range(0, n, max_gpu_batch):
                batch_indices = list(range(s, min(s + max_gpu_batch, n)))
                clips = [build_clip_around(i) for i in batch_indices]
                x = torch.stack(clips, dim=0).to(self.device, non_blocking=True)
                # x: (B, T, C, H, W)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(self.device.type == "cuda"),
                ):
                    feats = self.model.encode_video(x)
                embs.append(feats.detach().cpu())

        return torch.cat(embs, dim=0).float()

    def _encode_images_openai_clip(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """OpenAI CLIP RN50."""
        if not frames_bgr:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)
        pil_imgs = [self._bgr_to_pil(f) for f in frames_bgr]
        x = torch.stack([self.tokenizer(img) for img in pil_imgs], dim=0).to(
            self.device, non_blocking=True
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(self.device.type == "cuda"),
        ):
            feats = self.model.encode_image(x)
        return feats.float().detach().cpu()

    def _encode_images_clip4clip(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """CLIP4Clip (expects raw pixel tensors normalized manually)."""
        if not frames_bgr:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)
        preprocess = Compose(
            [
                Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        pil_imgs = [self._bgr_to_pil(f) for f in frames_bgr]
        x = torch.stack([preprocess(img) for img in pil_imgs], dim=0).to(
            self.device, non_blocking=True
        )
        with torch.inference_mode():
            out = self.model(x)["image_embeds"]
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-6)
        return out.float().detach().cpu()

    # ------------------------- video reading
    def _read_windows(self, video_path: str, window_size: int, stride: Optional[int] = None):
        windows, spans = [], []
        
        # If stride is None or 0, use window_size (non-overlapping windows)
        step = stride if stride is not None and stride > 0 else window_size

        if video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            # ---- read video ----
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0

            if window_size > frame_count:
                frame_count = window_size

            frames = []
            for _ in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

        else:
            # ---- read raw images in directory ----
            img_files = sorted(
                [
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            frame_count = len(img_files)
            fps = 12.0  # fallback since no video container
            for i in range(0, frame_count, step):
                chunk_files = img_files[i : i + window_size]
                if self.model_name in {"pe-l14", "pe-g14", "internvideo2-clip-s"}:
                    target_t = int(self.pe_target_T) if self.pe_target_T is not None else min(window_size, 16)
                    sample_indices = self._sample_indices(len(chunk_files), target_t, random=False)
                else:
                    sample_indices = [len(chunk_files) // 2] if chunk_files else []
                chunk = [cv2.imread(chunk_files[index]) for index in sample_indices]
                chunk = [frame for frame in chunk if frame is not None]
                if len(chunk) == 0:
                    continue
                windows.append(chunk)
                start_t = i / fps
                end_t = (i + len(chunk) - 1) / fps
                spans.append((start_t, end_t))
            return windows, spans, fps, frame_count

        # ---- make windows ----
        for i in range(0, frame_count, step):
            chunk = frames[i : i + window_size]
            if len(chunk) == 0:
                continue
            windows.append(chunk)
            start_t = i / fps
            end_t = (i + len(chunk) - 1) / fps
            spans.append((start_t, end_t))

        return windows, spans, fps, frame_count

    def _encode_windows(
        self, windows, frames_per_window, random, batch_size
    ) -> np.ndarray:
        # Specialized path for InternVideo2 CLIP-S: encode whole windows as clips.
        if self.model_name == "internvideo2-clip-s":
            video_bs = self.pe_video_batch_size or min(batch_size, 8)
            outs = []
            for s in range(0, len(windows), video_bs):
                batch_windows = windows[s : s + video_bs]
                feats = self._encode_windows_internvideo2_clip_s(batch_windows)
                outs.append(feats)
            if len(outs) == 0:
                return np.zeros((0, self.embed_dim), dtype=np.float32)
            return torch.cat(outs, dim=0).numpy()
        # Specialized path for PE-L/14: encode entire windows as video clips
        if self.model_name == "pe-l14" or self.model_name == "pe-g14":
            # Use configured per-GPU batch size for video clips if provided
            pe_bs = self.pe_video_batch_size or min(batch_size, 8)
            outs = []
            for s in range(0, len(windows), pe_bs):
                batch_windows = windows[s : s + pe_bs]
                feats = self._encode_windows_pe(batch_windows)
                outs.append(feats)
            if len(outs) == 0:
                return np.zeros((0, self.embed_dim), dtype=np.float32)
            return torch.cat(outs, dim=0).numpy()

        # Image encoders: sample frames and average per window
        all_samples = []
        map_win_to_slice = []
        cursor = 0
        for w in windows:
            idxs = self._sample_indices(len(w), frames_per_window, random)
            if not idxs:
                all_samples.append(w[0])
                map_win_to_slice.append((cursor, cursor + 1))
                cursor += 1
                continue
            for j in idxs:
                all_samples.append(w[j])
            map_win_to_slice.append((cursor, cursor + len(idxs)))
            cursor += len(idxs)

        if self.model_name in {"res50", "b32", "b16", "l14"} and hasattr(
            self.model, "encode_image"
        ):
            encode_fn = self._encode_images_openai_clip
        elif self.model_name in {"clip", "b32", "b16", "l14", "siglip", "siglipl14", "siglip2"}:
            encode_fn = self._encode_images_hf
        elif self.model_name == "clip4clip":
            encode_fn = self._encode_images_clip4clip
        else:
            raise ValueError(f"Unknown model_name {self.model_name}")

        outs = []
        for s in range(0, len(all_samples), batch_size):
            feats = encode_fn(all_samples[s : s + batch_size])
            outs.append(feats)

        if len(outs) == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        flat_feats = torch.cat(outs, dim=0)
        window_embs = []
        for a, b in map_win_to_slice:
            if b <= a:
                window_embs.append(torch.zeros(self.embed_dim))
            else:
                window_embs.append(flat_feats[a:b].mean(dim=0))
        return torch.stack(window_embs, dim=0).numpy()

    def _encode_windows_pe(self, windows: List[List[np.ndarray]]) -> torch.Tensor:
        """Encode a batch of windows (lists of BGR frames) as video clips.

        Pads each window in the batch to the batch's max temporal length by
        repeating the last frame so windows can be batched.
        Returns a CPU tensor of shape (B, D).
        """
        if not windows:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)

        # Preprocess each window: convert to tensors (T_i, C, H, W)
        clip_tensors = []
        max_T = 0
        for w in windows:
            if not w:
                # create a single black frame if window is empty
                black = np.zeros((336, 336, 3), dtype=np.uint8)
                w = [black]
            # If requested, uniformly sample to target temporal length
            if self.pe_target_T is not None and len(w) > 0:
                T = self.pe_target_T
                if len(w) >= T:
                    # uniform indices across [0, len(w)-1]
                    idxs = [int(round(i * (len(w) - 1) / (T - 1))) for i in range(T)]
                else:
                    # upsample by repeating last frame to reach T
                    idxs = list(range(len(w))) + [len(w) - 1] * (T - len(w))
                w = [w[i] for i in idxs]

            pil_imgs = [self._bgr_to_pil(f) for f in w]
            frames = [self.tokenizer(img) for img in pil_imgs]
            clip = torch.stack(frames, dim=0)
            clip_tensors.append(clip)
            max_T = max(max_T, clip.shape[0])

        # Pad all to max_T using last-frame repetition
        padded = []
        for clip in clip_tensors:
            if clip.shape[0] < max_T:
                pad = clip[-1:].expand(max_T - clip.shape[0], -1, -1, -1)
                clip = torch.cat([clip, pad], dim=0)
            padded.append(clip)

        x = torch.stack(padded, dim=0).to(self.device, non_blocking=True)  # (B,T,C,H,W)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(self.device.type == "cuda"),
        ):
            feats = self.model.encode_video(x)
        return feats.float().detach().cpu()

    def _encode_windows_internvideo2_clip_s(
        self, windows: List[List[np.ndarray]]
    ) -> torch.Tensor:
        """Encode windowed clips with InternVideo2_CLIP_S."""
        if not windows:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)

        target_t = (
            int(self.pe_target_T)
            if self.pe_target_T is not None
            else int(getattr(self.model.config, "num_frames", 8))
        )
        feats_list = []

        for w in windows:
            if not w:
                w = [np.zeros((224, 224, 3), dtype=np.uint8)]

            if len(w) >= target_t:
                if target_t > 1:
                    idxs = [
                        int(round(i * (len(w) - 1) / (target_t - 1)))
                        for i in range(target_t)
                    ]
                else:
                    idxs = [len(w) // 2]
            else:
                idxs = list(range(len(w))) + [len(w) - 1] * (target_t - len(w))

            frames = []
            for i in idxs:
                rgb = cv2.cvtColor(w[i], cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())

            clip = torch.stack(frames, dim=0).to(torch.uint8)  # (T,C,H,W)
            if callable(self.tokenizer):
                clip = self.tokenizer(clip)
            if not torch.is_tensor(clip):
                raise TypeError(
                    f"InternVideo2 transform returned unsupported type: {type(clip)}"
                )
            if clip.dim() == 4:
                clip = clip.unsqueeze(0)  # (1,T,C,H,W)

            clip = clip.to(self.device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(self.device.type == "cuda"),
            ):
                try:
                    raw_feats = self.model.encode_vision(clip, test=True)
                except TypeError:
                    raw_feats = self.model.encode_vision(clip)
                feats = self._normalize_video_feature_output(raw_feats)
                if feats.dim() >= 3:
                    feats = feats.mean(dim=1)
                feats_list.append(feats.float().detach().cpu())

        if len(feats_list) == 0:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)
        return torch.cat(feats_list, dim=0)

    # ------------------------- labels
    def extract_labels(self, path: str) -> Optional[str]:
        if self.dataset_name == "breakfast":
            label = path.split("/")[-1]
            return label.split("_")[1].replace(".mp4", "")
        elif self.dataset_name == "ucf101":
            label = path.split("/")[-1]
            return label.split("_")[1]
        elif self.dataset_name == "hmdb":
            return path.split("/")[-2]
        elif self.dataset_name == "something2":
            return path.split("/")[1]
        elif self.dataset_name in {"haa500", "haa100"}:
            return path.split("/")[-2]
        elif self.dataset_name == "barista":
            return os.path.basename(os.path.normpath(path))
        elif self.dataset_name in {
            "epic",
            "epic100",
            "epic_100",
            "epic55",
            "epic_55",
            "epicskitchen",
            "epic_kitchens",
            "epic_kitchens_100",
            "epic-kitchens-100",
        }:
            return os.path.splitext(os.path.basename(path))[0]
        return None

    # ------------------------- main
    def embed_video(
        self,
        video_paths,
        window_size,
        output_path,
        random=True,
        save_intermediate=False,
        frames_per_window=1,
        batch_size=256,
        stride: Optional[int] = None,
    ):
        os.makedirs(output_path, exist_ok=True)

        video_embedding_paths, labels, video_window_spans, video_meta = {}, [], {}, {}

        video_paths = sorted(video_paths)
        save_base = os.path.join(
            output_path, f"{self.dataset_name}_{self.model_name}_{window_size}_state"
        )
        final_path = save_base + ".npy"
        tmp_path = save_base + ".tmp.npy"

        def atomic_save_state(state):
            for attempt in range(2):
                try:
                    os.makedirs(output_path, exist_ok=True)
                    np.save(tmp_path, state, allow_pickle=True)
                    os.replace(tmp_path, final_path)
                    return
                except FileNotFoundError:
                    if attempt == 0:
                        continue
                    raise

        processed_count = 0
        if save_intermediate:
            load_path = (
                final_path
                if os.path.exists(final_path)
                else (tmp_path if os.path.exists(tmp_path) else None)
            )
            if load_path:
                try:
                    loaded = np.load(load_path, allow_pickle=True).item()
                    video_embedding_paths = loaded.get("video_embeddings", {})
                    labels = loaded.get("labels", [])
                    video_window_spans = loaded.get("video_window_spans", {})
                    video_meta = loaded.get("video_meta", {})
                    processed_count = len(video_embedding_paths)
                except Exception:
                    processed_count = 0
        if processed_count > 0:
            video_paths = video_paths[processed_count:]

        counter_since_last_save = 0
        for video_path in tqdm.tqdm(video_paths):
            labels.append(self.extract_labels(video_path))
            if (
                self.model_name in {"pe-l14", "pe-g14", "internvideo2-clip-s"}
                and os.path.isdir(video_path)
            ):
                window_embeddings, spans, fps, read_frames = (
                    self._encode_image_folder_windows_streaming(
                        video_path,
                        window_size,
                        frames_per_window=frames_per_window,
                        random=random,
                        batch_size=batch_size,
                        stride=stride,
                    )
                )
                video_embedding_paths[video_path] = window_embeddings
                video_window_spans[video_path] = spans
                video_meta[video_path] = {"fps": fps, "frame_count": float(read_frames)}
            elif str(video_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                window_embeddings, spans, fps, read_frames = (
                    self._encode_video_file_windows_streaming(
                        video_path,
                        window_size,
                        frames_per_window=frames_per_window,
                        random=random,
                        batch_size=batch_size,
                        stride=stride,
                    )
                )
                video_embedding_paths[video_path] = window_embeddings
                video_window_spans[video_path] = spans
                video_meta[video_path] = {"fps": fps, "frame_count": float(read_frames)}
            else:
                windows, spans, fps, read_frames = self._read_windows(
                    video_path, window_size, stride=stride
                )

                if len(windows) == 0:
                    video_embedding_paths[video_path] = np.zeros(
                        (0, self.embed_dim), dtype=np.float32
                    )
                    video_window_spans[video_path] = []
                    video_meta[video_path] = {"fps": fps, "frame_count": float(read_frames)}
                else:
                    window_embeddings = self._encode_windows(
                        windows, frames_per_window, random, batch_size
                    )
                    
                    video_embedding_paths[video_path] = window_embeddings
                    video_window_spans[video_path] = spans
                    video_meta[video_path] = {"fps": fps, "frame_count": float(read_frames)}

            counter_since_last_save += 1
            if save_intermediate:
                state = {
                    "video_embeddings": video_embedding_paths,
                    "labels": labels,
                    "video_window_spans": video_window_spans,
                    "video_meta": video_meta,
                }
                atomic_save_state(state)
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Save final state if save_intermediate is enabled
        # This ensures all processed videos are saved, even if the count wasn't a multiple of 10
        if save_intermediate and counter_since_last_save > 0:
            state = {
                "video_embeddings": video_embedding_paths,
                "labels": labels,
                "video_window_spans": video_window_spans,
                "video_meta": video_meta,
            }
            atomic_save_state(state)
            # Only delete tmp_path if it exists separately (shouldn't happen after os.replace, but be safe)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            # DO NOT delete final_path - it's needed for merging in multi-GPU scenarios

        self.video_embeddings = video_embedding_paths
        self.labels = labels
        self.video_window_spans = video_window_spans
        self.video_meta = video_meta

    def _encode_video_file_windows_streaming(
        self,
        video_path: str,
        window_size: int,
        frames_per_window=1,
        random=True,
        batch_size=256,
        stride: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], float, int]:
        step = stride if stride is not None and stride > 0 else window_size
        if step != window_size:
            # Overlapping windows need frame buffering; keep the old path for that rare case.
            windows, spans, fps, read_frames = self._read_windows(video_path, window_size, stride=stride)
            return self._encode_windows(windows, frames_per_window, random, batch_size), spans, fps, read_frames

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[embed_video] Warning: could not open video file: {video_path}", flush=True)
            cap.release()
            return np.zeros((0, self.embed_dim), dtype=np.float32), [], 0.0, 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        embeddings, spans, windows = [], [], []
        video_model = self.model_name in {"pe-l14", "pe-g14", "internvideo2-clip-s"}
        window_batch_size = (
            self.pe_video_batch_size or min(batch_size, 8)
            if video_model
            else max(1, min(batch_size, 64))
        )
        read_frames = 0

        def flush_windows() -> None:
            if not windows:
                return
            embeddings.append(
                self._encode_windows(windows, frames_per_window, random, batch_size)
            )
            windows.clear()

        try:
            while True:
                chunk = []
                start_frame_index = read_frames
                for _ in range(window_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    chunk.append(frame)
                    read_frames += 1
                if not chunk:
                    break
                if video_model:
                    encoded_chunk = chunk
                else:
                    sample_indices = self._sample_indices(len(chunk), frames_per_window, random)
                    if not sample_indices:
                        sample_indices = [len(chunk) // 2]
                    encoded_chunk = [chunk[index] for index in sample_indices]
                windows.append(encoded_chunk)
                spans.append((start_frame_index / fps, (read_frames - 1) / fps))
                if len(windows) >= window_batch_size:
                    flush_windows()
                if len(chunk) < window_size:
                    break
        finally:
            cap.release()

        flush_windows()
        if read_frames == 0:
            print(f"[embed_video] Warning: no frames read from video file: {video_path}", flush=True)
        if embeddings:
            return np.concatenate(embeddings, axis=0), spans, fps, read_frames
        return np.zeros((0, self.embed_dim), dtype=np.float32), spans, fps, read_frames

    def _encode_image_folder_windows_streaming(
        self,
        video_path: str,
        window_size: int,
        frames_per_window: int,
        random: bool,
        batch_size: int,
        stride: Optional[int],
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], float, int]:
        step = stride if stride is not None and stride > 0 else window_size
        fps = 12.0
        img_files = sorted(
            os.path.join(video_path, f)
            for f in os.listdir(video_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        frame_count = len(img_files)
        target_t = int(self.pe_target_T) if self.pe_target_T is not None else min(window_size, 16)
        windows, spans, embeddings = [], [], []
        pe_bs = self.pe_video_batch_size or min(batch_size, 8)

        def flush_windows() -> None:
            if not windows:
                return
            embeddings.append(
                self._encode_windows(windows, frames_per_window, random, batch_size)
            )
            windows.clear()

        for i in range(0, frame_count, step):
            chunk_files = img_files[i : i + window_size]
            sample_indices = self._sample_indices(len(chunk_files), target_t, random=False)
            chunk = [cv2.imread(chunk_files[index]) for index in sample_indices]
            chunk = [frame for frame in chunk if frame is not None]
            if len(chunk) == 0:
                continue
            windows.append(chunk)
            spans.append((i / fps, (i + len(chunk) - 1) / fps))
            if len(windows) >= pe_bs:
                flush_windows()
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        flush_windows()
        if not embeddings:
            return np.zeros((0, self.embed_dim), dtype=np.float32), [], fps, frame_count
        return np.concatenate(embeddings, axis=0), spans, fps, frame_count

    def process_data(
        self,
        folder_path,
        window_size,
        output_path,
        random=True,
        save_intermediate=False,
        frames_per_window=1,
        batch_size=256,
        stride: Optional[int] = None,
        number = 0
    ):
        os.makedirs(output_path, exist_ok=True)
        video_paths = []
        if isinstance(folder_path, list):
            for path in folder_path:
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(".mp4"):
                            video_paths.append(os.path.join(root, file))
        else:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(".mp4"):
                        video_paths.append(os.path.join(root, file))
        print(len(video_paths), "videos found in", folder_path)

        if number > 0:
            video_paths = video_paths[:number]
        
        start = time.time()
        self.embed_video(
            video_paths,
            window_size,
            output_path,
            random=random,
            save_intermediate=save_intermediate,
            frames_per_window=frames_per_window,
            batch_size=batch_size,
            stride=stride,
        )
        end = time.time()
        all_time = (end - start)  # store per-example time
        
        #average_time = all_time/number
        #print(average_time)
    


class Create_Concepts(_PickleBackendsMixin):
    def __init__(self, model_name, model, tokenizer, clip_model=None, clip_tokenizer=None):
        self.model_name = model_name.lower()
        self.model = model
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model = self.model.to(self.device).eval()
            # Keep CPU text inference in fp32 to avoid layernorm dtype errors with fp16 weights.
            if self.device.type != "cuda":
                self.model = self.model.float()
            # InternVideo2 text stack uses fp32 layer-norm wrappers; keep text encoder
            # in fp32 even when the rest of the model is fp16.
            if self.model_name == "internvideo2-clip-s" and hasattr(self.model, "text_encoder"):
                self.model.text_encoder = self.model.text_encoder.float()

        self.dataset_name = None
        self.video_embeddings = None
        self.labels = None
        self.text_concepts = None
        self.text_embeddings = None
        self.concept_types = None
        self.concept_name_to_type = None

    def embedd_text(self, *text):
        concepts = []

        # Case 1: multiple positional args given
        if len(text) > 1:
            for t in text:
                if isinstance(t, str):
                    concepts.extend([c.strip() for c in t.split(",") if c.strip()])
                elif isinstance(t, list):
                    for item in t:
                        concepts.extend(
                            [c.strip() for c in item.split(",") if c.strip()]
                        )
        else:
            t = text[0]
            if isinstance(t, str):
                concepts.extend([c.strip() for c in t.split(",") if c.strip()])
            elif isinstance(t, list):
                for item in t:
                    concepts.extend([c.strip() for c in item.split(",") if c.strip()])

        # Deduplicate while preserving order
        seen = set()
        concepts = [c for c in concepts if not (c in seen or seen.add(c))]
        # Tokenize & embed

        if self.model_name == "clip":
            inputs = self.tokenizer(
                text=concepts, return_tensors="pt", padding=True, truncation=True
            )
            text_inputs = {
                k: v.to(self.model.device)
                for k, v in inputs.items()
                if k in ("input_ids", "attention_mask", "position_ids")
            }
            with torch.no_grad():
                text_out = self.model.text_model(**text_inputs)
                outputs = self.model.text_projection(text_out.pooler_output)

        elif self.model_name == "pe-l14" or self.model_name == "pe-g14":
            inputs = self.tokenizer(
                concepts).to(self.device)
            with torch.no_grad():
                outputs = self.model.encode_text(inputs)
        elif self.model_name == "siglip" or self.model_name == "siglipl14":
            inputs = self.tokenizer(
                text=concepts, padding="max_length", return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
        elif self.model_name == "siglip2":
            inputs = self.tokenizer(
                text=concepts, padding=True, return_tensors="pt"
            ).to(self.model.device)
            # text_inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
        elif self.model_name == "res50":
            inputs = clip.tokenize(concepts)  # returns CPU tensor by default
            inputs = inputs.to(self.device)  # move tokens to model device
            with torch.no_grad():
                outputs = self.model.encode_text(inputs).detach().cpu()
        elif self.model_name == "clip4clip":
            inputs = self.tokenizer(
                concepts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)
            outputs = (
                self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )[0]
                .detach()
                .cpu()
            )
        elif self.model_name == "internvideo2-clip-s":
            if not callable(self.tokenizer):
                raise ValueError("InternVideo2_CLIP_S tokenizer is not callable.")

            inputs = self.tokenizer(concepts)
            if isinstance(inputs, dict):
                inputs = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
            elif hasattr(inputs, "to"):
                inputs = inputs.to(self.device)

            with torch.no_grad():
                if hasattr(self.model, "encode_text"):
                    outputs = self.model.encode_text(inputs)
                elif hasattr(self.model, "get_txt_feat"):
                    outputs = self.model.get_txt_feat(inputs)
                else:
                    raise NotImplementedError(
                        "InternVideo2_CLIP_S text API not found. Expected encode_text/get_txt_feat."
                    )
        else:
            outputs = None

        # Normalize model outputs across HF versions to a tensor [N, D].
        if outputs is not None and not torch.is_tensor(outputs):
            if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                outputs = outputs.text_embeds
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                if self.clip_model is not None and hasattr(self.clip_model, "text_projection"):
                    outputs = self.clip_model.text_projection(outputs.pooler_output)
                elif hasattr(self.model, "text_projection"):
                    outputs = self.model.text_projection(outputs.pooler_output)
                else:
                    outputs = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                outputs = outputs.last_hidden_state[:, 0, :]
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
                outputs = outputs[0]
            else:
                raise TypeError(
                    f"Unsupported text embedding output type: {type(outputs)}"
                )

        self.text_embeddings = outputs
        self.text_concepts = concepts
