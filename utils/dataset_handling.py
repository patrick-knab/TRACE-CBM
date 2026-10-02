import os

import random
import time
import pickle
import shutil
import numpy as np
import torch
from typing import Optional, List, Dict, Tuple, Any

try:
    from transformers import CLIPProcessor, CLIPModel
    from transformers import (
        CLIPVisionModelWithProjection,
        CLIPTokenizer,
        CLIPTextModelWithProjection,
    )
    from transformers import AutoProcessor, AutoModel  # siglip
except ImportError as error:
    class _MissingTransformersClass:
        def __init__(self, class_name: str, import_error: ImportError) -> None:
            self.class_name = class_name
            self.import_error = import_error

        def from_pretrained(self, *args, **kwargs):
            raise ImportError(
                f"{self.class_name} is unavailable in the installed transformers package. "
                "Install a transformers version with CLIP/SigLIP model support, or use a "
                "cached embedding file/backbone path that does not require this class."
            ) from self.import_error

    CLIPProcessor = _MissingTransformersClass("CLIPProcessor", error)
    CLIPModel = _MissingTransformersClass("CLIPModel", error)
    CLIPVisionModelWithProjection = _MissingTransformersClass("CLIPVisionModelWithProjection", error)
    CLIPTokenizer = _MissingTransformersClass("CLIPTokenizer", error)
    CLIPTextModelWithProjection = _MissingTransformersClass("CLIPTextModelWithProjection", error)
    AutoProcessor = _MissingTransformersClass("AutoProcessor", error)
    AutoModel = _MissingTransformersClass("AutoModel", error)

try:
    from .core.vision_encoder import pe
    from .core.vision_encoder import transforms as pe_transformer
    from .video_embedder import VideoEmbedder
except ImportError:
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as pe_transformer
    from video_embedder import VideoEmbedder
import clip

import multiprocessing as mp
from multiprocessing import Queue, set_start_method
from utils.paths import dataset_root as configured_dataset_root, embedding_root as configured_embedding_root

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# This must be done before creating any processes
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

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


# Set random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Map dataset keys to readable names
DATASET_MAP = {
    "breakfast": "Breakfast",
    "gtea": "GTEA_Gaze",
    "gtea_gaze": "GTEA_Gaze",
    "egtea": "GTEA_Gaze",
    "egtea_gaze": "GTEA_Gaze",
    "mpii": "MPII_Cooking_2",
    "mpii_cooking": "MPII_Cooking_2",
    "mpii_cooking_2": "MPII_Cooking_2",
    "mpiicooking2": "MPII_Cooking_2",
    "ucf101": "UCF101",
    "hmdb": "HMDB",
    "ssv2": "Something2",
    "haa500": "HAA500",
    "barista": "Barista",
    "epic": "EPIC-KITCHENS-100",
    "epic100": "EPIC-KITCHENS-100",
    "epic_100": "EPIC-KITCHENS-100",
    "epic_kitchens": "EPIC-KITCHENS-100",
    "epic_kitchens_100": "EPIC-KITCHENS-100",
    "epic-kitchens-100": "EPIC-KITCHENS-100",
}

CLIP_MODEL_ALIASES = {
    "pe_g14": "pe-g14",
    "pe_l14": "pe-l14",
}

OPENAI_CLIP_MODEL_IDS = {
    "b32": "ViT-B/32",
    "b16": "ViT-B/16",
    "l14": "ViT-L/14",
    "res50": "RN50",
}

HF_CLIP_MODEL_IDS = {
    "b32": "openai/clip-vit-base-patch32",
    "b16": "openai/clip-vit-base-patch16",
    "l14": "openai/clip-vit-large-patch14",
}


def canonicalize_clip_model_name(clip_model: str) -> str:
    """Normalize user-facing aliases to the internal model key."""
    model_name = str(clip_model).strip().lower()
    return CLIP_MODEL_ALIASES.get(model_name, model_name)


def _load_clip_image_model(clip_model_name: str, device: str = "cuda"):
    """Load an image CLIP backbone.

    Prefer HuggingFace CLIP when that API is available, but fall back to the
    OpenAI clip package for the standard ViT/RN50 backbones. The cluster env
    used for these embedding jobs currently has OpenAI clip but an older
    Transformers build without CLIPProcessor/CLIPModel.
    """
    clip_model_name = canonicalize_clip_model_name(clip_model_name)
    if clip_model_name == "res50":
        return clip.load("RN50", device=device)
    if clip_model_name not in HF_CLIP_MODEL_IDS:
        raise ValueError(f"Unsupported OpenAI/HF CLIP image model: {clip_model_name}")

    model_id = HF_CLIP_MODEL_IDS[clip_model_name]
    try:
        model = CLIPModel.from_pretrained(model_id).eval()
        processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
        return model, processor
    except ImportError as error:
        openai_id = OPENAI_CLIP_MODEL_IDS[clip_model_name]
        print(
            f"[clip-load] Falling back to OpenAI CLIP {openai_id} for {clip_model_name}: {error}",
            flush=True,
        )
        return clip.load(openai_id, device=device)


def _state_file_stem(
    dataset_key: str,
    dataset_name: str,
    model_name: str,
    window_size: int,
) -> str:
    return f"{dataset_name}_{model_name}_{window_size}_state"


def _embedder_filename(
    dataset_key: str,
    clip_model: str,
    random: bool,
    window_size: int,
) -> str:
    return f"{random}_{window_size}_clip_{clip_model}.pkl"


def _intermediate_dirs(
    output_dir: str,
    clip_model: str,
    window_size: int,
    num_gpus: int,
) -> List[str]:
    run_dir = os.path.join(output_dir, f"{clip_model}_{window_size}_intermediate")
    return [os.path.join(run_dir, f"gpu_{gpu_id}") for gpu_id in range(num_gpus)]


def _intermediate_state_paths(
    intermediate_dir: str,
    dataset_key: str,
    model_name: str,
    window_size: int,
) -> Tuple[str, str]:
    state_file_base = os.path.join(
        intermediate_dir, f"{dataset_key}_{model_name}_{window_size}_state"
    )
    return state_file_base + ".npy", state_file_base + ".tmp.npy"


def _prepare_intermediate_dirs(
    output_dir: str,
    dataset_key: str,
    clip_model: str,
    window_size: int,
    num_gpus: int,
) -> List[str]:
    """Create per-model intermediate dirs and migrate legacy partial states."""
    model_name = clip_model
    dirs = _intermediate_dirs(output_dir, clip_model, window_size, num_gpus)
    for gpu_id, intermediate_dir in enumerate(dirs):
        os.makedirs(intermediate_dir, exist_ok=True)

        legacy_dir = os.path.join(output_dir, f"gpu_{gpu_id}_intermediate")
        if legacy_dir == intermediate_dir or not os.path.isdir(legacy_dir):
            continue

        for legacy_path, new_path in zip(
            _intermediate_state_paths(legacy_dir, dataset_key, model_name, window_size),
            _intermediate_state_paths(intermediate_dir, dataset_key, model_name, window_size),
        ):
            if os.path.exists(legacy_path) and not os.path.exists(new_path):
                shutil.copy2(legacy_path, new_path)
                print(
                    f"Migrated legacy intermediate state for GPU {gpu_id}: "
                    f"{legacy_path} -> {new_path}",
                    flush=True,
                )
    return dirs


def get_all_video_paths(folder_path, dataset_name):
    """Collect all video paths from the dataset folder."""
    video_paths = []
    dataset_key = str(dataset_name).lower()
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    image_extensions = (".jpg", ".jpeg", ".png")
    if dataset_key in {"mpii_cooking_2", "gtea_gaze", "barista"}:
        image_root = str(folder_path).replace("/Video_data", "/Image_data")
        if os.path.isdir(image_root):
            for root, dirs, files in os.walk(image_root):
                if any(file.lower().endswith(image_extensions) for file in files):
                    video_paths.append(root)
                    dirs[:] = []
            if video_paths:
                return sorted(video_paths)
    if isinstance(folder_path, list):
        for path in folder_path:
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(video_extensions):
                        video_paths.append(os.path.join(root, file))
    else:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_paths.append(os.path.join(root, file))
    return sorted(video_paths)


def _dataset_root(dataset_name: str) -> str:
    dataset_base = str(configured_dataset_root())
    if dataset_name == "EPIC-KITCHENS-100":
        return (
            os.environ.get("EPIC_KITCHENS_ROOT")
            or os.path.join(dataset_base, dataset_name)
        )
    return os.path.join(dataset_base, dataset_name)


def _dataset_video_root(dataset_name: str) -> str:
    root = _dataset_root(dataset_name)
    if dataset_name == "EPIC-KITCHENS-100":
        epic_root = os.path.join(root, "EPIC-KITCHENS")
        return epic_root if os.path.isdir(epic_root) else root
    return os.path.join(root, "Video_data")


def worker_process(
    gpu_id: int,
    video_paths: List[str],
    dataset_key: str,
    clip_model_name: str,
    window_size: int,
    random: bool,
    batch_size: int,
    pe_video_batch_size: Optional[int],
    pe_target_T: Optional[int],
    enable_tf32: bool,
    seed: int,
    output_dir: str,
    intermediate_dir: str,
    result_queue: Queue,
    progress_queue: Queue,
):
    """Worker process that runs on a specific GPU."""
    try:
        clip_model_name = canonicalize_clip_model_name(clip_model_name)

        # With 'spawn' method, each process starts fresh
        # CUDA_VISIBLE_DEVICES is inherited from parent (set by SLURM to "0,1,2,3")
        # We just need to select the right device ID for this worker
        init_repro(seed)
        
        # Set the CUDA device for this worker process
        # gpu_id corresponds to the logical GPU index (0, 1, 2, 3)
        if torch.cuda.is_available():
            if gpu_id < torch.cuda.device_count():
                torch.cuda.set_device(gpu_id)
                print(f"[GPU {gpu_id}] Using CUDA device {gpu_id} (device name: {torch.cuda.get_device_name(gpu_id)})")
            else:
                raise RuntimeError(f"GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs available)")
        dataset_name = DATASET_MAP.get(dataset_key.lower())
        
        # Optional: enable TF32 for faster matmul on Ampere+
        if enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Load CLIP model & processor
        if clip_model_name in {"b32", "b16", "l14", "res50"}:
            model, processor = _load_clip_image_model(clip_model_name, device="cuda")
        elif clip_model_name == "clip4clip":
            model = CLIPVisionModelWithProjection.from_pretrained(
                "Searchium-ai/clip4clip-webvid150k"
            )
            model = model.eval()
            clip_full = CLIPModel.from_pretrained(
                "Searchium-ai/clip4clip-webvid150k"
            )
            model_text = CLIPTextModelWithProjection.from_pretrained(
                "Searchium-ai/clip4clip-webvid150k"
            )
            processor = CLIPTokenizer.from_pretrained(
                "Searchium-ai/clip4clip-webvid150k"
            )
        elif clip_model_name == "siglip":
            model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
            processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        elif clip_model_name == "siglipl14":
            model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
            processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        elif clip_model_name == "pe-l14":
            model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
            processor = pe_transformer.get_image_transform(model.image_size)
            tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
        elif clip_model_name == "pe-g14":
            model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True)
            processor = pe_transformer.get_image_transform(model.image_size)
            tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
        else:
            raise ValueError(f"Unknown CLIP model: {clip_model_name}")
        
        # Create embedder
        if (
            clip_model_name == "clip4clip"
            or clip_model_name == "siglip"
            or clip_model_name == "siglipl14"
            or clip_model_name == "res50"
            or clip_model_name == "pe-l14"
            or clip_model_name == "pe-g14"
        ):
            embedder = VideoEmbedder(
                clip_model_name, model, processor,
                pe_video_batch_size=pe_video_batch_size,
                pe_target_T=pe_target_T,
            )
        else:
            # Use the actual CLIP variant name (b32/b16/l14) so intermediate files
            # are uniquely named and can be merged correctly.
            embedder = VideoEmbedder(clip_model_name, model, processor)
        embedder.dataset_name = dataset_key
        
        # Check for existing intermediate state file
        state_file_base = os.path.join(
            intermediate_dir, f"{dataset_key}_{clip_model_name}_{window_size}_state"
        )
        state_file = state_file_base + ".npy"
        tmp_state_file = state_file_base + ".tmp.npy"
        
        already_processed = 0
        if os.path.exists(state_file) or os.path.exists(tmp_state_file):
            load_path = state_file if os.path.exists(state_file) else tmp_state_file
            try:
                loaded = np.load(load_path, allow_pickle=True).item()
                already_processed = len(loaded.get("video_embeddings", {}))
                print(f"[GPU {gpu_id}] Found existing state file: {load_path}")
                print(f"[GPU {gpu_id}] Resuming: {already_processed}/{len(video_paths)} videos already processed")
            except Exception as e:
                print(f"[GPU {gpu_id}] Warning: Could not load state file {load_path}: {e}")
                already_processed = 0
        
        if already_processed > 0:
            print(f"[GPU {gpu_id}] Resuming from checkpoint: {already_processed} videos already done")
        else:
            print(f"[GPU {gpu_id}] Starting fresh: Processing {len(video_paths)} videos...")
        
        start_time = time.time()
        
        # Process videos on this GPU (embed_video will automatically resume from state file)
        embedder.embed_video(
            video_paths,
            window_size,
            intermediate_dir,
            random=random,
            save_intermediate=True,
            batch_size=batch_size,
        )
        
        elapsed = time.time() - start_time
        print(f"[GPU {gpu_id}] Completed {len(video_paths)} videos in {elapsed:.2f} seconds")
        
        # Return results
        result_queue.put({
            "gpu_id": gpu_id,
            "num_videos": len(video_paths),
            "elapsed_time": elapsed,
            "intermediate_dir": intermediate_dir,
            "success": True
        })
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({
            "gpu_id": gpu_id,
            "success": False,
            "error": str(e)
        })


def merge_intermediate_results(
    intermediate_dirs: List[str], 
    output_path: str, 
    dataset_key: str,
    dataset_name: str,
    model_name: str,
    window_size: int
):
    """Merge intermediate embedding results from multiple GPUs."""
    print(f"Merging results from {len(intermediate_dirs)} GPUs...")
    
    # Find all intermediate state files (.npy format)
    # Prefer .npy over .tmp.npy (final over temporary)
    all_state_files = []
    for intermediate_dir in intermediate_dirs:
        # Look for final .npy files first
        final_state = os.path.join(
            intermediate_dir, f"{dataset_key}_{model_name}_{window_size}_state.npy"
        )
        tmp_state = os.path.join(
            intermediate_dir, f"{dataset_key}_{model_name}_{window_size}_state.tmp.npy"
        )
        
        # Prefer final over tmp, but use tmp if final doesn't exist
        if os.path.exists(final_state):
            all_state_files.append(final_state)
        elif os.path.exists(tmp_state):
            all_state_files.append(tmp_state)
            print(f"  Using temporary state file for {intermediate_dir}")
    
    if not all_state_files:
        raise ValueError(f"No intermediate state files found in {intermediate_dirs}!")
    
    print(f"Found {len(all_state_files)} intermediate state files")
    
    # Load and merge all embeddings
    merged_video_embeddings = {}
    merged_labels = []
    merged_video_window_spans = {}
    merged_video_meta = {}
    
    for state_file in sorted(all_state_files):
        print(f"Loading {os.path.basename(state_file)}...")
        try:
            state = np.load(state_file, allow_pickle=True).item()
            num_videos = len(state.get("video_embeddings", {}))
            print(f"  Found {num_videos} video embeddings in this file")
            
            merged_video_embeddings.update(state.get("video_embeddings", {}))
            merged_labels.extend(state.get("labels", []))
            merged_video_window_spans.update(state.get("video_window_spans", {}))
            merged_video_meta.update(state.get("video_meta", {}))
        except Exception as e:
            print(f"Warning: Failed to load {state_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Merged {len(merged_video_embeddings)} video embeddings")
    print(f"Total labels: {len(merged_labels)}")
    
    # Save merged state as .npy (same format as intermediate files)
    output_dir = os.path.dirname(output_path)
    save_base = os.path.join(
        output_dir,
        _state_file_stem(dataset_key, dataset_name, model_name, window_size),
    )
    merged_state_path = save_base + ".npy"
    tmp_path = save_base + ".tmp.npy"
    
    merged_state = {
        "video_embeddings": merged_video_embeddings,
        "labels": merged_labels,
        "video_window_spans": merged_video_window_spans,
        "video_meta": merged_video_meta,
    }
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(tmp_path, merged_state, allow_pickle=True)
    os.replace(tmp_path, merged_state_path)
    
    print(f"Merged state saved to {merged_state_path}")
    
    return merged_state_path


def process_dataset(
    dataset_key,
    clip_model,
    window_size=16,
    random=True,
    batch_size: int = 256,
    pe_video_batch_size: Optional[int] = None,
    pe_target_T: Optional[int] = None,
    enable_tf32: bool = True,
    seed: int = 42,
    num_gpus: Optional[int] = None,
    use_parallel: bool = True,
    cleanup_intermediate: bool = True,
    save_embedder_pickle: bool = True,
    save_portable_state_pickle: bool = True,
):
    """
    Process dataset with optional multi-GPU parallelization.
    
    Args:
        num_gpus: Number of GPUs to use. If None, uses all available GPUs.
        use_parallel: If True, use multi-GPU parallelization. If False, use single GPU.
        cleanup_intermediate: If True, delete per-GPU intermediate directories after a
            successful merge and final pickle save.
        save_embedder_pickle: If True, save a pickled `VideoEmbedder` object (requires
            `video_embedder` module to be importable when loading).
        save_portable_state_pickle: If True, also save a plain-python dict containing
            embeddings/labels/meta that can be loaded without `video_embedder`.
    """
    init_repro(seed)
    clip_model = canonicalize_clip_model_name(clip_model)
    dataset_name = DATASET_MAP.get(dataset_key.lower())
    if dataset_name is None:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    folder_path = _dataset_video_root(dataset_name)
    # Keep as a string: this path is used with os.path.join / os.makedirs.
    embedding_root = str(configured_embedding_root())
    output_dir = os.path.join(embedding_root, dataset_name)
    embedd_filename = _embedder_filename(
        dataset_key=dataset_key,
        clip_model=clip_model,
        random=random,
        window_size=window_size,
    )
    embedd_path = os.path.join(output_dir, embedd_filename)
    merged_state_path = os.path.join(
        output_dir,
        _state_file_stem(dataset_key, dataset_name, clip_model, window_size) + ".npy",
    )
    portable_state_path = embedd_path.replace(".pkl", "_state.pkl")
    
    # Check if final embedding already exists. The portable state is the robust
    # artifact used by newer training/eval code; the full embedder pickle is
    # optional because reconstructing model backends can be environment-fragile.
    if save_embedder_pickle and os.path.exists(embedd_path):
        print(f"Final embedding already exists at {embedd_path}")
        return embedd_path
    if os.path.exists(merged_state_path):
        print(f"Merged embedding state already exists at {merged_state_path}")
        return merged_state_path
    if os.path.exists(portable_state_path):
        print(f"Portable embedding state already exists at {portable_state_path}")
        return portable_state_path
    
    prepared_intermediate_dirs = None

    # Check for existing intermediate directories (for resume)
    if use_parallel:
        # Determine number of GPUs
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if num_gpus > 0:
            prepared_intermediate_dirs = _prepare_intermediate_dirs(
                output_dir, dataset_key, clip_model, window_size, num_gpus
            )
            intermediate_dirs = prepared_intermediate_dirs
            existing_intermediates = [d for d in intermediate_dirs if os.path.exists(d)]
            if existing_intermediates:
                model_name = clip_model
                state_files = []
                for intermediate_dir in existing_intermediates:
                    state_file, _ = _intermediate_state_paths(
                        intermediate_dir, dataset_key, model_name, window_size
                    )
                    if os.path.exists(state_file):
                        state_files.append(state_file)
                
                if state_files:
                    print(f"Found {len(state_files)} existing intermediate state files - will resume processing")
                    print(f"  Intermediate directories: {existing_intermediates}")
    
    # Get all video paths
    print(f"Collecting video paths from {folder_path}...")
    all_video_paths = get_all_video_paths(folder_path, dataset_key)
    print(f"Found {len(all_video_paths)} videos")

    if not use_parallel or len(all_video_paths) == 0:
        # Single GPU processing (original behavior)
        device = "cpu" if num_gpus == 0 else "cuda"
        return process_dataset_single_gpu(
            dataset_key, clip_model, window_size, random, batch_size,
            pe_video_batch_size, pe_target_T, enable_tf32, seed, embedd_path,
            device=device,
        )
    
    # Multi-GPU parallel processing
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("No GPUs available, falling back to single CPU processing")
        return process_dataset_single_gpu(
            dataset_key, clip_model, window_size, random, batch_size,
            pe_video_batch_size, pe_target_T, enable_tf32, seed, embedd_path,
            device = "cpu"
        )
    
    print(f"Using {num_gpus} GPUs for parallel processing")
    
    # Split video paths across GPUs
    videos_per_gpu = len(all_video_paths) // num_gpus
    video_splits = []
    for i in range(num_gpus):
        start_idx = i * videos_per_gpu
        if i == num_gpus - 1:
            # Last GPU gets remaining videos
            end_idx = len(all_video_paths)
        else:
            end_idx = (i + 1) * videos_per_gpu
        video_splits.append(all_video_paths[start_idx:end_idx])
    
    print(f"Split videos across GPUs:")
    for i, split in enumerate(video_splits):
        print(f"  GPU {i}: {len(split)} videos")
    
    # Create intermediate directories for each GPU
    if prepared_intermediate_dirs is not None and len(prepared_intermediate_dirs) == num_gpus:
        intermediate_dirs = prepared_intermediate_dirs
    else:
        intermediate_dirs = _prepare_intermediate_dirs(
            output_dir, dataset_key, clip_model, window_size, num_gpus
        )
    
    # Create queues for communication
    # Use 'spawn' context for CUDA compatibility
    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    result_queue = manager.Queue()
    progress_queue = manager.Queue()
    
    # Start worker processes
    processes = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=worker_process,
            args=(
                gpu_id,
                video_splits[gpu_id],
                dataset_key,
                clip_model,
                window_size,
                random,
                batch_size,
                pe_video_batch_size,
                pe_target_T,
                enable_tf32,
                seed,
                output_dir,
                intermediate_dirs[gpu_id],
                result_queue,
                progress_queue,
            )
        )
        p.start()
        processes.append(p)
        print(f"Started worker process for GPU {gpu_id}")
    
    # Wait for all processes to complete
    print("Waiting for all GPU workers to complete...")
    start_time = time.time()
    for p in processes:
        p.join(timeout=3600*24)  # 24 hour timeout per process
        if p.is_alive():
            print(f"WARNING: Process for GPU {p} did not complete in time!")
    
    total_time = time.time() - start_time
    print(f"All workers completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Ensure we have results from all GPUs
    if len(results) < num_gpus:
        print(f"WARNING: Only received {len(results)} results for {num_gpus} GPUs")
        print("Some processes may have failed silently")
    
    # Check for errors
    failed_gpus = [r for r in results if not r.get("success", False)]
    if failed_gpus:
        print(f"WARNING: {len(failed_gpus)} GPU(s) failed:")
        for r in failed_gpus:
            print(f"  GPU {r['gpu_id']}: {r.get('error', 'Unknown error')}")
    
    successful_gpus = [r for r in results if r.get("success", False)]
    print(f"\nCompleted processing on {len(successful_gpus)}/{num_gpus} GPUs")

    if len(successful_gpus) == 0:
        raise RuntimeError(
            "All GPU workers failed; no intermediate state files to merge. "
            "See per-GPU error messages above."
        )
    
    # Merge results
    successful_intermediate_dirs = [intermediate_dirs[r["gpu_id"]] for r in successful_gpus]
    
    # Determine model name for state file naming
    model_name = clip_model

    merged_state_path = merge_intermediate_results(
        successful_intermediate_dirs, 
        embedd_path, 
        dataset_key,
        dataset_name,
        model_name,
        window_size
    )

    # Save a portable pickle (plain dict) that doesn't require importing VideoEmbedder.
    # This is convenient for analysis notebooks.
    if save_portable_state_pickle:
        state = np.load(merged_state_path, allow_pickle=True).item()
        portable = {
            "config": {
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "clip_model": clip_model,
                "window_size": window_size,
                "random": random,
                "batch_size": batch_size,
                "pe_video_batch_size": pe_video_batch_size,
                "pe_target_T": pe_target_T,
                "enable_tf32": enable_tf32,
                "seed": seed,
            },
            **state,
        }
        portable_path = embedd_path.replace(".pkl", "_state.pkl")
        with open(portable_path, "wb") as f:
            pickle.dump(portable, f)
        print(f"Portable state saved to {portable_path}")
    
    if save_embedder_pickle:
        # Now create a final embedder with merged results
        # We need to load the model and create embedder, then load the merged state
        print("Creating final embedder with merged results...")
        final_embedder = create_embedder_with_state(
            dataset_key, clip_model, merged_state_path,
            pe_video_batch_size, pe_target_T, enable_tf32, seed
        )
        
        # Save final embedder
        os.makedirs(os.path.dirname(embedd_path), exist_ok=True)
        with open(embedd_path, "wb") as f:
            pickle.dump(final_embedder, f)
        
        print(f"Final embedder saved to {embedd_path}")
    
    if cleanup_intermediate:
        for d in intermediate_dirs:
            try:
                shutil.rmtree(d)
                print(f"Deleted intermediate directory: {d}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"WARNING: Failed to delete intermediate directory {d}: {e}")
        for parent in sorted({os.path.dirname(d) for d in intermediate_dirs}):
            try:
                os.rmdir(parent)
                print(f"Deleted intermediate run directory: {parent}")
            except OSError:
                pass

    print(f"Parallel processing complete. Merged results saved.")
    return merged_state_path


def create_embedder_with_state(
    dataset_key: str,
    clip_model: str,
    state_path: str,
    pe_video_batch_size: Optional[int],
    pe_target_T: Optional[int],
    enable_tf32: bool,
    seed: int,
):
    """Create an embedder and load merged state into it."""
    init_repro(seed)
    clip_model = canonicalize_clip_model_name(clip_model)
    dataset_name = DATASET_MAP.get(dataset_key.lower())
    
    if enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load model & processor (minimal, just for structure)
    if clip_model == "pe-g14":
        model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True)
        processor = pe_transformer.get_image_transform(model.image_size)
        tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
        clip_model = "pe-l14"
    elif clip_model == "pe-l14":
        model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
        processor = pe_transformer.get_image_transform(model.image_size)
        tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
    elif clip_model in {"b32", "b16", "l14", "res50"}:
        model, processor = _load_clip_image_model(
            clip_model, device="cuda" if torch.cuda.is_available() else "cpu"
        )
    elif clip_model == "clip4clip":
        model = CLIPVisionModelWithProjection.from_pretrained(
            "Searchium-ai/clip4clip-webvid150k"
        ).eval()
        processor = CLIPTokenizer.from_pretrained(
            "Searchium-ai/clip4clip-webvid150k"
        )
    elif clip_model == "siglip":
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval()
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    elif clip_model == "siglipl14":
        model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").eval()
        processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    else:
        raise ValueError(f"Unsupported clip_model for create_embedder_with_state: {clip_model}")
    
    # Create embedder
    if clip_model == "pe-l14":
        embedder = VideoEmbedder(
            clip_model, model, processor,
            pe_video_batch_size=pe_video_batch_size,
            pe_target_T=pe_target_T,
        )
    else:
        # For other models, create with the loaded backends
        embedder = VideoEmbedder(clip_model, model, processor)
    
    embedder.dataset_name = dataset_key
    
    # Load merged state
    state = np.load(state_path, allow_pickle=True).item()
    embedder.video_embeddings = state.get("video_embeddings", {})
    labels_from_state = state.get("labels", None)
    if isinstance(labels_from_state, list) and (
        len(labels_from_state) == len(embedder.video_embeddings)
    ):
        embedder.labels = labels_from_state
    else:
        embedder.labels = [
            embedder.extract_labels(p) for p in state.get("video_embeddings", {}).keys()
        ]
    embedder.video_window_spans = state.get("video_window_spans", {})
    embedder.video_meta = state.get("video_meta", {})
    
    return embedder


def process_dataset_single_gpu(
    dataset_key,
    clip_model,
    window_size,
    random,
    batch_size,
    pe_video_batch_size,
    pe_target_T,
    enable_tf32,
    seed,
    embedd_path,
    device = "cuda"
):
    """Original single-GPU processing function."""
    clip_model = canonicalize_clip_model_name(clip_model)
    dataset_name = DATASET_MAP.get(dataset_key.lower())
    folder_path = _dataset_video_root(dataset_name)
    output_dir = "../Embeddings/Datasets"

    # Optional: enable TF32 for faster matmul on Ampere+
    if enable_tf32 and torch.cuda.is_available() and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load CLIP model & processor
    if clip_model in {"b32", "b16", "l14", "res50"}:
        model, processor = _load_clip_image_model(clip_model, device=device)
    elif clip_model == "clip4clip":
        model = CLIPVisionModelWithProjection.from_pretrained(
            "Searchium-ai/clip4clip-webvid150k"
        )
        model = model.eval()
        clip_full = CLIPModel.from_pretrained(
            "Searchium-ai/clip4clip-webvid150k"
        )
        model_text = CLIPTextModelWithProjection.from_pretrained(
            "Searchium-ai/clip4clip-webvid150k"
        )
        processor = CLIPTokenizer.from_pretrained(
            "Searchium-ai/clip4clip-webvid150k"
        )
    elif clip_model == "siglip":
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    elif clip_model == "siglipl14":
        model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    elif clip_model == "pe-l14":
        model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
        processor = pe_transformer.get_image_transform(model.image_size)
        tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
    elif clip_model == "pe-g14":
        model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True)
        processor = pe_transformer.get_image_transform(model.image_size)
        tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
        clip_model = "pe-l14"
    else:
        raise ValueError(f"Unknown CLIP model: {clip_model}")

    # Create embedder
    if (
        clip_model == "clip4clip"
        or clip_model == "siglip"
        or clip_model == "siglipl14"
        or clip_model == "res50"
        or clip_model == "pe-l14"
    ):
        embedder = VideoEmbedder(
            clip_model, model, processor,
            pe_video_batch_size=pe_video_batch_size,
            pe_target_T=pe_target_T,
            device=device,
        )
    else:
        embedder = VideoEmbedder(clip_model, model, processor, device=device)
    embedder.dataset_name = dataset_key

    print(embedd_path)
    embedder.process_data(
        folder_path,
        window_size=window_size,
        output_path=output_dir,
        random=random,
        save_intermediate=True,
        batch_size=batch_size,
    )
    os.makedirs(os.path.dirname(embedd_path), exist_ok=True)
    with open(embedd_path, "wb") as f:
        pickle.dump(embedder, f)
    
    return embedd_path

if __name__ == "__main__":

    process_dataset(
        "breakfast",
        "b32",
        window_size=96,
        random=True,
        batch_size=256,            # ignored for PE path except as upper bound
        pe_video_batch_size=24,    # try 8–16 depending on VRAM
        pe_target_T=8,             # uniformly sample each window to 8 frames
        enable_tf32=True,
        use_parallel=True,         # Enable multi-GPU parallelization
        num_gpus=1,                # Use 2 GPUs
    )
