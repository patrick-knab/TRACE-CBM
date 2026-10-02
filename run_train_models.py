from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "train_models.py"
COMBINED_LABEL_MODES = {
    "gtea_gaze": ("verb", "noun"),
    "mpii_cooking_2": ("action", "raw_action"),
}
AVERAGED_METRICS = ("accuracy", "macro_f1", "top3_accuracy")
BOOL_OPTIONAL_FLAGS = {
    "binary",
    "learn_concept_threshold",
    "generate_embeddings",
    "enable_tf32",
    "random_windows",
    "activity_only",
    "activity_class_weighting",
    "forecast_class_weighting",
    "graph_disabled_eval",
    "graph_corruption_eval",
}
PROGRESS_INTERVAL_SECONDS = 30.0
PROGRESS_PREFIXES = (
    "[train_models]",
    "[prepare_data]",
    "epoch=",
    "refit_epoch=",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple train_models.py jobs across a GPU batch.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "train_batch_config.json")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated visible GPU ids. Overrides config.gpus. Defaults to CUDA_VISIBLE_DEVICES or cuda count.",
    )
    parser.add_argument("--max-parallel", type=int, default=None, help="Override the parallel worker count.")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=None,
        help="Concurrent training processes assigned to each GPU.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching training.")
    return parser.parse_args()


def latest_progress_line(log_path: str | Path, max_bytes: int = 128 * 1024) -> str:
    """Return the latest useful training-status line from an active run log."""
    path = Path(log_path)
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return "waiting for log output"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith(PROGRESS_PREFIXES):
            return line
    return lines[-1] if lines else "waiting for log output"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    gpus = parse_gpus(args.gpus, config.get("gpus"))
    if not gpus:
        raise ValueError("No GPUs selected. Set config.gpus or pass --gpus, for example --gpus 0,1.")

    defaults = dict(config.get("defaults", {}))
    raw_runs = expand_runs(config)
    runs = expand_combined_runs(defaults, raw_runs)
    if not runs:
        raise ValueError("Batch config must contain at least one run in runs or matrix.")

    workers_per_gpu = args.workers_per_gpu or int(config.get("workers_per_gpu", 1))
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be at least 1.")
    gpu_slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    max_parallel = args.max_parallel or int(config.get("max_parallel", len(gpu_slots)))
    max_parallel = max(1, min(max_parallel, len(gpu_slots)))
    batch_root = Path(config.get("batch_output_dir", defaults.get("output_dir", PROJECT_ROOT / "runs" / "train_models")))
    batch_label = sanitize(str(config.get("batch_name", defaults.get("dataset", "dataset"))))
    batch_dir = batch_root / f"{batch_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = batch_dir / "logs"
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=False)

    queue = deque(enumerate(runs, start=1))
    free_gpus = deque(gpu_slots[:max_parallel])
    active: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    last_progress_report = time.monotonic()

    print(
        f"[run_train_models] Loaded {len(runs)} run(s); using GPUs {gpus} "
        f"with workers_per_gpu={workers_per_gpu}, max_parallel={max_parallel}",
        flush=True,
    )

    while queue or active:
        while queue and free_gpus:
            run_index, run_overrides = queue.popleft()
            gpu = free_gpus.popleft()
            run_config = merge_run_config(defaults, run_overrides)
            run_name = str(run_config.get("run_name") or run_config.get("name") or f"run_{run_index:03d}")
            run_config["run_name"] = run_name
            run_config["output_dir"] = str(batch_dir / "models")
            metadata = run_result_metadata(run_config)
            command = build_command(run_config)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("WANDB_RUN_GROUP", batch_dir.name)
            env.setdefault("PYTHONUNBUFFERED", "1")
            if "device" not in run_config or str(run_config["device"]) == "auto":
                command.extend(["--device", "cuda:0"])

            log_path = log_dir / f"{run_index:03d}_{sanitize(run_name)}_gpu{sanitize(str(gpu))}.log"
            if args.dry_run:
                print(f"[dry-run] GPU {gpu}: {' '.join(command)}", flush=True)
                free_gpus.append(gpu)
                results.append({"run": run_name, "gpu": gpu, "returncode": 0, "dry_run": True, **metadata})
                continue

            log_handle = log_path.open("w", encoding="utf-8")
            print(f"[run_train_models] Starting {run_name} on GPU {gpu}; log={log_path}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append(
                {
                    "run": run_name,
                    "gpu": gpu,
                    "process": process,
                    "log_handle": log_handle,
                    "log_path": str(log_path),
                    "command": command,
                    "started_at": time.time(),
                    "metadata": metadata,
                }
            )

        if args.dry_run:
            continue

        time.sleep(5)
        still_active = []
        for job in active:
            returncode = job["process"].poll()
            if returncode is None:
                still_active.append(job)
                continue
            job["log_handle"].close()
            elapsed = time.time() - float(job["started_at"])
            result = {
                "run": job["run"],
                "gpu": job["gpu"],
                "returncode": int(returncode),
                "elapsed_seconds": elapsed,
                "log_path": job["log_path"],
                "command": job["command"],
                **job["metadata"],
            }
            results.append(result)
            status = "finished" if returncode == 0 else "failed"
            print(
                f"[run_train_models] {job['run']} {status} on GPU {job['gpu']} "
                f"with returncode={returncode}; elapsed={elapsed:.1f}s",
                flush=True,
            )
            free_gpus.append(job["gpu"])
        active = still_active

        now = time.monotonic()
        if active and now - last_progress_report >= PROGRESS_INTERVAL_SECONDS:
            for job in active:
                elapsed = time.time() - float(job["started_at"])
                progress = latest_progress_line(job["log_path"])
                print(
                    f"[run_train_models] Progress {job['run']} on GPU {job['gpu']} "
                    f"elapsed={elapsed:.0f}s: {progress}",
                    flush=True,
                )
            last_progress_report = now

    if not args.dry_run:
        write_json(
            batch_dir / "batch_config.resolved.json",
            resolved_batch_config(gpus, workers_per_gpu, defaults, runs),
        )
        write_json(batch_dir / "batch_summary.json", results)

    failures = [result for result in results if int(result["returncode"]) != 0]
    if failures:
        failed = ", ".join(str(result["run"]) for result in failures)
        raise SystemExit(f"[run_train_models] Failed run(s): {failed}")

    if not args.dry_run:
        combined_metrics = build_combined_metrics(batch_dir, runs, results)
        if combined_metrics["groups"]:
            write_json(batch_dir / "combined_metrics.json", combined_metrics)

    print(f"[run_train_models] All runs completed. Batch dir: {batch_dir}", flush=True)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Batch config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Batch config must be a JSON object.")
    return config


def parse_gpus(cli_value: str | None, config_value: Any) -> list[str]:
    if cli_value:
        raw = cli_value
    elif config_value is not None:
        if isinstance(config_value, list):
            return [str(value) for value in config_value]
        raw = str(config_value)
    else:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]

    try:
        import torch
    except ImportError:
        return []
    return [str(index) for index in range(torch.cuda.device_count())]


def expand_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    runs = config.get("runs", [])
    if not isinstance(runs, list):
        raise TypeError("config.runs must be a list.")
    expanded = [dict(run) for run in runs]

    matrix = config.get("matrix")
    if matrix:
        if not isinstance(matrix, dict):
            raise TypeError("config.matrix must be an object of key -> list values.")
        keys = list(matrix)
        value_lists = []
        for key in keys:
            values = matrix[key]
            if not isinstance(values, list) or not values:
                raise TypeError(f"config.matrix.{key} must be a non-empty list.")
            value_lists.append(values)
        for values in itertools.product(*value_lists):
            expanded.append(dict(zip(keys, values)))

    seeds = config.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, list) or not seeds:
            raise TypeError("config.seeds must be a non-empty list.")
        seeded_runs = []
        for run in expanded:
            for seed in seeds:
                seeded_run = dict(run)
                seeded_run["seed"] = int(seed)
                for name_key in ("name", "run_name"):
                    if name_key in seeded_run:
                        seeded_run[name_key] = str(seeded_run[name_key]).format(seed=int(seed))
                seeded_runs.append(seeded_run)
        expanded = seeded_runs
    return expanded


def expand_combined_runs(defaults: dict[str, Any], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for index, overrides in enumerate(runs, start=1):
        run_config = merge_run_config(defaults, overrides)
        label_mode = str(run_config.get("activity_label_mode", "action")).strip().lower().replace("-", "_")
        if label_mode != "combined":
            expanded.append(dict(overrides))
            continue

        dataset_key = canonical_dataset_key(run_config.get("dataset", defaults.get("dataset", "")))
        child_modes = COMBINED_LABEL_MODES.get(dataset_key)
        if child_modes is None:
            dataset_name = str(run_config.get("dataset", defaults.get("dataset", "")))
            print(
                f"[run_train_models] Warning: activity_label_mode=combined has no decomposition "
                f"for dataset={dataset_name!r}; falling back to action.",
                flush=True,
            )
            child_modes = ("action",)

        group_name = str(run_config.get("run_name") or run_config.get("name") or f"run_{index:03d}")
        for child_mode in child_modes:
            child = dict(overrides)
            child_name = f"{group_name}_{child_mode}"
            child["name"] = child_name
            child["run_name"] = child_name
            child["activity_label_mode"] = child_mode
            child["_combined_group_name"] = group_name
            child["_combined_label_mode"] = child_mode
            child["_combined_dataset_key"] = dataset_key
            expanded.append(child)
    return expanded


def merge_run_config(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = {**defaults, **overrides}
    if isinstance(defaults.get("model_hparams"), dict) or isinstance(overrides.get("model_hparams"), dict):
        merged["model_hparams"] = {
            **dict(defaults.get("model_hparams") or {}),
            **dict(overrides.get("model_hparams") or {}),
        }
    return merged


def build_command(run_config: dict[str, Any]) -> list[str]:
    command = [sys.executable, str(TRAIN_SCRIPT)]
    for key, value in sorted(run_config.items()):
        if key == "name" or key.startswith("_") or value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if key == "no_wandb":
            if bool(value):
                command.append("--no-wandb")
            continue
        if key in BOOL_OPTIONAL_FLAGS:
            command.append(flag if bool(value) else f"--no-{key.replace('_', '-')}")
            continue
        if key in {"model_hparams", "dataset_hparams", "forecast_horizon_loss_weights"}:
            value = json.dumps(value, sort_keys=True)
        command.extend([flag, str(value)])
    return command


def resolved_batch_config(
    gpus: list[str],
    workers_per_gpu: int,
    defaults: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "gpus": gpus,
        "workers_per_gpu": workers_per_gpu,
        "defaults": defaults,
        "runs": [merge_run_config(defaults, run) for run in runs],
    }


def build_combined_metrics(
    batch_dir: Path,
    runs: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    successful_runs = {str(result["run"]) for result in results if int(result["returncode"]) == 0}
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        group_name = run.get("_combined_group_name")
        if group_name is None:
            continue
        groups.setdefault(str(group_name), []).append(run)

    payload_groups = []
    for group_name, child_runs in sorted(groups.items()):
        children = []
        child_metrics = []
        for run in child_runs:
            run_name = str(run["run_name"])
            if run_name not in successful_runs:
                continue
            metrics_path = find_metrics_path(batch_dir, run_name)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            child_metrics.append(metrics)
            children.append(
                {
                    "run": run_name,
                    "activity_label_mode": str(run["_combined_label_mode"]),
                    "metrics_path": str(metrics_path),
                }
            )
        if len(children) != len(child_runs):
            continue
        payload_groups.append(
            {
                "combined_group": group_name,
                "dataset_key": str(child_runs[0].get("_combined_dataset_key", "")),
                "child_label_modes": [str(run["_combined_label_mode"]) for run in child_runs],
                "children": children,
                "averaged_metrics": average_metric_payloads(child_metrics),
            }
        )
    return {"metric_average": "equal_mean", "groups": payload_groups}


def average_metric_payloads(metric_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    averaged: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_payloads = [
            payload.get(split)
            for payload in metric_payloads
            if isinstance(payload.get(split), dict)
        ]
        if len(split_payloads) != len(metric_payloads):
            continue
        split_average: dict[str, Any] = {}
        task_names = sorted(set.intersection(*(set(payload) for payload in split_payloads)))
        for task_name in task_names:
            task_payloads = [
                payload.get(task_name)
                for payload in split_payloads
                if isinstance(payload.get(task_name), dict)
            ]
            if len(task_payloads) != len(metric_payloads):
                continue
            metric_average = {}
            for metric_name in AVERAGED_METRICS:
                values = [payload.get(metric_name) for payload in task_payloads]
                if all(isinstance(value, (int, float)) for value in values):
                    metric_average[metric_name] = sum(float(value) for value in values) / len(values)
            if metric_average:
                split_average[task_name] = metric_average
        if split_average:
            averaged[split] = split_average
    return averaged


def find_metrics_path(batch_dir: Path, run_name: str) -> Path:
    matches = []
    for args_path in sorted((batch_dir / "models").glob("*/args.json")):
        try:
            run_args = json.loads(args_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(run_args.get("run_name")) == run_name:
            metrics_path = args_path.parent / "metrics.json"
            if metrics_path.exists():
                matches.append(metrics_path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one metrics.json for run_name={run_name!r} in {batch_dir}, found {len(matches)}."
        )
    return matches[0]


def run_result_metadata(run_config: dict[str, Any]) -> dict[str, str]:
    metadata = {}
    if run_config.get("_combined_group_name") is not None:
        metadata["combined_group"] = str(run_config["_combined_group_name"])
    if run_config.get("_combined_label_mode") is not None:
        metadata["combined_label_mode"] = str(run_config["_combined_label_mode"])
    if run_config.get("_combined_dataset_key") is not None:
        metadata["combined_dataset_key"] = str(run_config["_combined_dataset_key"])
    return metadata


def canonical_dataset_key(dataset: object) -> str:
    key = str(dataset).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "gtea": "gtea_gaze",
        "egtea": "gtea_gaze",
        "egtea_gaze": "gtea_gaze",
        "gtea_gaze": "gtea_gaze",
        "mpii": "mpii_cooking_2",
        "mpii_cooking": "mpii_cooking_2",
        "mpii_cooking_2": "mpii_cooking_2",
        "breakfast": "breakfast",
        "barista": "barista",
    }
    return aliases.get(key, key)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sanitize(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    main()
