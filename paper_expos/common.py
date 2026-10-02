"""Shared checkpoint and artifact helpers for the paper experiment suite."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PAPER_EXPOS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PAPER_EXPOS_ROOT.parent
DEFAULT_PROTOCOLS = PAPER_EXPOS_ROOT / "configs" / "main_protocols_128_v1.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "paper_expos"


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint: Path
    args_path: Path
    dataset_key: str
    dataset: str
    split: str
    seed: int
    concept_set: str
    run_name: str


def read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row})
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def canonical_dataset(value: object) -> str:
    lowered = str(value).strip().lower().replace("-", "_")
    if lowered == "breakfast":
        return "breakfast"
    if lowered in {"mpii", "mpii_cooking_2", "mpiicooking2"}:
        return "mpii"
    if lowered == "barista":
        return "barista"
    return lowered


def load_protocols(path: str | Path = DEFAULT_PROTOCOLS) -> dict[str, object]:
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
        raise ValueError(f"Invalid protocol configuration: {path}")
    return payload


def protocol_by_key(protocols: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {str(row["key"]): row for row in protocols["datasets"]}


def _matching_protocol(
    protocols: Mapping[str, object], dataset: object, split: object
) -> Mapping[str, object] | None:
    dataset_key = canonical_dataset(dataset)
    split = str(split)
    matches = [
        row
        for row in protocols["datasets"]
        if canonical_dataset(row.get("dataset")) == dataset_key
        and split in {str(value) for value in row.get("splits", [])}
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous protocol for dataset={dataset!r}, split={split!r}")
    return matches[0] if matches else None


def _record_for_checkpoint(
    checkpoint: Path,
    protocols: Mapping[str, object],
) -> CheckpointRecord | None:
    args_path = checkpoint.parent / "args.json"
    if not args_path.exists():
        return None
    args = read_json(args_path)
    if not isinstance(args, dict):
        return None
    protocol = _matching_protocol(protocols, args.get("dataset"), args.get("test_split"))
    if protocol is None:
        return None
    dataset_key = str(protocol["key"])
    seed = int(args.get("seed", -1))
    if seed not in {int(value) for value in protocols.get("seeds", [])}:
        return None
    split = str(args.get("test_split", ""))
    allowed_splits = {str(value) for value in protocol.get("splits", [])}
    if split not in allowed_splits:
        return None
    concept_set = str(args.get("concept_set", ""))
    if concept_set != str(protocol.get("concept_set")):
        return None
    return CheckpointRecord(
        checkpoint=checkpoint.resolve(),
        args_path=args_path.resolve(),
        dataset_key=dataset_key,
        dataset=str(args.get("dataset")),
        split=split,
        seed=seed,
        concept_set=concept_set,
        run_name=str(args.get("run_name", checkpoint.parent.name)),
    )


def discover_checkpoints(
    source_batch: str | Path,
    protocols_path: str | Path = DEFAULT_PROTOCOLS,
    *,
    require_complete_matrix: bool = True,
    allow_duplicate_runs: bool = False,
) -> list[CheckpointRecord]:
    batch = Path(source_batch).resolve()
    if not batch.is_dir():
        raise FileNotFoundError(f"Source batch directory does not exist: {batch}")
    protocols = load_protocols(protocols_path)
    candidates = sorted((batch / "models").glob("*/model.pt"))
    records = [record for path in candidates if (record := _record_for_checkpoint(path, protocols))]
    if allow_duplicate_runs:
        records = sorted(records, key=lambda row: (row.dataset_key, row.seed, row.run_name))
    else:
        unique: dict[tuple[str, int], CheckpointRecord] = {}
        for record in records:
            key = (record.dataset_key, record.seed)
            if key in unique:
                raise RuntimeError(
                    f"Duplicate checkpoint for {key}: {unique[key].checkpoint} and {record.checkpoint}"
                )
            unique[key] = record
        records = sorted(unique.values(), key=lambda row: (row.dataset_key, row.seed))
    if require_complete_matrix:
        expected = {
            (str(dataset["key"]), int(seed))
            for dataset in protocols["datasets"]
            for seed in protocols["seeds"]
        }
        actual = {(record.dataset_key, record.seed) for record in records}
        if actual != expected:
            raise RuntimeError(
                f"Expected {len(expected)} completed main-protocol checkpoints in {batch}; "
                f"found {len(actual)}. Missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    return records


def require_batch_complete(source_batch: str | Path) -> None:
    batch = Path(source_batch)
    summary_path = batch / "batch_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing batch_summary.json: {summary_path}")
    payload = read_json(summary_path)
    rows = payload.get("runs", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No runs found in {summary_path}")
    failed = [row for row in rows if int(row.get("returncode", row.get("return_code", 1))) != 0]
    if failed:
        raise RuntimeError(f"Batch has {len(failed)} failed runs: {summary_path}")


def output_dir(output_root: str | Path, experiment: str) -> Path:
    root = Path(output_root).resolve()
    destination = root / experiment
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return parsed


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(float(value) * math.log(max(float(value), 1e-12)) for value in probabilities)


def deterministic_sample(values: Sequence[object], count: int, seed: int) -> list[object]:
    values = list(values)
    if len(values) <= count:
        return values
    generator = random.Random(int(seed))
    return generator.sample(values, int(count))


def required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is unset: {name}")
    return value


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return float(math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1)))
