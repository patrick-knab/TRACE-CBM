from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests/fixtures/trace_cpu_smoke_expected.json"


def nested_value(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for key in dotted_path.split("."):
        value = value[key]
    return value


def main() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    run = fixture["run"]

    with tempfile.TemporaryDirectory(prefix="trace_cpu_smoke_") as temporary_dir:
        output_root = Path(temporary_dir) / "runs"
        command = [sys.executable, str(PROJECT_ROOT / "train_models.py")]
        for key, value in run.items():
            command.append(f"--{key.replace('_', '-')}")
            command.append(
                json.dumps(value, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else str(value)
            )
        command.extend(
            [
                "--no-wandb",
                "--output-dir",
                str(output_root),
                "--run-name",
                "cpu_smoke",
            ]
        )

        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["WANDB_MODE"] = "disabled"
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)

        run_dirs = list(output_root.iterdir())
        if len(run_dirs) != 1 or not run_dirs[0].is_dir():
            raise RuntimeError(f"Expected one completed run directory, found {run_dirs}.")
        run_dir = run_dirs[0]

        for filename in fixture["required_artifacts"]:
            artifact = run_dir / filename
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty smoke-test artifact: {artifact.name}")

        args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
        for key, expected in fixture["expected_args"].items():
            if args.get(key) != expected:
                raise RuntimeError(f"args.json {key!r}: expected {expected!r}, got {args.get(key)!r}.")
        for split, expected_count in fixture["expected_splits"].items():
            key = f"{split}_sequences"
            actual_count = args["dataset_hparams"].get(key)
            if actual_count != expected_count:
                raise RuntimeError(
                    f"args.json dataset_hparams[{key!r}]: expected {expected_count}, got {actual_count}."
                )

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        for dotted_path in fixture["required_metric_paths"]:
            value = nested_value(metrics, dotted_path)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"Metric {dotted_path!r} is not a finite number: {value!r}.")
            if not 0.0 <= float(value) <= 1.0:
                raise RuntimeError(f"Accuracy metric {dotted_path!r} is outside [0, 1]: {value!r}.")

        print(
            "PASS: TRACE CPU smoke completed one synthetic epoch; "
            "checkpoint, metrics, summary, and history are valid."
        )


if __name__ == "__main__":
    main()
