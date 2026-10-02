#!/usr/bin/env python3
"""Log a completed paper-analysis directory to the TRACE W&B project."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def scalar_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            metrics.update(scalar_metrics(child, f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        metrics[prefix] = float(value)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", default="TRACE-ICLR2027")
    parser.add_argument("--mode", default="online", choices=("online", "offline", "disabled"))
    args = parser.parse_args()
    directory = args.path.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    import wandb

    run = wandb.init(project=args.project, name=args.name, job_type="paper-analysis", mode=args.mode)
    artifact = wandb.Artifact(args.name, type="paper-analysis")
    artifact.add_dir(str(directory))
    run.log_artifact(artifact)
    for filename in ("summary.json", "ablation_summary.json", "synthetic_summary.json"):
        for path in directory.rglob(filename):
            try:
                run.log(scalar_metrics(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError):
                pass
    run.finish()


if __name__ == "__main__":
    main()
