#!/usr/bin/env python3
"""Aggregate outputs produced by Slurm array tasks."""

from __future__ import annotations

import argparse
from pathlib import Path


def require_complete_array(
    experiment: str,
    output_dir: Path,
    expected_tasks: int | None = None,
) -> None:
    if experiment in {"long_horizon", "intervention_budget"}:
        files = [path for path in output_dir.glob("*_summary.csv") if path.name != "aggregate_summary.csv"]
        expected = 35 if expected_tasks is None else int(expected_tasks)
    elif experiment == "component_analysis":
        files = list(output_dir.glob("*_manifest.json"))
        expected = 21 if expected_tasks is None else int(expected_tasks)
    else:
        files = [path for path in output_dir.glob("*.json") if path.name != "summary.json"]
        expected = 35 if expected_tasks is None else int(expected_tasks)
    if len(files) != expected:
        raise RuntimeError(
            f"Refusing to aggregate incomplete {experiment} output: expected {expected} task artifacts, "
            f"found {len(files)} in {output_dir}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        choices=("long_horizon", "intervention_budget", "concept_robustness", "component_analysis"),
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-tasks", type=int, default=None)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    require_complete_array(args.experiment, args.output_dir, args.expected_tasks)

    if args.experiment == "long_horizon":
        from paper_expos.long_horizon import aggregate
    elif args.experiment == "intervention_budget":
        from paper_expos.intervention_budget import aggregate
    elif args.experiment == "component_analysis":
        from paper_expos.component_analysis import aggregate
    else:
        from paper_expos.concept_robustness import aggregate

    aggregate(args.output_dir)
    print(f"Aggregated {args.experiment} results in {args.output_dir}")


if __name__ == "__main__":
    main()
