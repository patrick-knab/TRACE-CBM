"""Matched concept-coordinate interventions for paired graph/dense rollouts.

This reuses the Sparse Linear evaluator's case selection, policy definitions,
metrics, and recursive intervention protocol, changing only checkpoint
discovery and the rollout family. It deliberately evaluates coordinate edits
only so graph and dense checkpoints share exactly the same intervention hook.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from paper_expos.sparse_linear_intervention import (
    POLICIES,
    SEEDS,
    Record,
    intervention_rows,
    load_workspace,
    select_cases,
    summarize,
    write_rows,
)


DENSE_TARGETS = {
    ("barista", "s1"): "barista",
    ("breakfast", "s1"): "breakfast_s1",
    ("mpii_cooking_2", "attr"): "mpii_attr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--concept-root", type=Path, required=True)
    parser.add_argument("--rollout-mode", choices=("controlled_graph", "controlled_dense"), default="controlled_dense")
    parser.add_argument("--expected-concepts", type=int, default=0)
    parser.add_argument("--case-manifest", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--record-index", type=int, default=None)
    parser.add_argument("--budgets", default="1,2,3,4,5")
    parser.add_argument("--errors", type=int, default=30)
    parser.add_argument("--correct", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    return parser.parse_args()


def discover_records(source_batch: Path, requested_rollout_mode: str) -> list[Record]:
    records = []
    for checkpoint in sorted((source_batch / "models").glob("*/model.pt")):
        args_path = checkpoint.parent / "args.json"
        if not args_path.exists():
            continue
        args = json.loads(args_path.read_text())
        dataset = str(args.get("dataset", "")).strip().lower()
        split = str(args.get("test_split", ""))
        dataset_key = DENSE_TARGETS.get((dataset, split))
        checkpoint_rollout_mode = str(args.get("model_hparams", {}).get("st_forecast_rollout_mode", ""))
        concept_set = str(args.get("concept_set", ""))
        if (
            dataset_key is not None
            and int(args.get("seed", -1)) in SEEDS
            and args.get("base_method") in {"trace", "graph_cbm", "concept_forecast_cbm"}
            and checkpoint_rollout_mode == requested_rollout_mode
            and concept_set.endswith("_visual_states_no_labels_v4_1")
        ):
            records.append(
                Record(
                    checkpoint=checkpoint.resolve(),
                    dataset=dataset,
                    split=split,
                    dataset_key=dataset_key,
                    seed=int(args["seed"]),
                    run_name=str(args.get("run_name", checkpoint.parent.name)),
                )
            )
    unique = {(record.dataset_key, record.seed): record for record in records}
    expected = {(dataset_key, seed) for dataset_key in DENSE_TARGETS.values() for seed in SEEDS}
    if set(unique) != expected:
        raise RuntimeError(
            f"Expected 9 matched {requested_rollout_mode} checkpoints; "
            f"missing={sorted(expected - set(unique))}, extra={sorted(set(unique) - expected)}"
        )
    return sorted(unique.values(), key=lambda row: (row.dataset_key, row.seed))


def run_record(record: Record, args: argparse.Namespace):
    os.environ["TRACE_CONCEPT_ROOT"] = str(args.concept_root)
    workspace = load_workspace(record.checkpoint, device=args.device, dataset_root=args.dataset_root)
    if args.expected_concepts and len(workspace.concept_names) != args.expected_concepts:
        raise RuntimeError(f"Expected {args.expected_concepts} concepts, found {len(workspace.concept_names)}")
    rows = intervention_rows(workspace, record, args)
    fixed_keys = None
    if args.case_manifest is not None:
        manifest = json.loads(args.case_manifest.read_text(encoding="utf-8"))
        fixed_keys = [tuple(map(int, pair)) for pair in manifest[record.dataset_key][str(record.seed)]]
    cases = select_cases(workspace, args.errors, args.correct, args.stride, record.seed, fixed_keys=fixed_keys)
    case_keys = [[int(case["instance"]["video_index"]), int(case["instance"]["timestep"])] for case in cases]
    return rows, summarize(rows), len(workspace.concept_names), case_keys


def main() -> None:
    args = parse_args()
    records = discover_records(args.source_batch, args.rollout_mode)
    selected = records if args.record_index is None else [records[int(args.record_index)]]
    all_rows = []
    all_summaries = []
    manifests = []
    case_manifest = {}
    for record in selected:
        print(f"[dense-intervention] {record.dataset_key} seed={record.seed} checkpoint={record.checkpoint}", flush=True)
        rows, summaries, concept_count, case_keys = run_record(record, args)
        all_rows.extend(rows)
        all_summaries.extend(summaries)
        manifests.append({"dataset": record.dataset_key, "seed": record.seed, "concept_count": concept_count, "checkpoint": str(record.checkpoint), "run_name": record.run_name})
        case_manifest.setdefault(record.dataset_key, {})[str(record.seed)] = case_keys
        print(f"[dense-intervention] rows={len(rows)} summaries={len(summaries)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "intervention_rows.csv", all_rows)
    write_rows(args.output_dir / "intervention_summary.csv", all_summaries)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "observed-coordinate clamp to the true t+1 standardized concept target, followed by recursive H1-H3 rollout",
                "rollout_mode": args.rollout_mode,
                "policies": list(POLICIES),
                "budgets": [int(value) for value in args.budgets.split(",") if value.strip()],
                "errors": args.errors,
                "correct": args.correct,
                "stride": args.stride,
                "records": manifests,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "case_manifest.json").write_text(json.dumps(case_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[dense-intervention] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
