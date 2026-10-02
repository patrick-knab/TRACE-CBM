"""Measure learned-edge stability across seeds and export consensus graph artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from paper_expos.common import DEFAULT_OUTPUT_ROOT, DEFAULT_PROTOCOLS, discover_checkpoints, mean, write_csv, write_json
from utils.graph_concept_ui import load_workspace
from utils.model import _active_graph_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "graph_stability")
    parser.add_argument("--max-edges", type=int, default=5000)
    parser.add_argument("--consensus-min-seeds", type=int, default=2)
    parser.add_argument("--consensus-max-edges", type=int, default=80)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def edge_key(edge: Mapping[str, object]) -> tuple[object, ...]:
    return (
        str(edge["branch"]),
        int(edge["layer"]),
        str(edge["kind"]),
        int(edge["source"]),
        int(edge["target"]),
    )


def dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def consensus_dot(
    path: Path,
    concept_names: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    node_ids = sorted({int(row["source"]) for row in rows} | {int(row["target"]) for row in rows})
    lines = ["digraph ConsensusGraph {", "  rankdir=LR;", '  node [shape=box, style="rounded,filled", fillcolor="#f6f8fa"];']
    for concept_idx in node_ids:
        label = concept_names[concept_idx] if concept_idx < len(concept_names) else str(concept_idx)
        lines.append(f'  c{concept_idx} [label="{dot_escape(label)}"];')
    for row in rows:
        color = "#2166ac" if float(row["mean_signed_weight"]) >= 0 else "#b2182b"
        style = "solid" if str(row["kind"]) == "spatial" else "dashed"
        label = f"{row['kind']} | {int(row['seed_count'])} seeds | {float(row['mean_signed_weight']):+.3f}"
        lines.append(
            f'  c{int(row["source"])} -> c{int(row["target"])} '
            f'[color="{color}", style="{style}", label="{dot_escape(label)}"];'
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = discover_checkpoints(args.source_batch, args.protocols)
    if args.dry_run:
        print(json.dumps({"records": [str(row.checkpoint) for row in records]}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    by_dataset: dict[str, list[tuple[object, list[dict[str, object]], list[str]]]] = defaultdict(list)
    all_rows: list[dict[str, object]] = []
    for record in records:
        workspace = load_workspace(record.checkpoint, device=args.device)
        edges = [dict(edge) for edge in _active_graph_edges(workspace.model, args.max_edges)]
        for edge in edges:
            all_rows.append(
                {
                    "dataset": record.dataset_key,
                    "seed": record.seed,
                    "branch": edge["branch"],
                    "layer": edge["layer"],
                    "kind": edge["kind"],
                    "source": edge["source"],
                    "source_name": workspace.concept_names[int(edge["source"])],
                    "target": edge["target"],
                    "target_name": workspace.concept_names[int(edge["target"])],
                    "absolute_weight": edge["score"],
                    "sign": edge["sign"],
                    "signed_weight": float(edge["score"]) * float(edge["sign"]),
                }
            )
        by_dataset[record.dataset_key].append((record, edges, list(workspace.concept_names)))
    write_csv(args.output_dir / "all_active_edges.csv", all_rows)

    pair_rows: list[dict[str, object]] = []
    consensus_rows: list[dict[str, object]] = []
    for dataset, bundles in sorted(by_dataset.items()):
        for (record_a, edges_a, _), (record_b, edges_b, _) in itertools.combinations(bundles, 2):
            map_a = {edge_key(edge): edge for edge in edges_a}
            map_b = {edge_key(edge): edge for edge in edges_b}
            keys_a, keys_b = set(map_a), set(map_b)
            overlap = keys_a & keys_b
            union = keys_a | keys_b
            pair_rows.append(
                {
                    "dataset": dataset,
                    "seed_a": record_a.seed,
                    "seed_b": record_b.seed,
                    "edges_a": len(keys_a),
                    "edges_b": len(keys_b),
                    "overlap": len(overlap),
                    "jaccard": len(overlap) / max(len(union), 1),
                    "sign_agreement": mean(
                        float(map_a[key]["sign"] == map_b[key]["sign"])
                        for key in overlap
                    ),
                }
            )

        grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
        concept_names = bundles[0][2]
        for _, edges, _ in bundles:
            for edge in edges:
                grouped[edge_key(edge)].append(edge)
        dataset_consensus = []
        for key, selected in grouped.items():
            seed_count = len(selected)
            if seed_count < int(args.consensus_min_seeds):
                continue
            branch, layer, kind, source, target = key
            signed = [float(edge["score"]) * float(edge["sign"]) for edge in selected]
            row = {
                "dataset": dataset,
                "branch": branch,
                "layer": layer,
                "kind": kind,
                "source": source,
                "source_name": concept_names[int(source)],
                "target": target,
                "target_name": concept_names[int(target)],
                "seed_count": seed_count,
                "sign_agreement": max(Counter(float(edge["sign"]) for edge in selected).values()) / seed_count,
                "mean_absolute_weight": mean(float(edge["score"]) for edge in selected),
                "mean_signed_weight": mean(signed),
            }
            consensus_rows.append(row)
            dataset_consensus.append(row)
        dataset_consensus.sort(
            key=lambda row: (int(row["seed_count"]), float(row["mean_absolute_weight"])),
            reverse=True,
        )
        consensus_dot(
            args.output_dir / f"{dataset}_consensus.dot",
            concept_names,
            dataset_consensus[: int(args.consensus_max_edges)],
        )
    write_csv(args.output_dir / "pairwise_seed_stability.csv", pair_rows)
    write_csv(args.output_dir / "consensus_edges.csv", consensus_rows)
    write_json(
        args.output_dir / "summary.json",
        {
            "source_batch": str(args.source_batch.resolve()),
            "datasets": sorted(by_dataset),
            "pairwise_mean_jaccard": {
                dataset: mean(float(row["jaccard"]) for row in pair_rows if row["dataset"] == dataset)
                for dataset in by_dataset
            },
            "pairwise_mean_sign_agreement": {
                dataset: mean(float(row["sign_agreement"]) for row in pair_rows if row["dataset"] == dataset)
                for dataset in by_dataset
            },
            "claim_boundary": "edge stability supports reproducibility of learned relations, not causal graph recovery",
        },
    )


if __name__ == "__main__":
    main()
