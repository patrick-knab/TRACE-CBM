"""Export fixed success, harm, and no-flip intervention examples with concept trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch

from paper_expos.common import DEFAULT_OUTPUT_ROOT, DEFAULT_PROTOCOLS, discover_checkpoints, write_json
from utils.graph_concept_ui import forward_outputs, load_workspace
from utils.intervention_notebook import _forecast_logits_by_horizon, select_instance
from utils.model import _active_graph_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--intervention-dir", type=Path, required=True)
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "exemplars")
    parser.add_argument("--max-concepts", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("*_rows.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def select_rows(rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    selected = []
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset and row["intervention_type"] == "concept"]
        successes = [row for row in subset if int(row["wrong_to_correct"]) == 1]
        harms = [row for row in subset if int(row["correct_to_wrong"]) == 1]
        stable = [
            row for row in subset
            if row["treatment"] != "no_op" and int(row["label_flip"]) == 0 and int(row["budget"]) >= 3
        ]
        if successes:
            selected.append(("successful_correction", max(successes, key=lambda row: float(row["true_probability_delta"]))))
        if harms:
            selected.append(("harmful_intervention", min(harms, key=lambda row: float(row["true_probability_delta"]))))
        if stable:
            selected.append(("no_label_flip", max(stable, key=lambda row: abs(float(row["future_concept_l1"])))))
    return selected


def probabilities(workspace, outputs: Mapping[str, object], horizon: int) -> np.ndarray:
    logits = _forecast_logits_by_horizon(workspace, outputs)[int(horizon)][0, -1]
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()


def plot_trajectory(path: Path, names: list[str], before: np.ndarray, after: np.ndarray) -> None:
    horizons = np.arange(1, before.shape[0] + 1)
    columns = 2
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10, max(3, rows * 2.5)), squeeze=False)
    for concept_index, name in enumerate(names):
        axis = axes.flat[concept_index]
        axis.plot(horizons, before[:, concept_index], marker="o", label="baseline")
        axis.plot(horizons, after[:, concept_index], marker="s", label="intervened")
        axis.set_title(name)
        axis.set_xlabel("forecast step")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(names) :]:
        axis.axis("off")
    axes.flat[0].legend()
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    records = discover_checkpoints(args.source_batch, args.protocols)
    record_map = {(record.dataset_key, record.seed): record for record in records}
    rows = read_rows(args.intervention_dir)
    choices = select_rows(rows)
    if args.dry_run:
        print(json.dumps({"candidate_rows": len(rows), "selected": [(kind, row["dataset"], row["seed"]) for kind, row in choices]}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = []
    for sample_index, (sample_kind, row) in enumerate(choices):
        record = record_map[(row["dataset"], int(row["seed"]))]
        workspace = load_workspace(record.checkpoint, device=args.device)
        instance = select_instance(
            workspace,
            "test",
            int(row["video_index"]),
            int(row["timestep"]),
        )
        baseline = forward_outputs(workspace, instance)
        payload = json.loads(row["payload"])
        intervened = forward_outputs(workspace, instance, intervention=payload)
        horizons = sorted(set(baseline.get("predicted_concepts_by_step", {})) & set(intervened.get("predicted_concepts_by_step", {})))
        before_all = np.stack([baseline["predicted_concepts_by_step"][h][0, -1].detach().cpu().numpy() for h in horizons])
        after_all = np.stack([intervened["predicted_concepts_by_step"][h][0, -1].detach().cpu().numpy() for h in horizons])
        changed = np.max(np.abs(after_all - before_all), axis=0)
        concept_indices = np.argsort(-changed)[: int(args.max_concepts)].tolist()
        names = [workspace.concept_names[index] for index in concept_indices]
        before = before_all[:, concept_indices]
        after = after_all[:, concept_indices]
        stem = f"{sample_index:02d}_{row['dataset']}_{sample_kind}"
        plot_trajectory(args.output_dir / stem, names, before, after)
        edges = []
        selected_set = set(concept_indices)
        for edge in _active_graph_edges(workspace.model, max_edges=500):
            if int(edge["source"]) in selected_set or int(edge["target"]) in selected_set:
                edges.append(
                    {
                        **edge,
                        "source_name": workspace.concept_names[int(edge["source"])],
                        "target_name": workspace.concept_names[int(edge["target"])],
                    }
                )
            if len(edges) >= 30:
                break
        horizon = max(workspace.forecast_horizons)
        before_prob = probabilities(workspace, baseline, horizon)
        after_prob = probabilities(workspace, intervened, horizon)
        artifact = {
            "selection_category": sample_kind,
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "checkpoint": str(record.checkpoint),
            "video_id": instance["video_id"],
            "video_path": instance["video_path"],
            "video_index": instance["video_index"],
            "timestep": instance["timestep"],
            "intervention_type": row["intervention_type"],
            "policy": row["policy"],
            "treatment": row["treatment"],
            "budget": int(row["budget"]),
            "payload": payload,
            "true_label": int(row["true_label"]),
            "prediction_before": int(before_prob.argmax()),
            "prediction_after": int(after_prob.argmax()),
            "true_probability_before": float(before_prob[int(row["true_label"])]),
            "true_probability_after": float(after_prob[int(row["true_label"])]),
            "concept_names": names,
            "concept_indices": concept_indices,
            "horizons": horizons,
            "concept_trajectory_before": before.tolist(),
            "concept_trajectory_after": after.tolist(),
            "relevant_edges": edges,
            "trajectory_pdf": f"{stem}.pdf",
            "trajectory_png": f"{stem}.png",
        }
        write_json(args.output_dir / f"{stem}.json", artifact)
        manifest.append(artifact)
    write_json(
        args.output_dir / "manifest.json",
        {
            "selection_rule": "largest successful correction, largest harmful effect, and largest non-flipping concept change per dataset",
            "samples": manifest,
        },
    )


if __name__ == "__main__":
    main()
