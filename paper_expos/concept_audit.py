"""Audit 128_v1 concept uniqueness, support, prevalence, and target-label proximity."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from paper_expos.common import DEFAULT_OUTPUT_ROOT, DEFAULT_PROTOCOLS, deterministic_sample, discover_checkpoints, mean, write_csv, write_json
from utils.graph_concept_ui import load_workspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--concept-generation-root", type=Path, default=Path("runs/concept_generation"))
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "concept_audit")
    parser.add_argument("--human-audit-concepts", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower().replace("sil", " ")))


def token_jaccard(left: str, right: str) -> float:
    a, b = set(normalize(left).split()), set(normalize(right).split())
    return len(a & b) / max(len(a | b), 1)


def load_concepts(name: str) -> list[str]:
    path = Path("concepts") / f"{name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    concepts = payload.get("concepts", {})
    values = concepts.get(name) if isinstance(concepts, dict) else None
    if not isinstance(values, list):
        raise ValueError(f"Could not resolve concept set {name} in {path}")
    return [str(value) for value in values]


def matching_generator_audit(root: Path, concepts: list[str]) -> tuple[Path | None, list[dict[str, object]]]:
    target = set(concepts)
    matches = []
    for path in root.glob("**/final_audit.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        texts = {str(row.get("text")) for row in payload if isinstance(row, dict)}
        overlap = len(target & texts)
        if overlap:
            matches.append((overlap, path.stat().st_mtime, path, payload))
    if not matches:
        return None, []
    _, _, path, payload = max(matches, key=lambda item: (item[0], item[1]))
    return path, [dict(row) for row in payload if str(row.get("text")) in target]


def activation_statistics(workspace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = workspace.standardized_splits["test"]
    key = "concepts_std" if "concepts_std" in split else "concepts"
    values = torch.as_tensor(split[key], dtype=torch.float32, device=workspace.device)
    valid = torch.as_tensor(split["mask"] > 0.0, dtype=torch.bool, device=workspace.device)
    calibrator = getattr(workspace.model, "calibrator", None)
    flattened = values.reshape(-1, values.shape[-1])
    with torch.no_grad():
        calibrated = calibrator(flattened) if callable(calibrator) else flattened
    calibrated = calibrated.reshape(values.shape).detach().cpu().numpy()
    selected = calibrated[np.asarray(valid.detach().cpu())]
    return selected.mean(axis=0), selected.std(axis=0), (selected >= 0.5).mean(axis=0)


def main() -> None:
    args = parse_args()
    records = discover_checkpoints(args.source_batch, args.protocols)
    first_records = {}
    for record in records:
        first_records.setdefault(record.dataset_key, record)
    if args.dry_run:
        print(json.dumps({key: str(record.checkpoint) for key, record in first_records.items()}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    concept_rows = []
    human_rows = []
    dataset_summary = {}
    for dataset, record in sorted(first_records.items()):
        concepts = load_concepts(record.concept_set)
        workspace = load_workspace(record.checkpoint, device=args.device)
        means, stds, prevalence = activation_statistics(workspace)
        audit_path, audit_rows = matching_generator_audit(args.concept_generation_root, concepts)
        audit_by_text = {str(row.get("text")): row for row in audit_rows}
        labels = [name for name in workspace.activity_names if normalize(name)]
        for index, concept in enumerate(concepts):
            nearest_label = max(labels, key=lambda label: token_jaccard(concept, label), default="")
            lexical_similarity = token_jaccard(concept, nearest_label) if nearest_label else 0.0
            audit = audit_by_text.get(concept, {})
            concept_rows.append(
                {
                    "dataset": dataset,
                    "concept_index": index,
                    "concept": concept,
                    "normalized": normalize(concept),
                    "token_count": len(normalize(concept).split()),
                    "activation_mean": float(means[index]),
                    "activation_std": float(stds[index]),
                    "activation_prevalence_ge_0p5": float(prevalence[index]),
                    "nearest_activity_label": nearest_label,
                    "lexical_label_jaccard": lexical_similarity,
                    "generator_support_score": audit.get("support_score"),
                    "generator_verification_confidence": audit.get("verification_confidence"),
                    "generator_verified_evidence_count": len(audit.get("verified_evidence", [])),
                    "generator_nearest_target": audit.get("nearest_target"),
                    "generator_target_similarity": audit.get("target_similarity"),
                    "generator_semantic_target_warning": bool(audit.get("semantic_target_warning", False)),
                    "generator_audit_path": str(audit_path) if audit_path else "",
                }
            )
        sampled = deterministic_sample(concepts, int(args.human_audit_concepts), 20260803 + record.seed)
        for concept in sampled:
            audit = audit_by_text.get(str(concept), {})
            human_rows.append(
                {
                    "dataset": dataset,
                    "concept": concept,
                    "retrieved_evidence_ids": json.dumps(audit.get("retrieved_evidence", [])),
                    "verified_evidence_ids": json.dumps(audit.get("verified_evidence", [])),
                    "visually_supported_0_or_1": "",
                    "clear_and_reusable_0_or_1": "",
                    "activity_label_leakage_0_or_1": "",
                    "auditor_notes": "",
                }
            )
        duplicates = len(concepts) - len({normalize(value) for value in concepts})
        selected_rows = [row for row in concept_rows if row["dataset"] == dataset]
        dataset_summary[dataset] = {
            "concept_count": len(concepts),
            "normalized_duplicate_count": duplicates,
            "mean_prevalence": mean(float(row["activation_prevalence_ge_0p5"]) for row in selected_rows),
            "near_zero_prevalence_fraction": mean(float(row["activation_prevalence_ge_0p5"] < 0.01) for row in selected_rows),
            "near_always_on_fraction": mean(float(row["activation_prevalence_ge_0p5"] > 0.99) for row in selected_rows),
            "high_lexical_label_overlap_fraction": mean(float(row["lexical_label_jaccard"] >= 0.8) for row in selected_rows),
            "semantic_target_warning_fraction": mean(float(row["generator_semantic_target_warning"]) for row in selected_rows),
            "generator_audit": str(audit_path) if audit_path else None,
        }
    write_csv(args.output_dir / "concept_audit.csv", concept_rows)
    write_csv(args.output_dir / "human_audit_manifest.csv", human_rows)
    write_json(
        args.output_dir / "summary.json",
        {
            "datasets": dataset_summary,
            "human_audit_definition": "blinded manual review fields are intentionally empty until an auditor completes them",
            "target_warning": "PE-L14 activations and generator verification are automated evidence, not human concept ground truth",
        },
    )


if __name__ == "__main__":
    main()
