"""Compare fixed-case intervention replay with the original balanced screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "accuracy_after",
    "wrong_to_correct_rate",
    "correct_to_wrong_rate",
    "label_flip_rate",
    "true_probability_delta",
    "future_concept_l1",
)
HEADLINES = {
    "concept": ("directional_oracle", "flip", 5),
    "activity": ("temporal_sources", "oracle_label", 5),
    "edge": ("strongest", "invert", 5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--reproduced-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {root / pattern}, found {len(matches)}")
    return matches[0]


def keyed_summary(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, int], dict[str, str]]:
    return {
        (
            row["intervention_type"],
            row["policy"],
            row["treatment"],
            int(row["budget"]),
        ): row
        for row in rows
    }


def main() -> None:
    args = parse_args()
    violations: list[str] = []
    dataset_reports = []
    for dataset in ("barista", "breakfast", "mpii"):
        original_manifest_path = one(args.original_dir, f"{dataset}_seed42_*balanced_all_manifest.json")
        reproduced_manifest_path = one(args.reproduced_dir, f"{dataset}_seed42_*exact_repro_manifest.json")
        original_manifest = read_json(original_manifest_path)
        reproduced_manifest = read_json(reproduced_manifest_path)
        identity_fields = ("case_index", "video_id", "video_index", "timestep", "true_label")
        original_identities = [tuple(row[field] for field in identity_fields) for row in original_manifest]
        reproduced_identities = [tuple(row[field] for field in identity_fields) for row in reproduced_manifest]
        identity_match = original_identities == reproduced_identities
        if not identity_match:
            violations.append(f"{dataset}: replayed case identities differ from original manifest")

        original_summary = keyed_summary(read_csv(one(args.original_dir, f"{dataset}_seed42_*balanced_all_summary.csv")))
        reproduced_summary = keyed_summary(read_csv(one(args.reproduced_dir, f"{dataset}_seed42_*exact_repro_summary.csv")))
        if set(original_summary) != set(reproduced_summary):
            violations.append(f"{dataset}: intervention summary key sets differ")
        comparison_rows = []
        for key in sorted(set(original_summary) & set(reproduced_summary)):
            row = {
                "intervention_type": key[0],
                "policy": key[1],
                "treatment": key[2],
                "budget": key[3],
            }
            for metric in METRICS:
                original_value = float(original_summary[key][metric])
                reproduced_value = float(reproduced_summary[key][metric])
                row[metric] = {
                    "original": original_value,
                    "reproduced": reproduced_value,
                    "difference": reproduced_value - original_value,
                }
            comparison_rows.append(row)

        headline = {}
        for intervention_type, suffix in HEADLINES.items():
            key = (intervention_type, *suffix)
            selected = next(
                (row for row in comparison_rows if (
                    row["intervention_type"], row["policy"], row["treatment"], row["budget"]
                ) == key),
                None,
            )
            if selected is None:
                violations.append(f"{dataset}: missing headline intervention {key}")
            headline[intervention_type] = selected

        provenance = read_json(one(args.reproduced_dir, f"{dataset}_seed42_*exact_repro_replay_provenance.json"))
        if not provenance.get("exact_evaluator_match"):
            violations.append(f"{dataset}: evaluator implementation hash differs from original")
        if not provenance.get("all_implementation_modules_loaded_from_original_repository"):
            violations.append(f"{dataset}: evaluator helper stack was not loaded from original repository")
        dataset_reports.append(
            {
                "dataset": dataset,
                "original_manifest": str(original_manifest_path.resolve()),
                "reproduced_manifest": str(reproduced_manifest_path.resolve()),
                "case_count": len(reproduced_manifest),
                "case_identity_match": identity_match,
                "original_clean_accuracy": sum(bool(row["baseline_correct"]) for row in original_manifest) / len(original_manifest),
                "reproduced_clean_accuracy": sum(bool(row["baseline_correct"]) for row in reproduced_manifest) / len(reproduced_manifest),
                "original_prediction_matches": int(provenance["original_prediction_matches"]),
                "exact_evaluator_match": bool(provenance["exact_evaluator_match"]),
                "all_implementation_modules_loaded_from_original_repository": bool(
                    provenance["all_implementation_modules_loaded_from_original_repository"]
                ),
                "headline": headline,
                "all_intervention_comparisons": comparison_rows,
            }
        )

    report = {
        "status": "pass" if not violations else "fail",
        "original_dir": str(args.original_dir.resolve()),
        "reproduced_dir": str(args.reproduced_dir.resolve()),
        "violations": violations,
        "headline_definition": {
            key: {"policy": value[0], "treatment": value[1], "budget": value[2]}
            for key, value in HEADLINES.items()
        },
        "datasets": dataset_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "violations": violations}, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
