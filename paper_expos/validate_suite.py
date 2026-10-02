#!/usr/bin/env python3
"""Fast preflight for the final paper repository and optional model batch."""

from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
from pathlib import Path

from paper_expos.common import discover_checkpoints


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "paper_expos/configs/main_protocols_128_v1.json"
MAIN_CONFIG = PROJECT_ROOT / "configs/generated/main_models.json"
EXPECTED_SCRIPTS = {
    "batch_scripts/01_embed_main.sbatch",
    "batch_scripts/02_train_main.sbatch",
    "batch_scripts/04_main_table.sbatch",
    "batch_scripts/06_component_analysis_train.sbatch",
    "batch_scripts/07_embed_additional.sbatch",
    "batch_scripts/08_train_additional.sbatch",
    "batch_scripts/10_additional_table.sbatch",
    "batch_scripts/15_matched_dense_rollout_intervention.sbatch",
    "batch_scripts/16_matched_sparse_rollout_intervention.sbatch",
    "batch_scripts/17_matched_capacity_dense_rollout_intervention.sbatch",
    "paper_expos/scripts/02_long_horizon.sbatch",
    "paper_expos/scripts/03_intervention_budget.sbatch",
    "paper_expos/scripts/04_graph_stability.sbatch",
    "paper_expos/scripts/05_exemplar_export.sbatch",
    "paper_expos/scripts/06_concept_audit.sbatch",
    "paper_expos/scripts/07_concept_robustness.sbatch",
    "paper_expos/scripts/08_component_analysis.sbatch",
    "paper_expos/scripts/09_llm_intervention_policy.sbatch",
    "paper_expos/scripts/10_aggregate_array.sbatch",
    "paper_expos/scripts/13_matched_rollout_intervention.sbatch",
    "paper_expos/scripts/14_aggregate_matched_rollout_intervention.sbatch",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, help="Completed jointly trained model batch.")
    args = parser.parse_args()

    protocols = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert len(protocols["datasets"]) == 7
    assert protocols["seeds"] == [42, 43, 44, 45, 46]

    config = json.loads(MAIN_CONFIG.read_text(encoding="utf-8"))
    assert len(config["runs"]) == 35
    assert config["seeds"] == [42, 43, 44, 45, 46]
    assert len(config["runs"]) * len(config["seeds"]) == 175
    methods = {row["base_method"] for row in config["runs"]}
    assert methods == {
        "linear_sparse_dynamics_shared_head",
        "motif",
        "trace",
        "feature_slowfast_tcn",
        "feature_transformer",
    }

    actual_scripts = {
        str(path.relative_to(PROJECT_ROOT))
        for root in (PROJECT_ROOT / "batch_scripts", PROJECT_ROOT / "paper_expos/scripts")
        for path in root.glob("*.sbatch")
    }
    missing_scripts = EXPECTED_SCRIPTS - actual_scripts
    if missing_scripts:
        raise RuntimeError(
            f"Missing required launchers: {sorted(missing_scripts)}"
        )
    for relative in sorted(actual_scripts):
        subprocess.run(["bash", "-n", str(PROJECT_ROOT / relative)], check=True)

    for path in PROJECT_ROOT.rglob("*.py"):
        if "__pycache__" not in path.parts:
            py_compile.compile(str(path), doraise=True)

    forbidden = ("/" + "home" + "/", "/" + "Users" + "/")
    offenders = []
    for pattern in ("*.py", "*.json", "*.sbatch", "*.sh", "*.md"):
        for path in PROJECT_ROOT.rglob(pattern):
            if path.parts[len(PROJECT_ROOT.parts)] in {"runs", "wandb", "slurm_logs"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(token in text for token in forbidden):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    if offenders:
        raise RuntimeError(f"Non-portable source paths remain: {sorted(set(offenders))}")

    concept_names = {str(run["concept_set"]) for run in config["runs"]}
    concept_paths = [PROJECT_ROOT / "concepts" / f"{name}.json" for name in sorted(concept_names)]
    for concept_path in concept_paths:
        payload = json.loads(concept_path.read_text(encoding="utf-8"))["concepts"]
        values = next(iter(payload.values()))
        if len(values) != 128:
            raise RuntimeError(f"Expected 128 concepts in {concept_path}, found {len(values)}")

    print("OK: 5 methods x 7 protocols x 5 seeds = 175 main runs")
    print(f"OK: {len(actual_scripts)} Slurm launchers, Python compilation, paths, and concepts")
    if args.source_batch:
        records = discover_checkpoints(args.source_batch)
        print(f"OK: source batch contains {len(records)} TRACE checkpoints")


if __name__ == "__main__":
    main()
