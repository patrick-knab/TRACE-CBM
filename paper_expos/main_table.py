"""Build the final main table from one jointly trained model batch."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from paper_expos.common import DEFAULT_PROTOCOLS, _matching_protocol, load_protocols, write_csv


METHOD_NAMES = {
    "linear_sparse_dynamics_shared_head": "Linear",
    "motif": "MoTIF",
    "graph_cbm": "TRACE",
    "trace": "TRACE",
    "concept_forecast_cbm": "TRACE",  # Existing result artifacts.
    "feature_slowfast_tcn": "SlowFast-TCN",
    "feature_transformer": "Feature Transformer",
}


def scalar(metrics: dict, *path: str) -> float:
    value: object = metrics
    for key in path:
        value = value[key]
    return float(value)


def batch_rows(batch: Path, protocols: dict) -> list[dict[str, object]]:
    rows = []
    for args_path in sorted((batch / "models").glob("*/args.json")):
        args = json.loads(args_path.read_text(encoding="utf-8"))
        method = str(args.get("base_method"))
        if method not in METHOD_NAMES:
            continue
        method_name = METHOD_NAMES[method]
        protocol = _matching_protocol(protocols, args.get("dataset"), args.get("test_split"))
        if protocol is None:
            continue
        metrics_path = args_path.parent / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "protocol": str(protocol["key"]),
                "dataset": str(args["dataset"]),
                "split": str(args["test_split"]),
                "method": method_name,
                "seed": int(args["seed"]),
                "forecast_accuracy": scalar(metrics, "test", "forecast", "accuracy"),
                "forecast_macro_f1": scalar(metrics, "test", "forecast", "macro_f1"),
                "forecast_top3_accuracy": scalar(metrics, "test", "forecast", "top3_accuracy"),
                "activity_accuracy": scalar(metrics, "test", "activity", "accuracy"),
                "checkpoint": str(args_path.parent / "model.pt"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-batch",
        type=Path,
        action="append",
        required=True,
        help="Completed batch containing one or more table methods; repeat for split batches.",
    )
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocols = load_protocols(args.protocols)
    rows = [
        row
        for batch in args.main_batch
        for row in batch_rows(batch, protocols)
    ]

    expected = {
        (str(protocol["key"]), method, int(seed))
        for protocol in protocols["datasets"]
        for method in METHOD_NAMES.values()
        for seed in protocols["seeds"]
    }
    actual = {(str(row["protocol"]), str(row["method"]), int(row["seed"])) for row in rows}
    if actual != expected:
        raise RuntimeError(f"Expected {len(expected)} final cells; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["protocol"]), str(row["method"]))].append(row)
    summary = []
    metric_keys = ["forecast_accuracy", "forecast_macro_f1", "forecast_top3_accuracy", "activity_accuracy"]
    for (protocol, method), group in sorted(grouped.items()):
        output: dict[str, object] = {"protocol": protocol, "method": method, "seeds": len(group)}
        for key in metric_keys:
            values = [float(row[key]) for row in group]
            output[f"{key}_mean"] = mean(values)
            output[f"{key}_std"] = stdev(values)
        summary.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "main_table_per_seed.csv", rows)
    write_csv(args.output_dir / "main_table_mean_std.csv", summary)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "main_batches": [str(batch.resolve()) for batch in args.main_batch],
                "rows": len(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} verified per-seed rows and {len(summary)} aggregate rows")


if __name__ == "__main__":
    main()
