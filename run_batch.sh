#!/usr/bin/env bash
#SBATCH --job-name=train_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_pro_6000:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.err

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash run_batch.sh <config.json> [run_train_models.py args...]
  sbatch run_batch.sh <config.json> [run_train_models.py args...]

Examples:
  bash run_batch.sh configs/generated/main_models.json --dry-run
  sbatch run_batch.sh configs/generated/main_models.json

Environment overrides:
  GPUS=0,1 WORKERS_PER_GPU=2 MAX_PARALLEL=4 sbatch run_batch.sh config.json
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

CONFIG="$1"
shift

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 2
fi

mkdir -p slurm_logs

TRACE_DATASET_ROOT="${TRACE_DATASET_ROOT:-$(cd "$(dirname "$0")" && pwd)/data/datasets}"
if [[ ! -d "$TRACE_DATASET_ROOT" ]]; then
  echo "Dataset root not found: $TRACE_DATASET_ROOT" >&2
  exit 2
fi
export TRACE_DATASET_ROOT

GPUS="${GPUS:-0,1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
RUN_OUTPUT="$(mktemp)"
trap 'rm -f "$RUN_OUTPUT"' EXIT
DRY_RUN=false

for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=true
    break
  fi
done

python run_train_models.py \
  --config "$CONFIG" \
  --gpus "$GPUS" \
  --workers-per-gpu "$WORKERS_PER_GPU" \
  --max-parallel "$MAX_PARALLEL" \
  "$@" | tee "$RUN_OUTPUT"

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run completed for config: $CONFIG"
  exit 0
fi

BATCH_DIR="$(sed -n 's/^\[run_train_models\] All runs completed\. Batch dir: //p' "$RUN_OUTPUT" | tail -1)"
if [[ -z "$BATCH_DIR" || ! -d "$BATCH_DIR" ]]; then
  echo "Could not determine completed batch directory." >&2
  exit 1
fi

echo "Completed batch directory: $BATCH_DIR"
