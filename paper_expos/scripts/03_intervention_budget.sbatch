#!/usr/bin/env bash
#SBATCH --job-name=paper_intervene
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-8%4
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
if [[ ! -d "$PROJECT_ROOT/paper_expos" ]]; then
  PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
fi
export PROJECT_ROOT
cd "$PROJECT_ROOT"
: "${SOURCE_BATCH:?Set SOURCE_BATCH to the completed final TRACE batch directory}"
TRACE_DATASET_ROOT="${TRACE_DATASET_ROOT:-$PROJECT_ROOT/data/datasets}"
if [[ ! -d "$TRACE_DATASET_ROOT" ]]; then
  echo "TRACE_DATASET_ROOT does not exist: $TRACE_DATASET_ROOT" >&2
  exit 2
fi
export TRACE_DATASET_ROOT
mkdir -p "$PROJECT_ROOT/slurm_logs" "$PROJECT_ROOT/runs/paper_expos"
export MPLCONFIGDIR="/tmp/paper-expos-mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DUP_ARG=""
if [[ "${ALLOW_DUPLICATE_RUNS:-0}" == "1" ]]; then
  DUP_ARG="--allow-duplicate-runs"
fi
EDGE_TREATMENT="${MAIN_PAPER_EDGE_TREATMENT:-delete}"
"$PYTHON_BIN" -m paper_expos.intervention_budget \
  --source-batch "$SOURCE_BATCH" \
  --protocols paper_expos/configs/main_protocols_single_split_128_v1.json \
  --record-index "$SLURM_ARRAY_TASK_ID" \
  --budgets 1,2,3,4,5 \
  --errors 30 \
  --correct 30 \
  --stride 5 \
  --forward-chunk 32 \
  --device cuda:0 \
  $DUP_ARG \
  --main-paper-only \
  --main-paper-edge-treatment "$EDGE_TREATMENT"
