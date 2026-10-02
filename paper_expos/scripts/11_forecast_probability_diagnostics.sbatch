#!/usr/bin/env bash
#SBATCH --job-name=paper_probdiag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-8%4
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"
: "${SOURCE_BATCH:?Set SOURCE_BATCH to the completed PE-L14 main-model batch directory}"
TRACE_DATASET_ROOT="${TRACE_DATASET_ROOT:-$PROJECT_ROOT/data/datasets}"
if [[ ! -d "$TRACE_DATASET_ROOT" ]]; then
  echo "TRACE_DATASET_ROOT does not exist: $TRACE_DATASET_ROOT" >&2
  exit 2
fi
export TRACE_DATASET_ROOT
mkdir -p slurm_logs runs/paper_expos
export MPLCONFIGDIR="/tmp/paper-expos-mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m paper_expos.forecast_probability_diagnostics \
  --source-batch "$SOURCE_BATCH" \
  --record-index "$SLURM_ARRAY_TASK_ID" \
  --batch-size 256 \
  --ece-bins 15 \
  --device cuda:0
