#!/usr/bin/env bash
#SBATCH --job-name=paper_h50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-34%4
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
: "${SOURCE_BATCH:?Set SOURCE_BATCH to the completed final TRACE batch directory}"
mkdir -p slurm_logs runs/paper_expos
export MPLCONFIGDIR="/tmp/paper-expos-mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m paper_expos.long_horizon \
  --source-batch "$SOURCE_BATCH" \
  --record-index "$SLURM_ARRAY_TASK_ID" \
  --max-horizon 50 \
  --stride 5 \
  --max-examples 500 \
  --device cuda:0
