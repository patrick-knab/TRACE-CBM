#!/usr/bin/env bash
#SBATCH --job-name=trace_components_eval
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
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"
: "${MAIN_BATCH:?Set MAIN_BATCH to the completed jointly trained TRACE batch}"
: "${COMPONENT_BATCH:?Set COMPONENT_BATCH to the completed factorial component batch}"
mkdir -p slurm_logs runs/paper_expos
export TRACE_DATASET_ROOT="${TRACE_DATASET_ROOT:-$PROJECT_ROOT/data/datasets}"
export MPLCONFIGDIR="/tmp/component-analysis-mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m paper_expos.component_analysis \
  --main-batch "$MAIN_BATCH" \
  --component-batch "$COMPONENT_BATCH" \
  --protocols paper_expos/configs/main_protocols_single_split_128_v1.json \
  --record-index "$SLURM_ARRAY_TASK_ID" \
  --seeds 42,43,44 \
  --concept-budgets 1,2,3,4,5 \
  --class-budgets 1,3 \
  --errors 30 \
  --correct 30 \
  --stride 5 \
  --forward-chunk 32 \
  --device cuda:0
