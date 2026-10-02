#!/usr/bin/env bash
# Training-free PE label-similarity baseline on every held-out window.
# Array entries cover seven protocols x train/dev/test; dev uses the project's val split.
#SBATCH --job-name=zero_shot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-20%3
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

# Experiment settings: change these two values for another matched PE run.
PE_VERSION="pe-l14"
NUM_CONCEPTS=128

# Use Slurm's recorded submission directory; inherited PROJECT_ROOT values can
# point to a non-writable staging directory on the compute node.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/slurm_logs" "$PROJECT_ROOT/runs/paper_expos"
export TRACE_DATASET_ROOT="${TRACE_DATASET_ROOT:-$PROJECT_ROOT/data/datasets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="$PROJECT_ROOT/runs/paper_expos/zero_shot_${PE_VERSION}_${NUM_CONCEPTS}_${SLURM_ARRAY_JOB_ID}"
SPLIT_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
case "$SPLIT_INDEX" in
  0) EVALUATION_SPLIT="train" ;;
  1) EVALUATION_SPLIT="dev" ;;
  2) EVALUATION_SPLIT="test" ;;
esac

"$PYTHON_BIN" -m paper_expos.zero_shot_activity \
  --record-index "$SLURM_ARRAY_TASK_ID" \
  --split "$EVALUATION_SPLIT" \
  --backbone "$PE_VERSION" \
  --num-concepts "$NUM_CONCEPTS" \
  --output-dir "$OUT" \
  --device cuda:0
