#!/usr/bin/env bash
#SBATCH --job-name=trace_embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
: "${TRACE_DATASET_ROOT:?Set TRACE_DATASET_ROOT to the extracted dataset directory}"
: "${TRACE_EMBEDDING_ROOT:?Set TRACE_EMBEDDING_ROOT to the embedding output directory}"
export TRACE_DATASET_ROOT="$TRACE_DATASET_ROOT" TRACE_EMBEDDING_ROOT="$TRACE_EMBEDDING_ROOT"
DATASETS=(barista Breakfast mpii_cooking_2)
python embed_dataset_backbones.py \
  --dataset "${DATASETS[$SLURM_ARRAY_TASK_ID]}" \
  --models pe-l14 \
  --window-size 32 \
  --num-gpus 1 \
  --pe-video-batch-size 1 \
  --pe-target-t 8 \
  --seed 42
