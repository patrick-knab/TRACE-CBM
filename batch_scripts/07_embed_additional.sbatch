#!/usr/bin/env bash
#SBATCH --job-name=trace_embed_extra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-1
#SBATCH --gpus=rtx_pro_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
: "${TRACE_DATASET_ROOT:?Set TRACE_DATASET_ROOT to the extracted dataset directory}"
: "${TRACE_EMBEDDING_ROOT:?Set TRACE_EMBEDDING_ROOT to the embedding output directory}"
export TRACE_DATASET_ROOT="$TRACE_DATASET_ROOT" TRACE_EMBEDDING_ROOT="$TRACE_EMBEDDING_ROOT"
DATASETS=(gtea_gaze epic_kitchens_100)
GPU_COUNTS=(1 2)
python embed_dataset_backbones.py \
  --dataset "${DATASETS[$SLURM_ARRAY_TASK_ID]}" \
  --models pe-l14 \
  --window-size 32 \
  --batch-size 256 \
  --num-gpus "${GPU_COUNTS[$SLURM_ARRAY_TASK_ID]}" \
  --pe-video-batch-size 1 \
  --pe-target-t 8 \
  --random-windows \
  --enable-tf32 \
  --cleanup-intermediate \
  --no-save-embedder-pickle \
  --save-portable-state-pickle
