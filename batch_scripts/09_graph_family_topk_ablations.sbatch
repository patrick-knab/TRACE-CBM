#!/usr/bin/env bash
#SBATCH --job-name=trace_grapharch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"
mkdir -p runs/train_models slurm_logs
export TRACE_DATASET_ROOT="${TRACE_DATASET_ROOT:-$PROJECT_ROOT/data/datasets}"

python run_train_models.py \
  --config configs/graph_family_topk_ablations_pe_l14_128_3seed.json \
  --gpus 0 \
  --workers-per-gpu 1 \
  --max-parallel 1
