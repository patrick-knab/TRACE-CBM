#!/usr/bin/env bash
#SBATCH --job-name=trace_loss_ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_pro_6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"
mkdir -p configs/generated runs/train_models slurm_logs
export TRACE_DATASET_ROOT="${TRACE_DATASET_ROOT:-$PROJECT_ROOT/data/datasets}"
python batch_scripts/build_loss_ablation_batch.py \
  --output configs/generated/loss_ablation_pe_l14_128_3seed.json
python run_train_models.py \
  --config configs/generated/loss_ablation_pe_l14_128_3seed.json \
  --gpus 0,1,2,3 \
  --workers-per-gpu 2 \
  --max-parallel 8
