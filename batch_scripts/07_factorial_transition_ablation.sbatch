#!/usr/bin/env bash
#SBATCH --job-name=trace_factorial
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
mkdir -p runs/train_models slurm_logs
export TRACE_DATASET_ROOT="${TRACE_DATASET_ROOT:-$PROJECT_ROOT/data/datasets}"
python run_train_models.py \
  --config configs/factorial_transition_ablation_main3_3seed.json \
  --gpus 0,1,2,3 \
  --workers-per-gpu 2 \
  --max-parallel 8
