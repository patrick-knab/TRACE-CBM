#!/usr/bin/env bash
#SBATCH --job-name=trace_matched_ng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
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
  --config configs/matched_non_graph_dynamics_pe_l14_128_3seed.json \
  --gpus 0 \
  --workers-per-gpu 1 \
  --max-parallel 1
