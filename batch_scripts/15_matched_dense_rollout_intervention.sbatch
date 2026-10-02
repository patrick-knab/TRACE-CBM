#!/usr/bin/env bash
#SBATCH --job-name=trace_matched_dense_intervention
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"
mkdir -p runs/train_models slurm_logs
export TRACE_DATASET_ROOT="${TRACE_DATASET_ROOT:-$PROJECT_ROOT/data/datasets}"
python run_train_models.py \
  --config configs/matched_dense_rollout_intervention_pe_l14_128_3seed.json \
  --gpus 0,1,2,3 \
  --workers-per-gpu 2 \
  --max-parallel 8
latest_batch="$(ls -dt runs/train_models/iclr2027_matched_dense_rollout_intervention_pe_l14_128_3seed_* | head -n 1)"
printf '%s\n' "$latest_batch" > runs/train_models/latest_matched_dense_rollout_intervention.txt
