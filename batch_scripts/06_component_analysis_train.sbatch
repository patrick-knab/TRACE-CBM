#!/usr/bin/env bash
#SBATCH --job-name=trace_components
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_pro_6000:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"
mkdir -p configs/generated runs/train_models slurm_logs
export TRACE_DATASET_ROOT="${TRACE_DATASET_ROOT:-$PROJECT_ROOT/data/datasets}"
python batch_scripts/build_component_analysis_batch.py \
  --output configs/generated/component_analysis_train.json
python run_train_models.py \
  --config configs/generated/component_analysis_train.json \
  --gpus 0,1 \
  --workers-per-gpu 2 \
  --max-parallel 4
