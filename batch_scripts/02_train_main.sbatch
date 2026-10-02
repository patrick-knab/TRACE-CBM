#!/usr/bin/env bash
#SBATCH --job-name=trace_main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_pro_6000:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
export TRACE_EMBEDDING_ROOT="${TRACE_EMBEDDING_ROOT:-$PROJECT_ROOT/data/embeddings}"
METHOD_GROUP="${METHOD_GROUP:-all}"
python batch_scripts/build_main_batch.py --method-group "$METHOD_GROUP"
python run_train_models.py --config configs/generated/main_models.json --gpus 0,1,2,3 --workers-per-gpu 2 --max-parallel 8
