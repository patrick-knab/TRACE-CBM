#!/usr/bin/env bash
#SBATCH --job-name=trace_extra_table
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
: "${MAIN_BATCH:?Set MAIN_BATCH to the completed 50-run additional-dataset batch}"
python -m paper_expos.main_table \
  --main-batch "$MAIN_BATCH" \
  --protocols batch_scripts/additional_protocols_128_v1.json \
  --output-dir "runs/paper_expos/additional_table_${SLURM_JOB_ID}"
