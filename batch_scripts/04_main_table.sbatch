#!/usr/bin/env bash
#SBATCH --job-name=trace_table
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
: "${MAIN_BATCH:?Set MAIN_BATCH to the completed 175-run main batch}"
python -m paper_expos.main_table \
  --main-batch "$MAIN_BATCH" \
  --output-dir "runs/paper_expos/main_table_${SLURM_JOB_ID}"
