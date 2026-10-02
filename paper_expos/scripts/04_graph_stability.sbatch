#!/usr/bin/env bash
#SBATCH --job-name=paper_graphstab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
: "${SOURCE_BATCH:?Set SOURCE_BATCH to the completed final TRACE batch directory}"
mkdir -p slurm_logs runs/paper_expos
export MPLCONFIGDIR="/tmp/paper-expos-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="runs/paper_expos/graph_stability_${SLURM_JOB_ID}"
"$PYTHON_BIN" -m paper_expos.graph_stability \
  --source-batch "$SOURCE_BATCH" \
  --output-dir "$OUT" \
  --device cpu
"$PYTHON_BIN" -m paper_expos.log_artifact --path "$OUT" --name "paper_graph_stability_${SLURM_JOB_ID}"
