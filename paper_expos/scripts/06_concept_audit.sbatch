#!/usr/bin/env bash
#SBATCH --job-name=paper_caudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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
OUT="runs/paper_expos/concept_audit_${SLURM_JOB_ID}"
"$PYTHON_BIN" -m paper_expos.concept_audit \
  --source-batch "$SOURCE_BATCH" \
  --output-dir "$OUT" \
  --human-audit-concepts 30 \
  --device cuda:0
"$PYTHON_BIN" -m paper_expos.log_artifact --path "$OUT" --name "paper_concept_audit_${SLURM_JOB_ID}"
