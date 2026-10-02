#!/usr/bin/env bash
#SBATCH --job-name=paper_aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: sbatch $0 {long_horizon|intervention_budget|concept_robustness|component_analysis} OUTPUT_DIR [EXPECTED_TASKS]" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECTED_ARGS=()
if [[ $# -eq 3 ]]; then
  EXPECTED_ARGS=(--expected-tasks "$3")
fi
"$PYTHON_BIN" -m paper_expos.aggregate "$1" "$2" "${EXPECTED_ARGS[@]}"
"$PYTHON_BIN" -m paper_expos.log_artifact --path "$2" --name "paper_${1}_${SLURM_JOB_ID}"
