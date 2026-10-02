#!/usr/bin/env bash
#SBATCH --job-name=paper_probdiag_aggregate
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
: "${DIAGNOSTIC_OUTPUT_DIR:?Set DIAGNOSTIC_OUTPUT_DIR to the completed diagnostic array output directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from paper_expos.forecast_probability_diagnostics import aggregate

aggregate(Path(os.environ["DIAGNOSTIC_OUTPUT_DIR"]), expected_tasks=9)
PY
