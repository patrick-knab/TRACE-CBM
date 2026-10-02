#!/usr/bin/env bash
#SBATCH --job-name=paper_matched_rollout_aggregate
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
: "${MATCHED_OUTPUT_DIR:?Set MATCHED_OUTPUT_DIR to the completed matched-rollout array output directory}"
: "${TRACE_BATCH:?Set TRACE_BATCH to the completed Table-3 TRACE batch}"
if [[ -z "${DENSE_BATCH:-}" ]]; then
  DENSE_BATCH="$(< runs/train_models/latest_matched_dense_rollout_intervention.txt)"
fi
if [[ -z "${SPARSE_BATCH:-}" ]]; then
  SPARSE_BATCH="$(< runs/train_models/latest_matched_sparse_rollout_intervention.txt)"
fi
: "${DENSE_BATCH:?Set DENSE_BATCH to the completed matched dense batch}"
: "${SPARSE_BATCH:?Set SPARSE_BATCH to the completed matched Sparse Linear batch}"
: "${TRACE_INTERVENTION_DIR:?Set TRACE_INTERVENTION_DIR to the Table-3 intervention output directory}"
if [[ -f "$DENSE_BATCH" ]]; then
  DENSE_BATCH="$(< "$DENSE_BATCH")"
fi
if [[ -f "$SPARSE_BATCH" ]]; then
  SPARSE_BATCH="$(< "$SPARSE_BATCH")"
fi
"${PYTHON_BIN:-python}" -m paper_expos.matched_rollout_intervention \
  --trace-batch "$TRACE_BATCH" \
  --dense-batch "$DENSE_BATCH" \
  --sparse-batch "$SPARSE_BATCH" \
  --trace-intervention-dir "$TRACE_INTERVENTION_DIR" \
  --dataset-root "${TRACE_DATASET_ROOT:-data/datasets}" \
  --output-dir "$MATCHED_OUTPUT_DIR" \
  --selection-policy "${SELECTION_POLICY:-arm_directional_oracle}" \
  ${REQUIRE_CAPACITY_MATCHED_DENSE:+--require-capacity-matched-dense} \
  --aggregate-only
