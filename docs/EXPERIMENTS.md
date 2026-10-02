# Final experiment protocol

## 1. Main predictive comparison

`batch_scripts/build_main_batch.py` is the single source of truth. It expands five methods over seven protocols and seeds 42–46 using PE-L14 embeddings: 35 templates and 175 runs. The generated JSON is deliberately committed/reviewable, while the builder makes path changes reproducible.

```bash
python batch_scripts/build_main_batch.py
python run_train_models.py --config configs/generated/main_models.json --dry-run
sbatch batch_scripts/02_train_main.sbatch
```

TRACE is trained once with the predictive and concept-forecast objectives. The checkpoint selected during that run is used unchanged for prediction, intervention analysis, and the web UI.

For a TRACE-only smoke manifest, use `python batch_scripts/build_main_batch.py --method-group trace --smoke`. For a TRACE-only full five-seed run, use `--method-group trace`. The default builder includes Linear, MoTIF, TRACE, SlowFast-TCN, and Feature Transformer.

## 2. Main table

`paper_expos/main_table.py` takes all five model families from the 175-run main batch. It refuses incomplete 7 × 5 × 5 result cells.

```bash
MAIN_BATCH=/path/to/main \
sbatch batch_scripts/04_main_table.sbatch
```

## 3. Intervention and explanation evaluations

Use the TRACE checkpoints in the completed main batch as `SOURCE_BATCH`:

| Launcher | Purpose | Tasks |
|---|---|---:|
| `paper_expos/scripts/02_long_horizon.sbatch` | recursive horizon degradation | 35 |
| `paper_expos/scripts/03_intervention_budget.sbatch` | concept, activity-belief, and edge edit response by budget | 35 |
| `paper_expos/scripts/04_graph_stability.sbatch` | graph consistency across seeds/protocols | 1 |
| `paper_expos/scripts/05_exemplar_export.sbatch` | qualitative examples | 1 |
| `paper_expos/scripts/06_concept_audit.sbatch` | vocabulary and activation audit | 1 |
| `paper_expos/scripts/07_concept_robustness.sbatch` | concept noise/dropout robustness | 35 |
| `paper_expos/scripts/10_aggregate_array.sbatch` | completeness-gated array aggregation | 1 |

Example:

```bash
export SOURCE_BATCH=/path/to/completed/main_batch
JOB=$(sbatch --parsable paper_expos/scripts/03_intervention_budget.sbatch)
sbatch --dependency=afterok:$JOB paper_expos/scripts/10_aggregate_array.sbatch \
  intervention_budget runs/paper_expos/intervention_budget_$JOB 35
```

### Matched rollout intervention efficiency

This separate three-seed study compares frozen TRACE checkpoints with a newly
trained parameter-matched dense rollout and Sparse Linear on the exact saved
Table-3 windows.  It clamps only H1 concept coordinates, re-rolls through H3,
and records every H1--H3 concept and activity effect.  The dense control keeps
TRACE's activity feedback; it changes only the relational layers.

```bash
sbatch batch_scripts/15_matched_dense_rollout_intervention.sbatch
sbatch batch_scripts/16_matched_sparse_rollout_intervention.sbatch

export TRACE_BATCH=/path/to/completed/table3_trace_batch
export TRACE_INTERVENTION_DIR=/path/to/table3_intervention_output
JOB=$(sbatch --parsable paper_expos/scripts/13_matched_rollout_intervention.sbatch)
export MATCHED_OUTPUT_DIR=runs/paper_expos/matched_rollout_intervention_$JOB
sbatch --dependency=afterok:$JOB paper_expos/scripts/14_aggregate_matched_rollout_intervention.sbatch
```

The aggregate writes H3 gain-by-budget plots and `efficiency_thresholds.csv`.
The primary threshold is a +5 pp mean H3 true-label probability gain; an arm
that does not reach it by budget five is recorded as not attained.

## 4. Qwen evaluations

The paper has two distinct Qwen workflows. They use different model sizes and answer different questions.

### Qwen intervention planning

`paper_expos/scripts/09_llm_intervention_policy.sbatch` serves Qwen3.6-35B-A3B locally through vLLM and evaluates ordered concept edits on matched H3 cases. The run includes a blind policy and oracle-context policies; oracle context exposes the true future activity and/or counterfactual edit effects, so report those as privileged planning diagnostics. The evaluator uses the true endpoint label directly for metrics and does not use an LLM judge.

Set `SOURCE_BATCH` to the completed TRACE checkpoint batch and `INTERVENTION_DIR` to the prepared intervention evidence directory, then submit from the package root:

```bash
export SOURCE_BATCH=/path/to/completed/main_batch
export INTERVENTION_DIR=/path/to/interventions
export CBM_PYTHON="$(command -v python)"
export VLLM_PYTHON="$(command -v python)"
sbatch paper_expos/scripts/09_llm_intervention_policy.sbatch
```

### Non-oracle target steering

`paper_expos/scripts/11_llm_target_steering.sbatch` uses the same Qwen3.6-35B-A3B endpoint to select concept and activity-belief edits toward a requested H3 activity. It does not expose the future label or future concept targets to Qwen. It reports target-probability change and target rank alongside the case-level outputs.

```bash
export SOURCE_BATCH=/path/to/completed/main_batch
export CBM_PYTHON="$(command -v python)"
export VLLM_PYTHON="$(command -v python)"
sbatch paper_expos/scripts/11_llm_target_steering.sbatch
```

Both launchers need local access to the source videos for storyboard frames, a GPU allocation suitable for the selected Qwen checkpoint, and installed vLLM. Set `TRACE_DATASET_ROOT` before submission. The concept-generation workflow is a separate offline stage using Qwen3.5-27B; it is documented in `DATASETS.md` and configured by `utils/concept_generation_qwen35.json`.

## 5. Component analysis

The component analysis uses three cumulative arms in decreasing order: the full jointly trained model, a retrained model without the shared or task-specific spatio-temporal concept graphs, and a retrained graph-free model without learned concept calibration. The latter two arms cover seeds 42–44 over all seven protocols (42 new training runs); the full arm reuses the main checkpoints.

```bash
sbatch batch_scripts/06_component_analysis_train.sbatch
```

After that batch completes, evaluate ordinary activity/H1--H3 accuracy, macro-F1, and top-3 accuracy together with matched corrective class and concept interventions:

```bash
export MAIN_BATCH=/path/to/completed/main_batch
export COMPONENT_BATCH=/path/to/completed/component_analysis_batch
JOB=$(sbatch --parsable paper_expos/scripts/08_component_analysis.sbatch)
sbatch --dependency=afterok:$JOB paper_expos/scripts/10_aggregate_array.sbatch \
  component_analysis runs/paper_expos/component_analysis_$JOB 21
```

Cases are selected once with the full model and reused across all four arms. Concept interventions use instance-target corrections ranked by their directional true-class margin effect; class interventions set preceding activity distributions toward their ground-truth classes.

## 6. Loss ablation

`batch_scripts/build_loss_ablation_batch.py` builds a matched five-arm loss study on PE-L14 with 128 concepts, the official BARISTA split, Breakfast s1, and MPII Attr, using seeds 42--44. The full objective is compared with one-term removals: concept-forecast supervision, transition-tolerant forecast loss, both SIL false-positive penalties, and classifier L1 regularization. This is 45 retrained runs; graph architecture, data protocol, optimizer, sampling, checkpoint metric, and forecast horizon remain fixed.

```bash
python batch_scripts/build_loss_ablation_batch.py
python run_train_models.py \
  --config configs/generated/loss_ablation_pe_l14_128_3seed.json \
  --dry-run
sbatch batch_scripts/12_loss_ablation.sbatch
```

After completion, aggregate the per-seed activity and H1--H3 metrics with:

```bash
python -m paper_expos.loss_ablation \
  --batch /path/to/completed/loss_ablation_batch
```

## Completion criteria

A submitted job is not a completed experiment. For every batch, confirm:

1. Slurm reports a terminal successful state.
2. `batch_summary.json` exists and every `returncode` is zero.
3. The expected number of `model.pt`, `args.json`, and metric artifacts exists.
4. Aggregators accept the matrix without `--allow-duplicate-runs` or reduced expected counts.
