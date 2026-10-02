# TRACE-CBM

TRACE (Temporal Relational and Correctable Concept Bottleneck Model) is a temporal concept bottleneck model for forecasting and correcting activity predictions. This repository contains its source code, experiment configurations, frozen concept vocabularies, and evaluation programs. It does not include licensed videos, pretrained model weights, generated video embeddings, training checkpoints, or result directories.

## Paper

The paper link will be added here when it is available.

## What is included

```text
batch_scripts/   data checks, embedding/training launchers, and config builders
concepts/        frozen 128-concept vocabularies and small bootstrap vocabularies
configs/         paper protocols, model settings, and ablation manifests
docs/            dataset, method, and experiment documentation
notebooks/       interactive single-dataset training and intervention walkthrough
paper_expos/     paper evaluations, including both Qwen studies
tests/           focused implementation checks
utils/           video preparation, PE embedding, VLM concept generation, and TRACE
```

The paper evaluates BARISTA, Breakfast, and MPII Cooking 2 in its main comparison. Supplementary evaluations cover MPII Dishes, GTEA Gaze, EPIC-KITCHENS-100, and a synthetic delayed-dependency task. See [docs/DATASETS.md](docs/DATASETS.md) for task descriptions, expected files, and split details.

## Setup

Use Python 3.11 and install a PyTorch/torchvision build compatible with the available CUDA runtime, then install the remaining requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
set -a; source .env; set +a
```

Set `TRACE_DATASET_ROOT` and `TRACE_EMBEDDING_ROOT` in `.env` or in the environment. Dataset videos and annotations must be acquired from their official sources; the package does not redistribute them. Experiment tracking defaults to offline mode; enable an external tracking service explicitly if desired.

The release intentionally excludes videos and annotations, generated embeddings, trained checkpoints, and paper result directories. Reproducing the real-video experiments therefore requires access to the corresponding official datasets and their terms, storage for generated artifacts, and compatible GPU resources for PE-L14 embedding and TRACE training. VLM concept regeneration and both Qwen evaluations additionally require locally provisioned Qwen models, vLLM, and suitable GPU memory; these are optional for training with the included frozen concept vocabularies.

To verify the core TRACE training path without downloading datasets or using a GPU, run the tiny CPU smoke test:

```bash
python tests/smoke_cpu_trace.py
```

It trains for one epoch on a generated synthetic delayed-dependency dataset, writes outputs to a temporary directory, and checks the checkpoint, metrics, and history against the structural expectations in [tests/fixtures/trace_cpu_smoke_expected.json](tests/fixtures/trace_cpu_smoke_expected.json). This is an execution check only—not a paper result—and metric values are intentionally not fixed across software/hardware versions.

## Interactive walkthrough

Open [notebooks/TRACE_interactive_walkthrough.ipynb](notebooks/TRACE_interactive_walkthrough.ipynb) in Jupyter and run its cells in order. It loads precomputed Breakfast embeddings, prepares the concept and activity sequences, trains a short TRACE demo, previews observed video windows, ranks concept-node and activity-belief edits by their effect on the selected forecast, and lets you apply either edit interactively. Obtain the dataset videos and annotations separately and set `TRACE_DATASET_ROOT` and `TRACE_EMBEDDING_ROOT` as described in the notebook. The notebook saves figures and its checkpoint under `runs/notebook_demo/`, which is ignored by Git. Its short training run is for exploration, not paper reproduction.

## End-to-end workflow

1. **Prepare data and windows.** The embedding pipeline reads the official videos and annotations, divides videos into non-overlapping 32-frame windows, retains a final non-empty partial window, and samples eight frames per window for PE-L14. Training samples sequences of five observed windows and predicts the next three windows.
2. **Extract embeddings.** For the three main datasets, generate PE-L14 features with:

   ```bash
   sbatch batch_scripts/01_embed_main.sbatch
   ```

   On a single machine, use `python embed_dataset_backbones.py --dataset barista --models pe-l14 --window-size 32 --num-gpus 1 --pe-video-batch-size 1 --pe-target-t 8 --seed 42`. Dataset-specific paths and supplementary embedding commands are in [docs/DATASETS.md](docs/DATASETS.md).
3. **Create concept vocabularies.** The frozen, dataset-specific 128-concept JSON files in `concepts/` are the defaults for reproduction; no VLM call is needed to train TRACE. To regenerate concepts, start a local OpenAI-compatible vLLM server for Qwen3.5-27B, then run `python utils/generate_concept_sets.py --config utils/concept_generation_qwen35.json`. The generator samples training storyboards, proposes visually grounded concepts, verifies them against retrieved windows, and filters duplicates and label overlap. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) before replacing any frozen vocabulary.
4. **Train TRACE.** Build and inspect the five-method, seven-protocol, five-seed manifest, then launch training:

   ```bash
   export METHOD_GROUP=trace
   python batch_scripts/build_main_batch.py --method-group "$METHOD_GROUP"
   python run_train_models.py --config configs/generated/main_models.json --dry-run
   sbatch --export=ALL,METHOD_GROUP="$METHOD_GROUP" batch_scripts/02_train_main.sbatch
   ```

   To reproduce the full baseline matrix, use `METHOD_GROUP=all`. The builder writes a 175-run manifest for Linear, MoTIF, TRACE, SlowFast-TCN, and Feature Transformer across BARISTA, Breakfast s1–s4, and MPII Attr/Dishes. The paper’s main table aggregates Breakfast folds and MPII protocols as documented in the paper protocol.
5. **Evaluate TRACE and Qwen.** Set `SOURCE_BATCH` to the completed training batch. Use `paper_expos/scripts/09_llm_intervention_policy.sbatch` for Qwen intervention planning and `paper_expos/scripts/11_llm_target_steering.sbatch` for non-oracle target steering. Both start a local OpenAI-compatible Qwen3.6-35B-A3B endpoint and require the dataset videos for storyboard inputs. The scripts record the model, prompt context, selected actions, and per-case metrics. Oracle-assisted intervention planning and non-oracle target steering are distinct evaluations; see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

For other checks, use `python batch_scripts/check_data.py`, `python batch_scripts/build_main_batch.py --method-group trace --smoke`, and `python -m paper_expos.validate_suite` after the main manifest is generated. The manifest `--smoke` option only builds a capped real-data run plan; it does not train a model. Slurm files assume a cluster with a compatible GPU scheduler; adapt resource directives to your environment.

## Dataset and environment paths

The default paths are `data/datasets/` and `data/embeddings/` inside this folder. To store them elsewhere:

```bash
export TRACE_DATASET_ROOT=/absolute/path/to/datasets
export TRACE_EMBEDDING_ROOT=/absolute/path/to/embeddings
```

Expected layouts, dataset label surfaces, official splits, and licensing notes are in [docs/DATASETS.md](docs/DATASETS.md). Model weights for PE-Core, Qwen, and other backbones are downloaded or provisioned separately according to their respective terms.

## Naming and compatibility

New manifests use `base_method: "trace"`. The loader also accepts `graph_cbm` and the historical `concept_forecast_cbm` identifier so existing TRACE checkpoints and manifests continue to load. The implementation class remains named `GraphCBM` internally.
