#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
python -m streamlit run diagnosis_repair_ui.py \
  --server.address 127.0.0.1 \
  --server.port "${PORT:-8504}" \
  --server.headless true \
  --server.fileWatcherType none \
  -- \
  --model-root runs/train_models \
  --device "${DEVICE:-cpu}"
