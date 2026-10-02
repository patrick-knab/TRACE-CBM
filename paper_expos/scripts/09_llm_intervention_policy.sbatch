#!/usr/bin/env bash
#SBATCH --job-name=paper_llmseq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"
: "${SOURCE_BATCH:?Set SOURCE_BATCH to the completed 128_v1 main-protocol batch directory}"
: "${INTERVENTION_DIR:?Set INTERVENTION_DIR to the aggregated interventions_<array-job-id> directory}"
mkdir -p slurm_logs runs/paper_expos
export TRACE_DATASET_ROOT="${TRACE_DATASET_ROOT:-$PROJECT_ROOT/data/datasets}"
MEDIA_ROOT="$(realpath "$TRACE_DATASET_ROOT")"
export MPLCONFIGDIR="/tmp/paper-expos-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

CBM_PYTHON="${CBM_PYTHON:-$(command -v python)}"
VLLM_PYTHON="${VLLM_PYTHON:-$(dirname "$(dirname "$CBM_PYTHON")")/../vllm/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
REVISION="${REVISION:-995ad96eacd98c81ed38be0c5b274b04031597b0}"
PORT="${PORT:-$((18000 + SLURM_JOB_ID % 10000))}"
export PATH="$(dirname "$VLLM_PYTHON"):$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
command -v ninja >/dev/null
ninja --version

"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --revision "$REVISION" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --gdn-prefill-backend triton \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  --allowed-local-media-path "$MEDIA_ROOT" \
  >"slurm_logs/paper_llmseq_server-${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

"$VLLM_PYTHON" -c "import time,urllib.request
url='http://127.0.0.1:${PORT}/health'
for _ in range(180):
    try:
        urllib.request.urlopen(url, timeout=5).read(); break
    except Exception: time.sleep(5)
else: raise SystemExit('vLLM server did not become healthy')"

"$VLLM_PYTHON" -c "import json,urllib.request
schema={'type':'object','properties':{'ordered_action_ids':{'type':'array','items':{'type':'string','enum':['a','b','c','d','e']},'minItems':5,'maxItems':5},'rationale':{'type':'string','maxLength':40}},'required':['ordered_action_ids','rationale'],'additionalProperties':False}
payload={'model':'${MODEL}','messages':[{'role':'user','content':'Return all five action IDs and a short rationale.'}],'max_tokens':128,'temperature':0.0,'response_format':{'type':'json_schema','json_schema':{'name':'schema_smoke','strict':True,'schema':schema}}}
request=urllib.request.Request('http://127.0.0.1:${PORT}/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
result=json.loads(urllib.request.urlopen(request,timeout=120).read().decode())
content=json.loads(result['choices'][0]['message']['content'])
assert len(content['ordered_action_ids']) == 5, content"

CUDA_VISIBLE_DEVICES="" "$CBM_PYTHON" -m paper_expos.llm_intervention_policy \
  --source-batch "$SOURCE_BATCH" \
  --intervention-dir "$INTERVENTION_DIR" \
  --protocols paper_expos/configs/main_protocols_single_split_128_v1.json \
  --output-dir "runs/paper_expos/llm_policy_${SLURM_JOB_ID}" \
  --endpoint "http://127.0.0.1:$PORT" \
  --model "$MODEL" \
  --errors-per-seed 30 \
  --correct-per-seed 30 \
  --policy-runs 1 \
  --endpoints h3 \
  --action-types concept \
  --candidate-concepts 10 \
  --intervention-mode persistent \
  --checkpoint-every 5 \
  --max-consecutive-request-errors 3 \
  --policy-attempts 8 \
  --oracle-context \
  --sequential-oracle-context \
  --invalid-next-action-fallback conditional_best \
  --device cpu
CUDA_VISIBLE_DEVICES="" "$CBM_PYTHON" -m paper_expos.log_artifact \
  --path "runs/paper_expos/llm_policy_${SLURM_JOB_ID}" \
  --name "paper_llm_policy_${SLURM_JOB_ID}"
