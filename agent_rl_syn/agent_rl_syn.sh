#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_PATH="${INPUT_PATH:-/dev/shm/ye/data/tool_use_no_think_v2.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-/dev/shm/ye/rl-data/agent_syn_data/recall/synthetic_mock_success_only_8.jsonl}"
PROGRESS_PATH="${PROGRESS_PATH:-/dev/shm/ye/rl-data/agent_syn_data/recall/synthetic_mock_progress_8.jsonl}"
BASE_URL="${BASE_URL:-http://127.0.0.1:6031/v1}"
MODEL="${MODEL:-/opt/users/Qwen/Qwen3.5-397B}"
NUM_WORKERS="${NUM_WORKERS:-50}"
MOCK_MAX_TOKENS="${MOCK_MAX_TOKENS:-8192}"
QA_MAX_TOKENS="${QA_MAX_TOKENS:-4096}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-180}"
MAX_REQUEST_RETRIES="${MAX_REQUEST_RETRIES:-2}"
REQUEST_RETRY_BACKOFF_SEC="${REQUEST_RETRY_BACKOFF_SEC:-3}"
LOG_PATH="${LOG_PATH:-/dev/shm/ye/logs/agent_rl_syn_nemotron_post_training_v1_tool_call8.log}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$PROGRESS_PATH")" "$(dirname "$LOG_PATH")"

nohup "$PYTHON_BIN" "$SCRIPT_DIR/agent_rl_syn.py" \
    --input-path "$INPUT_PATH" \
    --output-path "$OUTPUT_PATH" \
    --progress-path "$PROGRESS_PATH" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --num-workers "$NUM_WORKERS" \
    --mock-max-tokens "$MOCK_MAX_TOKENS" \
    --qa-max-tokens "$QA_MAX_TOKENS" \
    --request-timeout-sec "$REQUEST_TIMEOUT_SEC" \
    --max-request-retries "$MAX_REQUEST_RETRIES" \
    --request-retry-backoff-sec "$REQUEST_RETRY_BACKOFF_SEC" \
    > "$LOG_PATH" 2>&1 &
