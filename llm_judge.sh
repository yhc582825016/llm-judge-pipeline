#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

INPUT_PATH="${1:-/dev/shm/ye/rl-data/math-rlvr-unified.parquet}"
OUTPUT_PATH="${2:-/dev/shm/math-rlvr-unified-dup.math_difficulty.parquet.parts}"

API_KEY="${API_KEY:-YOUR_KEY}"
BASE_URL="${BASE_URL:-http://0.0.0.0:6029/v1}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-35b}"
CONCURRENCY="${CONCURRENCY:-300}"
PROMPT_KEY="${PROMPT_KEY:-math_difficulty_prompt}"
FIELD_NAME="${FIELD_NAME:-prompt}"
SAMPLE_SIZE="${SAMPLE_SIZE:--1}"
LOG_DIR="${LOG_DIR:-/dev/shm/ye/logs}"
ENABLE_NOHUP="${ENABLE_NOHUP:-1}"
SAVE_EVERY="${SAVE_EVERY:-200}"

ORIGINAL_ARGS=("$@")

if [[ "${ENABLE_NOHUP}" == "1" && "${LLM_JUDGE_NOHUP_CHILD:-0}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  timestamp="$(date -u +"%Y%m%d_%H%M%S")"
  output_name="$(basename "${2:-/dev/shm/math-rlvr-unified-dup.math_difficulty.parquet.parts}")"
  log_path="${LOG_DIR}/${output_name}.${timestamp}.log"
  nohup env \
    LLM_JUDGE_NOHUP_CHILD=1 \
    LOG_DIR="${LOG_DIR}" \
    ENABLE_NOHUP="${ENABLE_NOHUP}" \
    bash "$0" "${ORIGINAL_ARGS[@]}" >"${log_path}" 2>&1 < /dev/null &
  pid=$!
  echo "Started in background with nohup"
  echo "PID: ${pid}"
  echo "Log: ${log_path}"
  exit 0
fi

shift $(( $# >= 2 ? 2 : $# ))

echo "Running llm_judge.py with direct resume support"
echo "Input: ${INPUT_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "Concurrency: ${CONCURRENCY}"
echo "Sample size: ${SAMPLE_SIZE}"
echo "Save every: ${SAVE_EVERY}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/llm_judge.py" \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --field "${FIELD_NAME}" \
  --use-openai-client \
  --api-key "${API_KEY}" \
  --base-url "${BASE_URL}" \
  --model "${MODEL_NAME}" \
  --concurrency "${CONCURRENCY}" \
  --prompt-key "${PROMPT_KEY}" \
  --prompt-map-file "${SCRIPT_DIR}/prompt.py" \
  --sample "${SAMPLE_SIZE}" \
  --save-every "${SAVE_EVERY}" \
  "$@"
