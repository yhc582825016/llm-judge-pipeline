#!/usr/bin/env bash

set -euo pipefail

MODEL="qwen3.5_397b"
MODEL_TAG=$(echo "$MODEL" | sed 's#[^0-9A-Za-z._-]#_#g')
INPUT_PARQUET="/mnt/code/yehangcheng/all_data/rl_data_repo/IF/Nemotron-post-training/if_rl_dataset_train_swift_len12k_clean.parquet"
OUTPUT_DIR="/mnt/code/yehangcheng/Intruct_augment/pipline/rl_response_filter/IF/inference_res/$MODEL_TAG"
SUCCESS_PARQUET="$OUTPUT_DIR/if_rl_following_success.parquet"
FAILED_PARQUET="$OUTPUT_DIR/if_rl_following_failed.parquet"
LOG_DIR="/mnt/code/yehangcheng/Intruct_augment/pipline/rl_response_filter/IF/logs"
LOG_PATH="$LOG_DIR/${MODEL_TAG}_if_rl_response_filter.log"

BASE_URL="http://127.0.0.1:6032"
BASE_URL_WEIGHTS="http://127.0.0.1:6032:1"
API_KEY="EMPTY"

CONCURRENCY=100
TIMEOUT=300
REQUEST_MAX_RETRIES=2
MAX_ATTEMPTS=3
SLEEP_DURATION=1
TEMPERATURE=0.6
MAX_TOKENS=12000
THINKING_MODE="off"
SAVE_EVERY=200

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

nohup python /mnt/code/yehangcheng/Intruct_augment/pipline/rl_response_filter/IF/if_rl_response_filter.py \
  --input-parquet "$INPUT_PARQUET" \
  --output-success "$SUCCESS_PARQUET" \
  --output-failed "$FAILED_PARQUET" \
  --model "$MODEL" \
  --api-key "$API_KEY" \
  --base-url "$BASE_URL" \
  --base-url-weights "$BASE_URL_WEIGHTS" \
  --concurrency "$CONCURRENCY" \
  --timeout "$TIMEOUT" \
  --request-max-retries "$REQUEST_MAX_RETRIES" \
  --max-attempts "$MAX_ATTEMPTS" \
  --sleep-duration "$SLEEP_DURATION" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --thinking-mode "$THINKING_MODE" \
  --save-every "$SAVE_EVERY" > "$LOG_PATH" 2>&1 &

echo "started: $LOG_PATH"
