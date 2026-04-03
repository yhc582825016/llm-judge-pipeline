#!/usr/bin/env bash
set -euo pipefail

MODELS=("local")

TEST_DATA_PATH=/mnt/code/yehangcheng/all_data/sft_data/Step-3.5-Flash-SFT/step3p5_flash_sft_ms_swift_zh.jsonl
SAVE_PATH=/mnt/code/yehangcheng/all_data/sft_data/Step-3.5-Flash-SFT/step3p5_flash_sft_ms_swift_zh_regen_response.jsonl
LOG_PATH=/mnt/code/yehangcheng/Intruct_augment/pipline/logs/step3p5_flash_sft_ms_swift_zh_regen_response.log
BASE_URL=${BASE_URL:-http://10.16.80.150:6032/v1}
RESUME=${RESUME:-true}

NUM_RETURN_SEQUENCES=1
TEMPERATURE=0.6
MAX_TOKENS=12000
MAX_SAMPLES=1000000
NUM_THREADS=256

mkdir -p "$(dirname "$SAVE_PATH")" "$(dirname "$LOG_PATH")"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

for MODEL in "${MODELS[@]}"; do
    EXTRA_ARGS=()
    if [[ "$RESUME" == "true" ]]; then
        EXTRA_ARGS+=(--resume)
    else
        EXTRA_ARGS+=(--no_resume)
    fi

    nohup python /mnt/code/yehangcheng/Intruct_augment/pipline/llm_generate_mul.py \
        --temperature "$TEMPERATURE" \
        --data_path "$TEST_DATA_PATH" \
        --model "$MODEL" \
        --save_path "$SAVE_PATH" \
        --base_url "$BASE_URL" \
        --max_tokens "$MAX_TOKENS" \
        --max_samples "$MAX_SAMPLES" \
        --num_threads "$NUM_THREADS" \
        --num_return_sequences "$NUM_RETURN_SEQUENCES" \
        "${EXTRA_ARGS[@]}" > "$LOG_PATH" 2>&1 &
    echo "Started model=$MODEL pid=$! base_url=$BASE_URL resume=$RESUME log=$LOG_PATH"
done
