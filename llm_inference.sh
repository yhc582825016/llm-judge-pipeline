MODEL="qwen3.5_397b"
MODEL_TAG=$(echo "$MODEL" | sed 's#[^0-9A-Za-z._-]#_#g')
INPUT_PARQUET="/opt/users/ye/data/core_data/filter_all_over_8_to_10_168w_concat_rw_lb_if_leetcode_zh_202w.for_infer.parquet"
OUTPUT_PARQUET="/mnt/code/yehangcheng/Intruct_augment/pipline/inference_res/$MODEL_TAG/filter_all_over_8_to_10_168w_concat_rw_lb_if_leetcode_zh_202w_${MODEL_TAG}_generated.parquet"
LOG_DIR="/mnt/code/yehangcheng/Intruct_augment/pipline/logs"
LOG_PATH="$LOG_DIR/${MODEL_TAG}_202w.log"

mkdir -p "/mnt/code/yehangcheng/Intruct_augment/pipline/inference_res/$MODEL_TAG"
mkdir -p "$LOG_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

nohup python /mnt/code/yehangcheng/Intruct_augment/pipline/llm_inference.py \
  -i "$INPUT_PARQUET" \
  -o "$OUTPUT_PARQUET" \
  --field prompt \
  --use-openai-client \
  --api-key EMPTY \
  --base-url http://127.0.0.1:6029 \
  --base-url-weights http://127.0.0.1:6029:1 \
  --model "$MODEL" \
  --num-samples 1 \
  --concurrency 500 \
  --timeout 60 \
  --max-retries 2 \
  --sleep-duration 1 \
  --thinking-mode off \
  --save-every 2000 \
  --resume \
  --retry-failed > "$LOG_PATH" 2>&1 &
