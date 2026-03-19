python /mnt/code/yehangcheng/Intruct_augment/pipline/llm_judge.py \
  -i /mnt/code/yehangcheng/Evaluation/GEN/save_data/V3/judged/quality_rating/chatling-1204-1214-non_customized_df-v3-decuped.parquet \
  -o /mnt/code/yehangcheng/Evaluation/GEN/save_data/V3/judged/complex_rating/chatling-1204-1214-non_customized_df-v3-decuped.parquet \
  --use-openai-client \
  --api-key YOUR_KEY \
  --base-url http://10.16.80.150:8031/v1 \
  --model "qwen3-235b" \
  --concurrency 200 \
  --prompt-key complex_prompt \
  --prompt-map-file /mnt/code/yehangcheng/Evaluation/GEN/prompt.py \
  --sample -1
# rate_prompt