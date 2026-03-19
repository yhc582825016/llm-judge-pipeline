export CUDA_VISIBLE_DEVICES=4,5,6,7
source=chatling-1204-1214-customized_df-v3
MODEL_PATH=/opt/users/ye/Qwen3-Embedding-0.6B
MODEL_TAG=$(basename "$MODEL_PATH")
echo $source
# nohup 
python3 /mnt/code/yehangcheng/Intruct_augment/pipline/embedding.py \
  --input /mnt/code/yehangcheng/Evaluation/GEN/eval_data/V3/customized_df.parquet \
  --model_path "$MODEL_PATH" \
  --out_dir /mnt/code/yehangcheng/Intruct_augment/pipline/emb_shards/$MODEL_TAG/$source/ \
  --embedding_batch_size 16 \
  --max_input_tokens 8192 \
  --tensor_parallel_size 4 \
  #  > /mnt/code/yehangcheng/Intruct_augment/pipline/get_embedding.log 2>&1 &
  # --concat-user-turns
