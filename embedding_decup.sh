EMB_MODEL_TAG=Qwen3-Embedding-0.6B
SOURCE=chatling-1204-1214-non_customized_df-v3
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 /mnt/code/yehangcheng/Intruct_augment/pipline/embedding_decup.py \
  --mode self_dedup \
  --train_emb_dir /mnt/code/yehangcheng/Intruct_augment/pipline/emb_shards/$EMB_MODEL_TAG/$SOURCE \
  --out /mnt/code/yehangcheng/Intruct_augment/pipline/decup_result/$EMB_MODEL_TAG/${SOURCE}_kept.parquet \
  --dump_removed /mnt/code/yehangcheng/Intruct_augment/pipline/decup_result/$EMB_MODEL_TAG/${SOURCE}_removed-0.9.parquet \
  --threshold 0.9 \
  --device cuda \
  --chunk_db 10000 \
  --chunk_q 128
  # --test_emb_dir /mnt/code/yehangcheng/Intruct_augment/pipline/emb_shards/AceReason-1.1-SFT-qwen3-235b-ins \
