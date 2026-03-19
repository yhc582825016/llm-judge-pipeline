export CUDA_VISIBLE_DEVICES=0,1
MODEL_TAG=Qwen3-Embedding-0.6B
SOURCE=filter_all_over_8_to_10_168w_concat_rw_lb_if_leetcode_zh_202w
python3 /code/yehangcheng/Intruct_augment/pipline/cluster.py \
  --parquet_dir /mnt/code/yehangcheng/Intruct_augment/pipline/emb_shards/$MODEL_TAG/$SOURCE \
  --output_dir /mnt/code/yehangcheng/Intruct_augment/pipline/cluster_output/$MODEL_TAG/$SOURCE \
  --k_min 5 --k_max 200 --rng_seed 42 --do_umap --no_write_back --write_labels_to_output_dir
