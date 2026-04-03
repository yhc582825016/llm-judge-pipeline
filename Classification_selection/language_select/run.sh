python /mnt/code/yehangcheng/Intruct_augment/pipline/Classification_selection/language_select/filter_chinese_jsonl.py \
  --input /opt/users/ye/data/step3p5_flash_sft_ms_swift.jsonl \
  --output /mnt/code/yehangcheng/all_data/sft_data/Step-3.5-Flash-SFT/step3p5_flash_sft_ms_swift_zh.jsonl \
  --min-ratio 0.1 \
  --min-cjk-count 4 \
  --user-policy all \
  --method lingua \
  --resume
