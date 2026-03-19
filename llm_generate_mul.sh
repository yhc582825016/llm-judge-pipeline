# 定义模型列表
MODELS=("local")
# /mnt/code/yehangcheng/sft_data/lmsys-chat-1m/lmsys-chat-1m-multi-turn.jsonl
# /mnt/code/yehangcheng/sft_data/Infinity-Instruct/7M_core_multi_turn.jsonl
# /mnt/code/yehangcheng/sft_data/multiturn_chat_0.8M/multiturn_chat_0.8M_processed.jsonl
# /mnt/code/yehangcheng/sft_data/Magpie-Llama-3.1-Pro-MT-300K-Filtered/multi_turn_processed.jsonl
TEST_DATA_PATH=/mnt/code/yehangcheng/sft_data/Magpie-Llama-3.1-Pro-MT-300K-Filtered/multi_turn_processed.jsonl
SAVE_PATH=/mnt/code/yehangcheng/sft_data/Magpie-Llama-3.1-Pro-MT-300K-Filtered/multi_turn_processed_qwen3-30b-judge.jsonl
NUM_RETURN_SEQUENCES=1
TEMPERATRUE=0.6
unset http_proxy https_proxy
for MODEL in "${MODELS[@]}"
do
    nohup python /mnt/code/yehangcheng/Intruct_augment/pipline/llm_generate_mul.py \
        --temperature $TEMPERATRUE \
        --data_path $TEST_DATA_PATH \
        --model $MODEL \
        --save_path ${SAVE_PATH} \
        --max_tokens 12000 \
        --max_samples 1000000 \
        --num_threads 256 \
        --num_return_sequences $NUM_RETURN_SEQUENCES > /mnt/code/yehangcheng/Intruct_augment/gen_data/Magpie-Llama-3.1-Pro-MT-300K-Filtered-qwen3-30b-judge.log 2>&1 &
done
