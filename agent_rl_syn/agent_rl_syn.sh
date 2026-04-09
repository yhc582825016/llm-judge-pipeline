#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_PATH="/dev/shm/ye/rl-data/agent_syn_data/tool_use_no_think_v2_first_2w.jsonl"
OUTPUT_PATH="/dev/shm/ye/rl-data/agent_syn_data/recall/synthetic_mock_success_only_5.jsonl"
PROGRESS_PATH="/dev/shm/ye/rl-data/agent_syn_data/recall/synthetic_mock_progress.jsonl"

nohup python "$SCRIPT_DIR/agent_rl_syn.py" \
    --input-path "$INPUT_PATH" \
    --output-path "$OUTPUT_PATH" \
    --progress-path "$PROGRESS_PATH" \
    > /dev/shm/ye/logs/agent_rl_syn_nemotron_post_training_v1_tool_call3.log 2>&1 &