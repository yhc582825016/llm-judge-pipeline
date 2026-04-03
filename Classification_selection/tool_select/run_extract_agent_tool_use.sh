#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_PATH="${1:-/opt/users/ye/data/step3p5_flash_sft_ms_swift.jsonl}"
OUTPUT_PATH="${2:-/mnt/code/yehangcheng/all_data/sft_data/Step-3.5-Flash-SFT/step3p5_flash_sft_ms_swift.agent_tool_use.jsonl}"
STATS_PATH="${3:-/mnt/code/yehangcheng/all_data/sft_data/Step-3.5-Flash-SFT/step3p5_flash_sft_ms_swift.agent_tool_use.stats.json}"

python "${SCRIPT_DIR}/extract_agent_tool_use_from_step35_flash_sft.py" \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --stats "${STATS_PATH}"
