#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm


AGENT_SYSTEM_PATTERNS = [
    re.compile(r"can interact with a computer", re.IGNORECASE),
    re.compile(r"\bexecute commands\b", re.IGNORECASE),
    re.compile(r"\bmodify code\b", re.IGNORECASE),
    re.compile(r"\bmake a tool call\b", re.IGNORECASE),
    re.compile(r"\bavailable tools\b", re.IGNORECASE),
    re.compile(r"\bfunction[_ -]?calling\b", re.IGNORECASE),
    re.compile(r"<ROLE>|<EFFICIENCY>|<FILE_SYSTEM_GUIDELINES>|<CODE_QUALITY>|<VERSION_CONTROL>", re.IGNORECASE),
]

ASSISTANT_TOOL_INTENT_PATTERNS = [
    re.compile(r"\bI need to use the `[^`]+` function\b", re.IGNORECASE),
    re.compile(r"\bLet me call that function\b", re.IGNORECASE),
    re.compile(r"\bI'll call the `[^`]+` function\b", re.IGNORECASE),
    re.compile(r"\bI(?:'ll| will) retrieve .* using the `[^`]+` function\b", re.IGNORECASE),
    re.compile(r"\btool call\b", re.IGNORECASE),
    re.compile(r"\bfunction call\b", re.IGNORECASE),
]

STRUCTURED_TOOL_KEYS = {
    "tool_calls",
    "tool_call",
    "function_call",
    "function_calls",
    "tools",
    "tool",
}

TOOL_ROLES = {"tool", "function"}


def iter_message_texts(messages: Iterable[dict]) -> Iterable[Tuple[str, str, dict]]:
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        text_parts.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        text_parts.append(item["content"])
                elif isinstance(item, str):
                    text_parts.append(item)
            text = "\n".join(text_parts)
        elif isinstance(content, str):
            text = content
        else:
            text = ""
        yield role, text, message


def match_reasons(record: dict) -> List[str]:
    reasons: List[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list):
        return reasons

    for role, text, message in iter_message_texts(messages):
        if role.lower() in TOOL_ROLES:
            reasons.append("explicit_tool_role")
            break

        if any(key in message for key in STRUCTURED_TOOL_KEYS):
            reasons.append("structured_tool_field")
            break

        if role.lower() == "system" and any(p.search(text) for p in AGENT_SYSTEM_PATTERNS):
            reasons.append("agentic_system_prompt")
            break

    for role, text, message in iter_message_texts(messages):
        if role.lower() == "assistant" and any(p.search(text) for p in ASSISTANT_TOOL_INTENT_PATTERNS):
            reasons.append("assistant_tool_intent")
            break

    serialized = json.dumps(record, ensure_ascii=False)
    if any(token in serialized for token in ('"tool_calls"', '"function_call"', '"function_calls"', '"role": "tool"', '"role":"tool"')):
        reasons.append("serialized_tool_trace")

    deduped: List[str] = []
    seen = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    return deduped


def extract_agent_tool_use(input_path: Path, output_path: Path, stats_path: Path) -> Dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    reason_counter: Counter[str] = Counter()

    total_bytes = input_path.stat().st_size
    progress = tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Extracting agent tool use",
    )

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        for total, line in enumerate(fin, 1):
            progress.update(len(line.encode("utf-8")))
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                reason_counter["json_decode_error"] += 1
                continue

            reasons = match_reasons(record)
            if not reasons:
                continue

            record["_agent_tool_use_match_reasons"] = reasons
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            reason_counter.update(reasons)
    progress.close()

    stats = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_records": total,
        "kept_records": kept,
        "keep_ratio": round(kept / total, 6) if total else 0.0,
        "reason_counts": dict(reason_counter),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract agent/tool-use style samples from Step-3.5-Flash-SFT jsonl.")
    parser.add_argument(
        "--input",
        default="/opt/users/ye/data/step3p5_flash_sft_ms_swift.jsonl",
        help="Source jsonl path.",
    )
    parser.add_argument(
        "--output",
        default="/mnt/code/yehangcheng/all_data/sft_data/Step-3.5-Flash-SFT/step3p5_flash_sft_ms_swift.agent_tool_use.jsonl",
        help="Filtered jsonl output path.",
    )
    parser.add_argument(
        "--stats",
        default="/mnt/code/yehangcheng/all_data/sft_data/Step-3.5-Flash-SFT/step3p5_flash_sft_ms_swift.agent_tool_use.stats.json",
        help="Stats json output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = extract_agent_tool_use(
        input_path=Path(args.input),
        output_path=Path(args.output),
        stats_path=Path(args.stats),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
