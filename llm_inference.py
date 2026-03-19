#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sglang_runner_simple_direct_user.py

说明：
- 默认不使用 prompt_map / build_prompt；
  直接取每条记录中最后一轮 user content（extract_text_from_orig(..., field="prompt")）作为输入。
- 新增 --use-multi-turn / --max-turns：
  开启 --use-multi-turn 且 --use-openai-client 时，将从 orig 中解析 conversations/messages 等多轮对话，
  转成 OpenAI ChatCompletions 的 messages=[{"role":...,"content":...}] 直接调用。
  （requests 路径仍沿用单文本 payload={"text":...}；若你也要 requests 多轮，请按你网关协议自行扩展。）
- 支持 --num-samples 控制每条样本调用模型次数，结果以 list 形式存入 llm_response（写入 parquet 前统一序列化为 JSON 字符串）。
- 结果仅写回原始 DataFrame 的新列 "llm_response"。
- 保持原有并发、重试、openai client / requests 两种调用逻辑。
"""
from __future__ import annotations

import argparse
import bisect
import json
import logging
import random
import time
import re
import shutil
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Optional, List
import sys

import pandas as pd
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

# 默认仅输出告警和错误，避免刷屏；进度由 tqdm 展示。
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sglang_runner_simple_direct_user")
# 抑制 OpenAI/httpx 的成功请求日志（如 HTTP 200），仅保留异常重试相关信息。
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

# 默认最大文本长度（可运行时修改）
DEFAULT_MAX_TEXT_LENGTH = 4000
FAILED_RESULT_PREFIX = "failed_after_"


# ------------------------------
# 基础函数
# ------------------------------
def extract_text_from_orig(orig: Any, field: str = "prompt") -> Optional[str]:
    """提取最后一轮 user（或 assistant）文本"""
    if orig is None:
        return None
    if isinstance(orig, str):
        try:
            orig = json.loads(orig)
        except Exception:
            return orig
    convs = None
    if isinstance(orig, dict):
        convs = orig.get("conversations") or orig.get("conversation") or orig.get("messages") or orig.get("dialogue")
    if convs is None:
        if isinstance(orig, dict) and "content" in orig and "role" in orig:
            return orig.get("content")
        if isinstance(orig, dict):
            for k in ("text", "message", "content", "prompt", "input"):
                if k in orig and isinstance(orig[k], str):
                    return orig[k]
        return None
    last_user, last_assistant = None, None
    for item in convs:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("sender") or item.get("from")
        content = item.get("content") or item.get("text") or item.get("message")
        if role is None:
            continue
        r = str(role).lower()
        if r in ("user", "human", "usr", "客户", "用户"):
            last_user = content
        elif r in ("assistant", "agent", "bot", "客服"):
            last_assistant = content
    return last_user if field == "prompt" else last_assistant


def extract_conversations_from_orig(orig: Any) -> Optional[list]:
    """
    从 orig 中提取 conversations/messages/dialogue 等结构，返回 list[dict]。
    兼容：
      - orig 为 JSON 字符串
      - 键名：conversations / conversation / messages / dialogue
      - 单条 {role, content/text/message}
    """
    if orig is None:
        return None
    if isinstance(orig, str):
        try:
            orig = json.loads(orig)
        except Exception:
            return None

    if isinstance(orig, dict):
        convs = orig.get("conversations") or orig.get("conversation") or orig.get("messages") or orig.get("dialogue")
        if isinstance(convs, list):
            return convs

        if "role" in orig and ("content" in orig or "text" in orig or "message" in orig):
            return [orig]

    return None


def sanitize_and_truncate_text(text: Any, max_len: Optional[int] = None) -> str:
    """清理文本并截断"""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    effective_max = max_len if (max_len is not None) else DEFAULT_MAX_TEXT_LENGTH
    try:
        if len(text) > effective_max:
            return text[:effective_max] + "...(truncated)"
    except Exception:
        return text
    return text


def is_missing_value(value: Any) -> bool:
    if isinstance(value, (list, tuple, dict)):
        return False
    try:
        missing = pd.isna(value)
        if missing is pd.NA:
            return True
        if isinstance(missing, bool):
            return missing
        return False
    except Exception:
        return False


def parse_response_list(value: Any) -> tuple[List[Any], bool]:
    """
    将 llm_response 解析为 list，并返回该字段是否“有值”。
    - has_value=False：字段缺失/NaN/None
    - has_value=True：字段存在（包括 "[]"）
    """
    if isinstance(value, list):
        return list(value), True
    if isinstance(value, tuple):
        return list(value), True
    if value is None or is_missing_value(value):
        return [], False
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return [], False
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed, True
            return [parsed], True
        except Exception:
            return [value], True
    return [value], True


def is_failed_result_item(item: Any) -> bool:
    if item is None:
        return True
    if isinstance(item, dict):
        status = str(item.get("status", "")).lower().strip()
        return status not in ("", "ok", "success")
    if isinstance(item, str):
        s = item.strip()
        if not s:
            return True
        if s.startswith(FAILED_RESULT_PREFIX):
            return True
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                status = str(parsed.get("status", "")).lower().strip()
                if status not in ("", "ok", "success"):
                    return True
        except Exception:
            pass
    return False


def build_retry_slots(existing_results: Optional[List[Any]], target_num_samples: int) -> tuple[List[Any], List[int]]:
    """
    基于已有 llm_response 构造本次需要补跑的槽位。
    - 只补跑明确失败的样本；
    - 若已有部分成功结果但数量不足 num_samples，会补齐缺失槽位；
    - 对 "[]" 这类显式空结果不自动补跑，避免对无有效输入样本无限重试。
    """
    normalized = list(existing_results) if isinstance(existing_results, list) else []
    normalized = normalized[:max(0, target_num_samples)]

    if target_num_samples <= 0:
        return normalized, []

    if len(normalized) == 0:
        return normalized, []

    retry_slots: List[int] = []
    if len(normalized) < target_num_samples:
        normalized.extend([None] * (target_num_samples - len(normalized)))
        retry_slots.extend(idx for idx, item in enumerate(normalized) if item is None)

    for idx in range(min(len(normalized), target_num_samples)):
        if is_failed_result_item(normalized[idx]):
            retry_slots.append(idx)

    retry_slots = sorted(set(retry_slots))
    return normalized[:target_num_samples], retry_slots


def needs_retry_from_stored_value(value: Any, target_num_samples: int) -> tuple[bool, List[Any]]:
    results, has_value = parse_response_list(value)
    if not has_value:
        return True, results
    normalized, retry_slots = build_retry_slots(results, target_num_samples)
    return len(retry_slots) > 0, normalized


def build_openai_messages_from_convs(convs: list, max_text_len: Optional[int] = None, max_turns: int = 0) -> List[Dict[str, str]]:
    """
    将数据里的 convs 转为 OpenAI ChatCompletion 的 messages=[{role, content}, ...]
    - role 兼容：role/sender/from
    - content 兼容：content/text/message
    - max_turns>0 时仅保留最后 max_turns 条消息（按消息条数，不区分 user/assistant）
    """
    if not isinstance(convs, list) or len(convs) == 0:
        return []

    items = convs[-max_turns:] if (isinstance(max_turns, int) and max_turns > 0 and len(convs) > max_turns) else convs
    messages: List[Dict[str, str]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        role = item.get("role") or item.get("sender") or item.get("from")
        content = item.get("content") or item.get("text") or item.get("message")

        if role is None or content is None:
            continue

        r = str(role).lower().strip()

        # 角色归一化到 OpenAI 允许的：system/user/assistant
        if r in ("system", "sys"):
            rr = "system"
        elif r in ("assistant", "agent", "bot", "客服"):
            rr = "assistant"
        elif r in ("user", "human", "usr", "客户", "用户"):
            rr = "user"
        else:
            # 未知角色，默认当 user（也可以改为跳过）
            rr = "user"

        msg_text = sanitize_and_truncate_text(content, max_len=max_text_len)
        if msg_text.strip() == "":
            continue

        messages.append({"role": rr, "content": msg_text})

    return messages


def try_parse_json_from_text(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """尝试从字符串中解析 JSON"""
    if text is None:
        return None
    if isinstance(text, dict):
        return text
    t = text.strip()
    try:
        parsed = json.loads(t)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        for fix in (lambda x: x, lambda x: x.replace("'", '"')):
            try:
                parsed = json.loads(fix(candidate))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return None


def parse_base_urls(base_url: str) -> List[str]:
    """
    支持通过 '+' 传入多个 base_url，例如：
    http://10.16.80.154:7777+http://10.16.80.150:7777
    """
    if not isinstance(base_url, str):
        raise ValueError("base_url 必须是字符串")

    urls = [item.strip() for item in base_url.split("+") if item.strip()]
    if not urls:
        raise ValueError("base_url 不能为空")
    return urls


def parse_ratio_value(raw_ratio: str) -> float:
    """支持 '2/3' 或 '0.66' 形式的比例。"""
    ratio = raw_ratio.strip()
    if not ratio:
        raise ValueError("比例不能为空")
    if "/" in ratio:
        num_s, den_s = ratio.split("/", 1)
        num = float(num_s.strip())
        den = float(den_s.strip())
        if den == 0:
            raise ValueError("比例分母不能为 0")
        value = num / den
    else:
        value = float(ratio)
    if value <= 0:
        raise ValueError("比例必须大于 0")
    return value


def parse_base_url_weights(base_url_weights: str) -> Dict[str, float]:
    """
    解析按 '+' 分隔的权重配置，例如：
    http://10.16.80.150:6027:2/3+http://10.16.80.154:6027:1/3
    """
    weights: Dict[str, float] = {}
    items = [item.strip() for item in base_url_weights.split("+") if item.strip()]
    if not items:
        raise ValueError("base_url_weights 不能为空")
    for item in items:
        if ":" not in item:
            raise ValueError(f"权重项格式错误（缺少比例）: {item}")
        url, ratio_raw = item.rsplit(":", 1)
        url = url.strip()
        if not url:
            raise ValueError(f"权重项 URL 为空: {item}")
        weights[url] = parse_ratio_value(ratio_raw)
    return weights


def build_weighted_base_urls(base_urls: List[str], base_url_weights: Optional[str]) -> List[Dict[str, Any]]:
    """
    组装带权重的 base_url 列表。
    - 未传 base_url_weights 时默认等权重。
    - 传入后仅保留配置了权重且出现在 base_urls 里的 URL，未配置 URL 会被忽略。
    - 最终权重会做归一化，保证总和为 1。
    """
    weighted: List[Dict[str, Any]] = []
    if base_url_weights:
        weight_map = parse_base_url_weights(base_url_weights)
        for url in base_urls:
            if url in weight_map:
                weighted.append({"url": url, "weight": float(weight_map[url])})
    else:
        for url in base_urls:
            weighted.append({"url": url, "weight": 1.0})

    if not weighted:
        if base_url_weights:
            raise ValueError("base_url_weights 与 base_url 没有可用的 URL 交集")
        raise ValueError("没有可用的 base_url")

    total_weight = sum(float(item["weight"]) for item in weighted)
    if total_weight <= 0:
        raise ValueError("base_url 权重总和必须大于 0")
    for item in weighted:
        item["weight"] = float(item["weight"]) / total_weight
    return weighted


def choose_base_url(base_urls: List[Dict[str, Any]]) -> str:
    """每次请求按权重随机选择一个 base_url，用于多线程分流。"""
    if not base_urls:
        raise ValueError("base_urls 不能为空")
    urls = [str(item["url"]) for item in base_urls]
    weights = [float(item["weight"]) for item in base_urls]
    return random.choices(urls, weights=weights, k=1)[0]


# ------------------------------
# 模型请求
# ------------------------------
def build_extra_body(extra_body: Optional[dict], enable_thinking: bool) -> dict:
    """
    统一构造 extra_body，支持通过参数开关思考模式。
    """
    extra: Dict[str, Any] = dict(extra_body) if isinstance(extra_body, dict) else {}
    extra.setdefault("repetition_penalty", 1.05)
    ctk = extra.get("chat_template_kwargs")
    if not isinstance(ctk, dict):
        ctk = {}
    ctk["enable_thinking"] = enable_thinking
    extra["chat_template_kwargs"] = ctk
    return extra


def forward_local_api_openai(message: str, api_key: Optional[str], base_urls: List[Dict[str, Any]], model: str,
                             temperature: float, max_retries: int, sleep_duration: float,
                             extra_body: Optional[dict], enable_thinking: bool):
    try:
        from openai import OpenAI
    except Exception as e:
        return {"status": "import_error", "text": None, "raw": str(e)}
    stop_tokens = ["<|eot_id|>", "<|im_end|>", "</s>", "<|endoftext|>", "</answer>"]
    extra = build_extra_body(extra_body, enable_thinking=enable_thinking)
    attempt = 0
    selected_base_url = ""
    while attempt <= max_retries:
        try:
            selected_base_url = choose_base_url(base_urls)
            base = selected_base_url if selected_base_url.endswith("/v1") else selected_base_url.rstrip("/") + "/v1"
            client = OpenAI(api_key=api_key, base_url=base)
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
                stop=stop_tokens,
                max_tokens=12000,
                extra_body=extra,
            )
            text_out = completion.choices[0].message.content
            if text_out and "</think>" in text_out:
                text_out = text_out.split("</think>")[-1]
            return {"status": "ok", "text": text_out, "raw": completion}
        except Exception as e:
            logger.warning("openai client 调用异常(base_url=%s): %s", selected_base_url, e)
            attempt += 1
            if attempt > max_retries:
                break
            time.sleep(sleep_duration)
    return {"status": "failed", "text": None, "raw": f"failed_after_{max_retries}"}


def forward_local_api_openai_messages(messages: List[Dict[str, str]], api_key: Optional[str], base_urls: List[Dict[str, Any]], model: str,
                                      temperature: float, max_retries: int, sleep_duration: float,
                                      extra_body: Optional[dict], enable_thinking: bool):
    """
    多轮 messages 调用 OpenAI ChatCompletions。
    """
    try:
        from openai import OpenAI
    except Exception as e:
        return {"status": "import_error", "text": None, "raw": str(e)}

    stop_tokens = ["<|eot_id|>", "<|im_end|>", "</s>", "<|endoftext|>", "</answer>"]
    extra = build_extra_body(extra_body, enable_thinking=enable_thinking)

    attempt = 0
    selected_base_url = ""
    while attempt <= max_retries:
        try:
            selected_base_url = choose_base_url(base_urls)
            base = selected_base_url if selected_base_url.endswith("/v1") else selected_base_url.rstrip("/") + "/v1"
            client = OpenAI(api_key=api_key, base_url=base)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stop=stop_tokens,
                max_tokens=12000,
                extra_body=extra,
            )
            text_out = completion.choices[0].message.content
            return {"status": "ok", "text": text_out, "raw": completion}
        except Exception as e:
            logger.warning("openai client(多轮) 调用异常(base_url=%s): %s", selected_base_url, e)
            attempt += 1
            if attempt > max_retries:
                break
            time.sleep(sleep_duration)

    return {"status": "failed", "text": None, "raw": f"failed_after_{max_retries}"}


def forward_local_api_requests(message: str, endpoints: List[Dict[str, Any]], headers: Dict[str, str],
                               timeout: float, max_retries: int, sleep_duration: float):
    payload = {"text": message}
    attempt = 0
    endpoint = ""
    while attempt <= max_retries:
        try:
            endpoint = choose_base_url(endpoints)
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            try:
                raw = resp.json()
            except Exception:
                raw = resp.text
            if resp.status_code == 200:
                text = None
                if isinstance(raw, dict):
                    text = raw.get("text") or raw.get("result") or raw.get("output") or raw.get("reply")
                if text is None:
                    text = str(raw)
                return {"status": "ok", "text": text, "raw": raw}
            else:
                logger.warning("非200响应 %s", resp.status_code)
                attempt += 1
                time.sleep(sleep_duration)
        except Exception as e:
            logger.warning("requests 调用异常(endpoint=%s): %s", endpoint, e)
            attempt += 1
            time.sleep(sleep_duration)
    return {"status": "failed", "text": None, "raw": f"failed_after_{max_retries}"}


# ------------------------------
# Worker 函数
# ------------------------------
def worker(row_idx: int, row: pd.Series, field: str, use_openai: bool, api_key: Optional[str],
           base_urls: List[Dict[str, Any]], model: str, timeout: float, max_retries: int, sleep_duration: float,
           headers: Dict[str, str], extra_body: Optional[dict], num_samples: int,
           use_multi_turn: bool, max_turns: int, max_text_len: int, enable_thinking: bool,
           existing_results: Optional[List[Any]] = None):
    """每条记录可重复生成多次，返回 (row_idx, [result1, result2, ...])"""
    orig = row.get("orig")

    messages = None
    msg = None

    # 1) 组装输入：多轮 or 单轮（最后一轮）
    if use_openai and use_multi_turn:
        convs = extract_conversations_from_orig(orig)
        if not convs:
            return (row_idx, [])
        messages = build_openai_messages_from_convs(convs, max_text_len=max_text_len, max_turns=max_turns)
        if not messages:
            return (row_idx, [])
    else:
        text = extract_text_from_orig(orig, field=field)
        if text is None or (isinstance(text, str) and text.strip() == ""):
            return (row_idx, [])
        msg = sanitize_and_truncate_text(text, max_len=max_text_len)

    results_list, retry_slots = build_retry_slots(existing_results, num_samples)
    if existing_results is None:
        retry_slots = list(range(num_samples))
        results_list = []

    for slot_idx in retry_slots:
        if use_openai:
            if use_multi_turn:
                res = forward_local_api_openai_messages(
                    messages=messages,  # type: ignore[arg-type]
                    api_key=api_key,
                    base_urls=base_urls,
                    model=model,
                    temperature=0,
                    max_retries=max_retries,
                    sleep_duration=sleep_duration,
                    extra_body=extra_body,
                    enable_thinking=enable_thinking,
                )
            else:
                res = forward_local_api_openai(
                    message=msg or "",
                    api_key=api_key,
                    base_urls=base_urls,
                    model=model,
                    temperature=0,
                    max_retries=max_retries,
                    sleep_duration=sleep_duration,
                    extra_body=extra_body,
                    enable_thinking=enable_thinking,
                )
        else:
            # requests 分支仍按单文本 payload={"text":...} 发送；
            # 若你要 requests 多轮，请按网关协议改造 forward_local_api_requests 及此处 payload。
            res = forward_local_api_requests(
                message=msg or "",
                endpoints=base_urls,
                headers=headers,
                timeout=timeout,
                max_retries=max_retries,
                sleep_duration=sleep_duration
            )

        text_out = res.get("text")

        if text_out is not None:
            output_value = text_out
        else:
            output_value = res.get("raw")

        if existing_results is None:
            results_list.append(output_value)
        else:
            while len(results_list) <= slot_idx:
                results_list.append(None)
            results_list[slot_idx] = output_value
        time.sleep(0.1)  # 轻微间隔，防止速率触发

    return (row_idx, results_list)


# ------------------------------
# 序列化辅助（避免 pyarrow 写 parquet 时列类型混合出错）
# ------------------------------
def safe_serialize(v: Any) -> str:
    """
    将任意 python 对象转换为 JSON 字符串以便 parquet 写入（使用 pyarrow）。
    - 对 None 返回 "[]"
    - 对已是字符串的，尝试 json.loads 判断是不是已经是 JSON 字符串；否则直接 json.dumps(str)
    - 捕获不可序列化对象并降级为其 str() 表示
    """
    if v is None:
        return "[]"
    # 如果已经是字符串，尽量保证是 JSON 格式（若不是则作为字符串序列化）
    if isinstance(v, str):
        s = v.strip()
        # 若看起来像 JSON（以 [ 或 { 开头），尝试解析，若成功直接返回原字符串（保证原样）
        if s.startswith("{") or s.startswith("["):
            try:
                _ = json.loads(s)
                return s
            except Exception:
                pass
        return json.dumps(s, ensure_ascii=False)
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        try:
            return json.dumps(str(v), ensure_ascii=False)
        except Exception:
            return "[]"


# ------------------------------
# 主函数
# ------------------------------
def load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state_path: Path, next_row: int, part_id: int, total_rows: int, completed: bool = False) -> None:
    state = {
        "next_row": int(next_row),
        "part_id": int(part_id),
        "total_rows": int(total_rows),
        "completed": bool(completed),
        "updated_at": int(time.time()),
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_parts_to_output(parts_dir: Path, output_path: Path) -> None:
    part_files = sorted(parts_dir.glob("part_*.parquet"))
    if not part_files:
        raise RuntimeError(f"没有可合并的分片: {parts_dir}")
    if output_path.exists():
        output_path.unlink()

    writer = None
    try:
        for pf in part_files:
            table = pq.read_table(pf)
            if writer is None:
                writer = pq.ParquetWriter(output_path.as_posix(), table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def build_part_infos(parts_dir: Path) -> List[Dict[str, Any]]:
    infos: List[Dict[str, Any]] = []
    start_row = 0
    for pf in sorted(parts_dir.glob("part_*.parquet")):
        num_rows = pq.ParquetFile(pf).metadata.num_rows
        infos.append(
            {
                "path": pf,
                "start_row": start_row,
                "end_row": start_row + num_rows,
                "num_rows": num_rows,
            }
        )
        start_row += num_rows
    return infos


def locate_part_info(row_idx: int, part_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not part_infos:
        raise ValueError("part_infos 不能为空")
    starts = [int(info["start_row"]) for info in part_infos]
    pos = bisect.bisect_right(starts, row_idx) - 1
    if pos < 0 or pos >= len(part_infos):
        raise IndexError(f"row_idx={row_idx} 不在任何分片范围内")
    info = part_infos[pos]
    if not (int(info["start_row"]) <= row_idx < int(info["end_row"])):
        raise IndexError(f"row_idx={row_idx} 不在任何分片范围内")
    return info


def repair_failed_rows(
    df: pd.DataFrame,
    parts_dir: Path,
    field: str,
    use_openai: bool,
    api_key: Optional[str],
    base_urls: List[Dict[str, Any]],
    model: str,
    timeout: float,
    max_retries: int,
    sleep_duration: float,
    headers: Dict[str, str],
    extra_body: Optional[dict],
    num_samples: int,
    use_multi_turn: bool,
    max_turns: int,
    max_text_len: int,
    enable_thinking: bool,
    concurrency: int,
) -> int:
    if not parts_dir.exists():
        logger.warning("未找到分片目录，跳过失败样本补跑: %s", parts_dir)
        return 0

    part_infos = build_part_infos(parts_dir)
    if not part_infos:
        return 0

    candidates: List[tuple[int, List[Any]]] = []
    for info in part_infos:
        table = pq.read_table(info["path"], columns=["llm_response"])
        values = table.column("llm_response").to_pylist()
        start_row = int(info["start_row"])
        for offset, value in enumerate(values):
            needs_retry, parsed_results = needs_retry_from_stored_value(value, num_samples)
            if needs_retry:
                candidates.append((start_row + offset, parsed_results))

    if not candidates:
        logger.info("未检测到失败样本，无需补跑。")
        return 0

    logger.warning("检测到 %d 条失败/未完成样本，开始补跑。", len(candidates))
    repaired_results: Dict[int, List[Any]] = {}
    max_workers = max(1, concurrency)

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        future_map = {
            exe.submit(
                worker,
                row_idx,
                df.iloc[row_idx],
                field,
                use_openai,
                api_key,
                base_urls,
                model,
                timeout,
                max_retries,
                sleep_duration,
                headers,
                extra_body,
                num_samples,
                use_multi_turn,
                max_turns,
                max_text_len,
                enable_thinking,
                existing_results,
            ): row_idx
            for row_idx, existing_results in candidates
        }

        with tqdm(total=len(future_map), desc="RepairFailed", unit="row") as pbar:
            while future_map:
                done, _ = wait(set(future_map.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    row_idx = future_map.pop(fut)
                    try:
                        _, new_results = fut.result()
                        repaired_results[row_idx] = new_results if isinstance(new_results, list) else [new_results]
                    except Exception as e:
                        logger.exception("失败样本补跑异常 row_idx=%s err=%s", row_idx, e)
                    pbar.update(1)

    if not repaired_results:
        return 0

    updates_by_part: Dict[Path, List[tuple[int, List[Any]]]] = {}
    for row_idx, results in repaired_results.items():
        info = locate_part_info(row_idx, part_infos)
        local_idx = row_idx - int(info["start_row"])
        part_path = info["path"]
        updates_by_part.setdefault(part_path, []).append((local_idx, results))

    for info in part_infos:
        part_path = info["path"]
        updates = updates_by_part.get(part_path)
        if not updates:
            continue
        part_df = pd.read_parquet(part_path, engine="pyarrow")
        for local_idx, results in updates:
            part_df.at[local_idx, "llm_response"] = safe_serialize(results)
        part_df.to_parquet(part_path, index=False, engine="pyarrow")

    logger.warning("失败样本补跑完成，已回写 %d 条记录。", len(repaired_results))
    return len(repaired_results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--field", choices=["prompt", "response"], default="prompt")
    p.add_argument("--use-openai-client", action="store_true")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default="http://localhost:8035/v1")
    p.add_argument(
        "--base-url-weights",
        default=None,
        help="按 '+' 分隔 URL 与比例，格式如 http://a:7777:2/3+http://b:7777:1/3；未配置时默认等权"
    )
    p.add_argument("--model", default="Qwen3-30B-A3B-Base-general_totals_202w_concat_lmsys_278w-ep3")
    p.add_argument("--concurrency", type=int, default=200)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--sleep-duration", type=float, default=5.0)
    p.add_argument("--header", action="append")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--extra-body", type=str, default=None)
    p.add_argument("--max-text-len", type=int, default=DEFAULT_MAX_TEXT_LENGTH)
    p.add_argument("--num-samples", type=int, default=1, help="每条样本生成次数")
    p.add_argument("--thinking-mode", choices=["on", "off"], default="off",
                   help="控制 chat_template_kwargs.enable_thinking")
    p.add_argument("--save-every", type=int, default=2000, help="每累计多少条结果落盘一个增量分片")
    p.add_argument("--resume", dest="resume", action="store_true", default=True, help="开启断点续跑")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="禁用断点续跑并清理旧检查点")
    p.add_argument("--retry-failed", dest="retry_failed", action="store_true", default=True,
                   help="重启时扫描已写入分片中的失败样本并补跑")
    p.add_argument("--no-retry-failed", dest="retry_failed", action="store_false",
                   help="禁用失败样本补跑")

    # 新增：多轮开关与截断
    p.add_argument(
        "--use-multi-turn",
        action="store_true",
        help="开启后（仅对 --use-openai-client 生效）使用 orig 中 conversations/messages 多轮对话作为 OpenAI messages 调用"
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="多轮模式下仅保留最后 N 条消息；0 表示不截断"
    )

    args = p.parse_args()

    # header / body 解析
    headers = {}
    if args.header:
        for h in args.header:
            if ":" in h:
                k, v = h.split(":", 1)
                headers[k.strip()] = v.strip()
    extra_body = None
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except Exception:
            extra_body = None
    enable_thinking = args.thinking_mode == "on"
    base_urls = parse_base_urls(args.base_url)
    weighted_base_urls = build_weighted_base_urls(base_urls, args.base_url_weights)

    logger.info("读取 parquet：%s", args.input)
    df = pd.read_parquet(args.input, engine="pyarrow")
    logger.info("数据行数：%d", len(df))
    if args.sample and args.sample > 0:
        df = df.iloc[:args.sample].copy()
        logger.info("sample 模式，处理前 %d 行", len(df))
    df = df.reset_index(drop=True)   # 放在 sample 之后更稳
    globals()["DEFAULT_MAX_TEXT_LENGTH"] = args.max_text_len

    n = len(df)

    if args.use_multi_turn and not args.use_openai_client:
        logger.warning("--use-multi-turn 已开启，但未使用 --use-openai-client；requests 路径仍将按单文本调用。")
    logger.info("已配置 %d 个 base_url，用于随机分流: %s", len(base_urls), weighted_base_urls)

    output_path = Path(args.output)
    ckpt_dir = Path(f"{args.output}.ckpt")
    parts_dir = ckpt_dir / "parts"
    state_path = ckpt_dir / "state.json"

    if not args.resume:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        if output_path.exists():
            output_path.unlink()

    parts_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(state_path) if args.resume else {}
    start_row = int(state.get("next_row", 0)) if args.resume else 0
    part_id = int(state.get("part_id", 0)) if args.resume else 0
    if start_row < 0:
        start_row = 0
    if start_row > n:
        start_row = n

    logger.info("断点续跑状态: start_row=%d part_id=%d total=%d", start_row, part_id, n)
    did_process_new_rows = start_row < n

    save_every = max(1, int(args.save_every))
    max_in_flight = max(args.concurrency * 4, args.concurrency)
    next_submit = start_row
    next_write = start_row
    in_flight: Dict[Any, int] = {}
    pending_results: Dict[int, List[Any]] = {}
    flush_buffer: List[Dict[str, Any]] = []

    def flush_buffer_to_part() -> None:
        nonlocal part_id, flush_buffer
        if not flush_buffer:
            return
        part_path = parts_dir / f"part_{part_id:07d}.parquet"
        pd.DataFrame(flush_buffer).to_parquet(part_path, index=False, engine="pyarrow")
        part_id += 1
        flush_buffer = []
        save_state(state_path, next_row=next_write, part_id=part_id, total_rows=n, completed=False)

    logger.info("启动线程池：%d 并发，增量落盘每 %d 条", args.concurrency, save_every)
    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        with tqdm(total=n - start_row, desc="Processing", unit="row") as pbar:
            while next_submit < n or in_flight:
                while next_submit < n and len(in_flight) < max_in_flight:
                    row = df.iloc[next_submit]
                    fut = exe.submit(
                        worker, next_submit, row, args.field, args.use_openai_client, args.api_key,
                        weighted_base_urls, args.model, args.timeout, args.max_retries,
                        args.sleep_duration, headers, extra_body, args.num_samples,
                        args.use_multi_turn, args.max_turns, args.max_text_len, enable_thinking
                    )
                    in_flight[fut] = next_submit
                    next_submit += 1

                if not in_flight:
                    continue

                done, _ = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    idx = in_flight.pop(fut)
                    try:
                        r = fut.result()
                        if r is None:
                            pending_results[idx] = []
                        else:
                            ri, val = r
                            pending_results[ri] = val if isinstance(val, list) else [val]
                    except Exception as e:
                        logger.exception("future 异常 %s", e)
                        pending_results[idx] = []

                while next_write in pending_results:
                    vals = pending_results.pop(next_write)
                    row_dict = df.iloc[next_write].to_dict()
                    row_dict["llm_response"] = safe_serialize(vals)
                    flush_buffer.append(row_dict)
                    next_write += 1
                    pbar.update(1)
                    if len(flush_buffer) >= save_every:
                        flush_buffer_to_part()

            flush_buffer_to_part()

    save_state(state_path, next_row=next_write, part_id=part_id, total_rows=n, completed=(next_write == n))
    if next_write != n:
        raise RuntimeError(f"任务未完成: next_write={next_write}, total={n}")

    repaired_rows = 0
    if args.retry_failed:
        repaired_rows = repair_failed_rows(
            df=df,
            parts_dir=parts_dir,
            field=args.field,
            use_openai=args.use_openai_client,
            api_key=args.api_key,
            base_urls=weighted_base_urls,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            sleep_duration=args.sleep_duration,
            headers=headers,
            extra_body=extra_body,
            num_samples=args.num_samples,
            use_multi_turn=args.use_multi_turn,
            max_turns=args.max_turns,
            max_text_len=args.max_text_len,
            enable_thinking=enable_thinking,
            concurrency=args.concurrency,
        )

    if did_process_new_rows or repaired_rows > 0 or not output_path.exists():
        logger.info("开始合并到最终 parquet: %s", output_path)
        merge_parts_to_output(parts_dir=parts_dir, output_path=output_path)
    else:
        logger.info("输出已存在且无需更新，跳过最终合并: %s", output_path)

    save_state(state_path, next_row=n, part_id=part_id, total_rows=n, completed=True)
    logger.info("全部完成。输出文件：%s", output_path)


if __name__ == "__main__":
    main()
