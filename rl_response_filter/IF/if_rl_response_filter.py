#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import logging
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("if_rl_response_filter")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

STOP_TOKENS = ["<|eot_id|>", "<|im_end|>", "</s>", "<|endoftext|>", "</answer>"]
IFEVAL_DIR = "/mnt/code/yehangcheng/ms-swift/plugin/IFeval"

if IFEVAL_DIR not in sys.path:
    sys.path.insert(0, IFEVAL_DIR)

import ifeval_instructions_registry as instructions_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 IF RL 数据生成 response，并按 IFeval 规则筛选严格满足指令的样本。")
    parser.add_argument(
        "--input-parquet",
        default="/mnt/code/yehangcheng/all_data/rl_data_repo/IF/Nemotron-post-training/if_rl_dataset_train_swift_len12k_clean.parquet",
    )
    parser.add_argument(
        "--output-success",
        default="/mnt/code/yehangcheng/Intruct_augment/pipline/rl_response_filter/IF/if_rl_following_success.parquet",
    )
    parser.add_argument(
        "--output-failed",
        default="/mnt/code/yehangcheng/Intruct_augment/pipline/rl_response_filter/IF/if_rl_following_failed.parquet",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--base-url", required=True, help="支持多个 URL，用 '+' 分隔。")
    parser.add_argument(
        "--base-url-weights",
        default=None,
        help="格式如 http://127.0.0.1:8000:1+http://127.0.0.1:8001:2",
    )
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--request-max-retries", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3, help="单条样本最大生成尝试次数。")
    parser.add_argument("--sleep-duration", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--thinking-mode", choices=["on", "off"], default="off")
    parser.add_argument("--extra-body", type=str, default=None)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=200)
    return parser.parse_args()


def ensure_nltk_resources() -> None:
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
    except Exception as exc:
        logger.warning("nltk 资源检查失败，运行时可能影响部分 IF 规则: %s", exc)


def parse_ratio_value(raw_ratio: str) -> float:
    ratio = raw_ratio.strip()
    if "/" in ratio:
        num_s, den_s = ratio.split("/", 1)
        value = float(num_s.strip()) / float(den_s.strip())
    else:
        value = float(ratio)
    if value <= 0:
        raise ValueError(f"非法权重: {raw_ratio}")
    return value


def parse_base_urls(base_url: str) -> List[str]:
    urls = [item.strip() for item in base_url.split("+") if item.strip()]
    if not urls:
        raise ValueError("base_url 不能为空")
    return urls


def parse_base_url_weights(base_url_weights: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in [x.strip() for x in base_url_weights.split("+") if x.strip()]:
        url, ratio_raw = item.rsplit(":", 1)
        weights[url.strip()] = parse_ratio_value(ratio_raw)
    return weights


def build_weighted_base_urls(base_urls: List[str], base_url_weights: Optional[str]) -> List[Dict[str, Any]]:
    if not base_url_weights:
        return [{"url": url, "weight": 1.0 / len(base_urls)} for url in base_urls]

    weight_map = parse_base_url_weights(base_url_weights)
    weighted = [{"url": url, "weight": float(weight_map[url])} for url in base_urls if url in weight_map]
    if not weighted:
        raise ValueError("base_url_weights 与 base_url 没有交集")
    total = sum(item["weight"] for item in weighted)
    return [{"url": item["url"], "weight": item["weight"] / total} for item in weighted]


def choose_base_url(weighted_base_urls: List[Dict[str, Any]]) -> str:
    urls = [item["url"] for item in weighted_base_urls]
    weights = [item["weight"] for item in weighted_base_urls]
    return random.choices(urls, weights=weights, k=1)[0]


def build_extra_body(extra_body: Optional[dict], enable_thinking: bool) -> dict:
    extra: Dict[str, Any] = dict(extra_body) if isinstance(extra_body, dict) else {}
    extra.setdefault("repetition_penalty", 1.05)
    chat_template_kwargs = extra.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    chat_template_kwargs["enable_thinking"] = enable_thinking
    extra["chat_template_kwargs"] = chat_template_kwargs
    return extra


def normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()

    if isinstance(messages, str):
        stripped = messages.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            messages = parsed
        except Exception:
            return [{"role": "user", "content": stripped}]

    if isinstance(messages, dict):
        messages = [messages]

    if not isinstance(messages, list):
        return [{"role": "user", "content": str(messages)}]

    normalized: List[Dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = item.get("content") or item.get("text") or item.get("message")
        if content is None:
            continue
        text = str(content).strip()
        if not text:
            continue
        normalized.append({"role": role, "content": text})
    return normalized


def forward_openai_messages(
    messages: List[Dict[str, str]],
    api_key: str,
    weighted_base_urls: List[Dict[str, Any]],
    model: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
    request_max_retries: int,
    sleep_duration: float,
    extra_body: Optional[dict],
    enable_thinking: bool,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:
        return {"status": "import_error", "text": None, "raw": str(exc)}

    extra = build_extra_body(extra_body, enable_thinking=enable_thinking)
    attempt = 0
    selected_base_url = ""
    while attempt <= request_max_retries:
        try:
            selected_base_url = choose_base_url(weighted_base_urls)
            base = selected_base_url if selected_base_url.endswith("/v1") else selected_base_url.rstrip("/") + "/v1"
            client = OpenAI(api_key=api_key, base_url=base, timeout=timeout)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=STOP_TOKENS,
                extra_body=extra,
            )
            text_out = completion.choices[0].message.content
            if text_out and "</think>" in text_out:
                text_out = text_out.split("</think>")[-1]
            return {"status": "ok", "text": text_out, "raw": completion}
        except Exception as exc:
            logger.warning("请求失败(base_url=%s, attempt=%s): %s", selected_base_url, attempt + 1, exc)
            attempt += 1
            if attempt > request_max_retries:
                break
            time.sleep(sleep_duration)
    return {"status": "failed", "text": None, "raw": f"failed_after_{request_max_retries}"}


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _normalize_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    return value


def parse_extra_info(extra_info: Any) -> Dict[str, Any]:
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        try:
            parsed = json.loads(extra_info)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def evaluate_instruction_following(solution_str: str, extra_info: Dict[str, Any]) -> Dict[str, Any]:
    instruction_id_list = [str(x) for x in _to_list(extra_info.get("instruction_id_list", []))]
    kwargs_list = _to_list(extra_info.get("instruction_kwargs", []))

    if len(instruction_id_list) == 0:
        return {"passed": False, "instruction_results": [], "fail_reason": "missing_instruction_ids"}

    if len(kwargs_list) != len(instruction_id_list):
        return {
            "passed": False,
            "instruction_results": [],
            "fail_reason": "instruction_kwargs_length_mismatch",
        }

    instruction_results: List[Dict[str, Any]] = []
    for idx, inst_id in enumerate(instruction_id_list):
        inst_cls = instructions_registry.INSTRUCTION_DICT[inst_id]
        inst = inst_cls(inst_id)

        raw_kwargs = kwargs_list[idx] or {}
        if not isinstance(raw_kwargs, dict):
            raw_kwargs = {}
        accepted = set(inspect.signature(inst.build_description).parameters.keys())
        filtered_kwargs = {k: _normalize_value(v) for k, v in raw_kwargs.items() if k in accepted}
        inst.build_description(**filtered_kwargs)
        hit = bool(solution_str.strip()) and bool(inst.check_following(solution_str))
        instruction_results.append(
            {
                "instruction_id": inst_id,
                "instruction_kwargs": filtered_kwargs,
                "hit": hit,
            }
        )

    return {
        "passed": bool(instruction_results) and all(item["hit"] for item in instruction_results),
        "instruction_results": instruction_results,
        "fail_reason": "",
    }


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_outputs(
    success_records: List[Dict[str, Any]],
    failed_records: List[Dict[str, Any]],
    success_path: Path,
    failed_path: Path,
) -> None:
    success_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(success_records).to_parquet(success_path, index=False, engine="pyarrow")
    pd.DataFrame(failed_records).to_parquet(failed_path, index=False, engine="pyarrow")


def process_one_row(
    row_idx: int,
    row_dict: Dict[str, Any],
    api_key: str,
    weighted_base_urls: List[Dict[str, Any]],
    model: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
    request_max_retries: int,
    max_attempts: int,
    sleep_duration: float,
    extra_body: Optional[dict],
    enable_thinking: bool,
) -> Dict[str, Any]:
    messages = normalize_messages(row_dict.get("messages"))
    extra_info = parse_extra_info(row_dict.get("extra_info"))
    attempts_log: List[Dict[str, Any]] = []

    if not messages:
        return {
            "row_idx": row_idx,
            "status": "failed",
            "attempts_used": 0,
            "best_response": None,
            "instruction_results": [],
            "all_attempts": attempts_log,
            "fail_reason": "empty_messages",
        }

    for attempt_idx in range(1, max_attempts + 1):
        api_result = forward_openai_messages(
            messages=messages,
            api_key=api_key,
            weighted_base_urls=weighted_base_urls,
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            request_max_retries=request_max_retries,
            sleep_duration=sleep_duration,
            extra_body=extra_body,
            enable_thinking=enable_thinking,
        )
        response_text = api_result.get("text") or ""
        judge_result = evaluate_instruction_following(response_text, extra_info)
        attempts_log.append(
            {
                "attempt": attempt_idx,
                "request_status": api_result.get("status"),
                "response": response_text,
                "instruction_results": judge_result["instruction_results"],
                "passed": judge_result["passed"],
                "fail_reason": judge_result["fail_reason"],
            }
        )
        if judge_result["passed"]:
            return {
                "row_idx": row_idx,
                "status": "success",
                "attempts_used": attempt_idx,
                "best_response": response_text,
                "instruction_results": judge_result["instruction_results"],
                "all_attempts": attempts_log,
                "fail_reason": "",
            }
        if attempt_idx < max_attempts:
            time.sleep(sleep_duration)

    last_attempt = attempts_log[-1] if attempts_log else {}
    return {
        "row_idx": row_idx,
        "status": "failed",
        "attempts_used": max_attempts,
        "best_response": last_attempt.get("response"),
        "instruction_results": last_attempt.get("instruction_results", []),
        "all_attempts": attempts_log,
        "fail_reason": last_attempt.get("fail_reason") or "not_all_instructions_satisfied",
    }


def main() -> None:
    args = parse_args()
    ensure_nltk_resources()

    extra_body = json.loads(args.extra_body) if args.extra_body else None
    enable_thinking = args.thinking_mode == "on"
    input_path = Path(args.input_parquet)
    success_path = Path(args.output_success)
    failed_path = Path(args.output_failed)

    base_urls = parse_base_urls(args.base_url)
    weighted_base_urls = build_weighted_base_urls(base_urls, args.base_url_weights)

    df = pd.read_parquet(input_path, engine="pyarrow")
    if args.sample and args.sample > 0:
        df = df.iloc[: args.sample].copy()
    df = df.reset_index(drop=True)
    total = len(df)

    logger.info("读取 %s，共 %d 条数据", input_path, total)
    logger.info("使用 %d 个 base_url，并发=%d，max_attempts=%d", len(weighted_base_urls), args.concurrency, args.max_attempts)

    success_records: List[Dict[str, Any]] = []
    failed_records: List[Dict[str, Any]] = []
    completed = 0
    next_submit = 0
    in_flight: Dict[Any, int] = {}
    max_in_flight = max(args.concurrency * 4, args.concurrency)

    def append_record(result: Dict[str, Any]) -> None:
        nonlocal completed
        row_idx = int(result["row_idx"])
        row_dict = df.iloc[row_idx].to_dict()
        record = {
            **row_dict,
            "row_idx": row_idx,
            "model": args.model,
            "generated_response": result["best_response"],
            "attempts_used": int(result["attempts_used"]),
            "instruction_results": safe_json_dumps(result["instruction_results"]),
            "all_attempts": safe_json_dumps(result["all_attempts"]),
            "fail_reason": result["fail_reason"],
        }
        if result["status"] == "success":
            success_records.append(record)
        else:
            failed_records.append(record)
        completed += 1
        if completed % max(1, args.save_every) == 0 or completed == total:
            write_outputs(success_records, failed_records, success_path, failed_path)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        with tqdm(total=total, desc="Filtering", unit="row") as pbar:
            while next_submit < total or in_flight:
                while next_submit < total and len(in_flight) < max_in_flight:
                    row_dict = df.iloc[next_submit].to_dict()
                    future = executor.submit(
                        process_one_row,
                        next_submit,
                        row_dict,
                        args.api_key,
                        weighted_base_urls,
                        args.model,
                        args.timeout,
                        args.temperature,
                        args.max_tokens,
                        args.request_max_retries,
                        args.max_attempts,
                        args.sleep_duration,
                        extra_body,
                        enable_thinking,
                    )
                    in_flight[future] = next_submit
                    next_submit += 1

                if not in_flight:
                    continue

                done, _ = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future, None)
                    result = future.result()
                    append_record(result)
                    pbar.update(1)

    write_outputs(success_records, failed_records, success_path, failed_path)
    logger.info("处理完成，成功 %d 条，失败 %d 条", len(success_records), len(failed_records))
    logger.info("成功文件: %s", success_path)
    logger.info("失败文件: %s", failed_path)


if __name__ == "__main__":
    main()
