#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("math_rl_generate_filter")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]

STOP_TOKENS = ["<|eot_id|>", "<|im_end|>", "</s>", "<|endoftext|>", "</answer>"]
ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 math RL 数据生成答案并用 ground truth 过滤正确 response。")
    parser.add_argument(
        "--input-parquet",
        default="/mnt/code/yehangcheng/all_data/rl_data_repo/math-rlvr-unified.math_difficulty.gt8.parquet",
    )
    parser.add_argument(
        "--output-success",
        default="/mnt/code/yehangcheng/Intruct_augment/pipline/inference_res/math_rl_correct_responses.parquet",
    )
    parser.add_argument(
        "--output-failed",
        default="/mnt/code/yehangcheng/Intruct_augment/pipline/inference_res/math_rl_failed_responses.parquet",
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
    parser.add_argument("--request-max-retries", type=int, default=2, help="单次请求内部重试次数。")
    parser.add_argument("--max-attempts", type=int, default=3, help="单条样本最大生成尝试次数。")
    parser.add_argument("--sleep-duration", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--thinking-mode", choices=["on", "off"], default="off")
    parser.add_argument("--extra-body", type=str, default=None)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=200)
    return parser.parse_args()


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


def normalize_messages(prompt: Any) -> List[Dict[str, str]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()

    if isinstance(prompt, str):
        stripped = prompt.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            prompt = parsed
        except Exception:
            return [{"role": "user", "content": stripped}]

    if isinstance(prompt, dict):
        prompt = [prompt]

    if not isinstance(prompt, list):
        return [{"role": "user", "content": str(prompt)}]

    messages: List[Dict[str, str]] = []
    for item in prompt:
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
        messages.append({"role": role, "content": text})
    return messages


def ensure_answer_instruction(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not messages:
        return messages
    updated = list(messages)
    last_message = dict(updated[-1])
    content = last_message["content"]
    if "Answer:" not in content:
        content = (
            content.rstrip()
            + "\n\nPlease reason step by step, and provide your final answer on the last line in the following format:\n"
            + "Answer: xxx"
        )
        last_message["content"] = content
        updated[-1] = last_message
    return updated


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


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed{")
    if idx < 0:
        idx = string.rfind("boxed{")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1]
    if s.startswith("boxed{") and s.endswith("}"):
        return s[len("boxed{") : -1]
    raise ValueError(f"box error: {s}")


def normalize_final_answer(final_answer: str) -> str:
    final_answer = str(final_answer).split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def normalize_ground_truth(gt: Any) -> str:
    gt_text = "" if gt is None else str(gt).strip()
    boxed = last_boxed_only_string(gt_text)
    if boxed is not None:
        return normalize_final_answer(remove_boxed(boxed))
    return normalize_final_answer(gt_text)


def extract_final_answer(solution_str: Any) -> Tuple[str, str]:
    text = "" if solution_str is None else str(solution_str)
    answer_matches = ANSWER_PATTERN.findall(text)
    if answer_matches:
        return answer_matches[-1].strip(), "answer"
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        try:
            return remove_boxed(boxed).strip(), "boxed"
        except Exception:
            pass
    return "[INVALID]", "invalid"


def judge_response(response_text: Any, ground_truth: Any) -> Dict[str, Any]:
    extracted_answer, extract_method = extract_final_answer(response_text)
    normalized_prediction = normalize_final_answer(extracted_answer)
    normalized_ground_truth = normalize_ground_truth(ground_truth)
    is_correct = normalized_prediction == normalized_ground_truth and extracted_answer != "[INVALID]"
    return {
        "is_correct": is_correct,
        "extracted_answer": extracted_answer,
        "extract_method": extract_method,
        "normalized_prediction": normalized_prediction,
        "normalized_ground_truth": normalized_ground_truth,
    }


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_outputs(success_records: List[Dict[str, Any]], failed_records: List[Dict[str, Any]], success_path: Path, failed_path: Path) -> None:
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
    prompt_messages = ensure_answer_instruction(normalize_messages(row_dict.get("prompt")))
    ground_truth = None
    reward_model = row_dict.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
    elif isinstance(reward_model, str):
        try:
            reward_model_obj = json.loads(reward_model)
            if isinstance(reward_model_obj, dict):
                ground_truth = reward_model_obj.get("ground_truth")
        except Exception:
            ground_truth = None

    normalized_ground_truth = normalize_ground_truth(ground_truth)
    attempts_log: List[Dict[str, Any]] = []

    if not prompt_messages:
        return {
            "row_idx": row_idx,
            "status": "failed",
            "attempts_used": 0,
            "ground_truth": ground_truth,
            "normalized_ground_truth": normalized_ground_truth,
            "best_response": None,
            "best_extracted_answer": "[INVALID]",
            "best_extract_method": "invalid",
            "best_normalized_prediction": "[INVALID]",
            "all_attempts": attempts_log,
            "fail_reason": "empty_prompt",
        }

    for attempt_idx in range(1, max_attempts + 1):
        api_result = forward_openai_messages(
            messages=prompt_messages,
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
        response_text = api_result.get("text")
        judge = judge_response(response_text, ground_truth)
        attempts_log.append(
            {
                "attempt": attempt_idx,
                "request_status": api_result.get("status"),
                "response": response_text,
                "extracted_answer": judge["extracted_answer"],
                "extract_method": judge["extract_method"],
                "normalized_prediction": judge["normalized_prediction"],
                "is_correct": judge["is_correct"],
            }
        )
        if judge["is_correct"]:
            return {
                "row_idx": row_idx,
                "status": "success",
                "attempts_used": attempt_idx,
                "ground_truth": ground_truth,
                "normalized_ground_truth": judge["normalized_ground_truth"],
                "best_response": response_text,
                "best_extracted_answer": judge["extracted_answer"],
                "best_extract_method": judge["extract_method"],
                "best_normalized_prediction": judge["normalized_prediction"],
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
        "ground_truth": ground_truth,
        "normalized_ground_truth": normalized_ground_truth,
        "best_response": last_attempt.get("response"),
        "best_extracted_answer": last_attempt.get("extracted_answer", "[INVALID]"),
        "best_extract_method": last_attempt.get("extract_method", "invalid"),
        "best_normalized_prediction": last_attempt.get("normalized_prediction", "[INVALID]"),
        "all_attempts": attempts_log,
        "fail_reason": "incorrect_after_max_attempts",
    }


def main() -> None:
    args = parse_args()
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
            "attempts_used": int(result["attempts_used"]),
            "ground_truth": result["ground_truth"],
            "normalized_ground_truth": result["normalized_ground_truth"],
            "generated_response": result["best_response"],
            "extracted_answer": result["best_extracted_answer"],
            "extract_method": result["best_extract_method"],
            "normalized_prediction": result["best_normalized_prediction"],
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
