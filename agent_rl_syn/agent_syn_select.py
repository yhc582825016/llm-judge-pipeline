import argparse
import json
import os
import re
import sys
import time
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from .prompt import build_agent_syn_select_prompt_with_context
except ImportError:
    from prompt import build_agent_syn_select_prompt_with_context


class JsonIO:
    @staticmethod
    def load_json(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_jsonl(path: str):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    @staticmethod
    def load_json_or_jsonl(path: str):
        try:
            return JsonIO.load_json(path)
        except Exception:
            return JsonIO.load_jsonl(path)

    @staticmethod
    def read_existing_sample_ids_from_jsonl(path: str, key: str = "sample_idx") -> Set[int]:
        existed = set()
        if not os.path.exists(path):
            return existed

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if key in obj:
                        existed.add(int(obj[key]))
                except Exception:
                    continue
        return existed


class ThreadSafeJsonlWriter:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: Dict[str, Any]):
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


@dataclass
class SelectionConfig:
    input_path: str
    output_path: str
    review_path: str

    base_url: str = "http://127.0.0.1:6031/v1"
    api_key: str = "EMPTY"
    model: str = "/opt/users/Qwen/Qwen3.5-397B"

    temperature: float = 0.0
    max_tokens: int = 2048
    repetition_penalty: float = 1.05
    enable_thinking: bool = False
    request_timeout_sec: float = 180.0
    max_request_retries: int = 2
    request_retry_backoff_sec: float = 3.0

    num_workers: int = 8
    max_samples: Optional[int] = None
    resume: bool = True
    overwrite_output: bool = False
    overwrite_review: bool = False
    save_raw_response: bool = True

    min_tool_definitions: int = 2
    min_tool_calls: int = 1
    require_user_question: bool = True


class LocalLLMClient:
    def __init__(self, config: SelectionConfig):
        self.config = config
        self._local = threading.local()

    def _get_client(self) -> OpenAI:
        if OpenAI is None:
            raise ImportError(
                "openai package is required to run agent_syn_select.py. "
                "Please install it in the current Python environment."
            )
        if not hasattr(self._local, "client"):
            self._local.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.request_timeout_sec,
                max_retries=0,
            )
        return self._local.client

    def _reset_client(self) -> None:
        if hasattr(self._local, "client"):
            delattr(self._local, "client")

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        request_name: str,
    ) -> str:
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.config.max_request_retries + 2):
            try:
                client = self._get_client()
                completion = client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=max_tokens,
                    timeout=self.config.request_timeout_sec,
                    extra_body={
                        "repetition_penalty": self.config.repetition_penalty,
                        "chat_template_kwargs": {
                            "enable_thinking": self.config.enable_thinking
                        },
                    },
                )
                result = completion.choices[0].message.content or ""
                if "</think>" in result:
                    result = result.split("</think>")[-1]
                return result.strip()
            except Exception as exc:
                last_exc = exc
                self._reset_client()
                if attempt > self.config.max_request_retries:
                    break

                sleep_sec = self.config.request_retry_backoff_sec * attempt
                print(
                    f"[warn] {request_name} attempt {attempt}/{self.config.max_request_retries + 1} "
                    f"failed with {type(exc).__name__}: {exc}. Retrying in {sleep_sec:.1f}s.",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(sleep_sec)

        raise RuntimeError(
            f"{request_name} failed after {self.config.max_request_retries + 1} attempts: "
            f"{type(last_exc).__name__}: {last_exc}"
        )


class SelectionResponseParser:
    JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
    ALLOWED_ANSWER_TYPES = {
        "single_value",
        "short_object",
        "short_list",
        "binary_decision",
        "not_recommended",
    }

    @classmethod
    def parse(cls, text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = cls._strip_code_fence(raw)

        try:
            obj = json.loads(raw)
        except Exception:
            match = cls.JSON_PATTERN.search(raw)
            if not match:
                raise ValueError("failed to extract JSON object from model response")
            obj = json.loads(match.group(0))

        if not isinstance(obj, dict):
            raise ValueError("selection response must be a JSON object")

        return cls._normalize(obj)

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _to_string_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            if item is None:
                continue
            item = str(item).strip()
            if item:
                result.append(item)
        return result

    @staticmethod
    def _to_int(value: Any, default: int, low: int, high: int) -> int:
        try:
            value = int(value)
        except Exception:
            return default
        return max(low, min(high, value))

    @classmethod
    def _normalize(cls, obj: Dict[str, Any]) -> Dict[str, Any]:
        decision = str(obj.get("decision", "")).strip().lower()
        if decision not in {"accept", "reject"}:
            raise ValueError(f"invalid decision: {decision}")

        scores = obj.get("dimension_scores")
        if not isinstance(scores, dict):
            scores = {}

        answer_type = str(obj.get("suggested_answer_type", "not_recommended")).strip()
        if answer_type not in cls.ALLOWED_ANSWER_TYPES:
            answer_type = "not_recommended"

        normalized = {
            "decision": decision,
            "confidence": cls._to_int(obj.get("confidence", 0), default=0, low=0, high=100),
            "summary": str(obj.get("summary", "")).strip(),
            "fatal_issues": cls._to_string_list(obj.get("fatal_issues")),
            "minor_issues": cls._to_string_list(obj.get("minor_issues")),
            "dimension_scores": {
                "tool_synergy": cls._to_int(scores.get("tool_synergy", 0), default=0, low=0, high=5),
                "multi_step_depth": cls._to_int(scores.get("multi_step_depth", 0), default=0, low=0, high=5),
                "ground_truth_feasibility": cls._to_int(scores.get("ground_truth_feasibility", 0), default=0, low=0, high=5),
                "synthesis_potential": cls._to_int(scores.get("synthesis_potential", 0), default=0, low=0, high=5),
                "realism_focus": cls._to_int(scores.get("realism_focus", 0), default=0, low=0, high=5),
            },
            "suggested_task_pattern": str(obj.get("suggested_task_pattern", "")).strip() or "not_recommended",
            "suggested_answer_type": answer_type,
        }

        if normalized["decision"] == "accept" and normalized["fatal_issues"]:
            raise ValueError("accepted sample must not contain fatal_issues")

        return normalized


class AgentSynSelectPipeline:
    def __init__(self, config: SelectionConfig):
        self.config = config
        self._ensure_output_file_parent(config.output_path, "output_path")
        self._ensure_output_file_parent(config.review_path, "review_path")
        self.llm = LocalLLMClient(config)
        self.output_writer = ThreadSafeJsonlWriter(config.output_path)
        self.review_writer = ThreadSafeJsonlWriter(config.review_path)

    @staticmethod
    def _ensure_output_file_parent(path: str, label: str) -> None:
        if os.path.isdir(path):
            raise IsADirectoryError(f"{label} must be a file path, got directory: {path}")

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    @staticmethod
    def ensure_tools_obj(tools_field: Any) -> List[Dict[str, Any]]:
        if tools_field is None:
            return []
        if isinstance(tools_field, str):
            tools_field = tools_field.strip()
            return json.loads(tools_field) if tools_field else []
        if isinstance(tools_field, list):
            return tools_field
        raise TypeError(f"unsupported tools type: {type(tools_field)}")

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    @staticmethod
    def extract_tool_names_from_defs(tools: List[Dict[str, Any]]) -> List[str]:
        names: List[str] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function", {})
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name", "")).strip()
            if name:
                names.append(name)
        return AgentSynSelectPipeline._dedupe_preserve_order(names)

    @staticmethod
    def count_tool_calls(sample: Dict[str, Any]) -> int:
        msgs = sample.get("messages", []) if isinstance(sample, dict) else []
        total = 0

        for message in msgs:
            if not isinstance(message, dict):
                continue

            if message.get("role") == "tool_call":
                content = message.get("content")
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        total += len(parsed) if isinstance(parsed, list) else 1
                    except Exception:
                        total += 1
                elif isinstance(content, list):
                    total += len(content)
                elif content is not None:
                    total += 1
                continue

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                total += len(tool_calls)

        return total

    @staticmethod
    def extract_observed_tool_names(sample: Dict[str, Any]) -> List[str]:
        msgs = sample.get("messages", []) if isinstance(sample, dict) else []
        names: List[str] = []

        def add_name(value: Any) -> None:
            name = str(value or "").strip()
            if name:
                names.append(name)

        def parse_tool_call_obj(obj: Any) -> None:
            if isinstance(obj, list):
                for item in obj:
                    parse_tool_call_obj(item)
                return

            if not isinstance(obj, dict):
                return

            if "name" in obj:
                add_name(obj.get("name"))

            fn = obj.get("function")
            if isinstance(fn, dict):
                add_name(fn.get("name"))

        for message in msgs:
            if not isinstance(message, dict):
                continue

            if message.get("role") == "tool_call":
                content = message.get("content")
                if isinstance(content, str):
                    try:
                        parse_tool_call_obj(json.loads(content))
                    except Exception:
                        pass
                else:
                    parse_tool_call_obj(content)

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                parse_tool_call_obj(tool_calls)

        return AgentSynSelectPipeline._dedupe_preserve_order(names)

    @staticmethod
    def extract_original_user_question(sample: Dict[str, Any]) -> str:
        msgs = sample.get("messages", []) if isinstance(sample, dict) else []
        if not isinstance(msgs, list):
            return ""

        for message in msgs:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue

            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            text_parts.append(text.strip())
                if text_parts:
                    return "\n".join(text_parts)

        return ""

    def inspect_sample(self, sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
        tool_parse_error = None
        try:
            tools = self.ensure_tools_obj(sample.get("tools"))
        except Exception as exc:
            tools = []
            tool_parse_error = f"{type(exc).__name__}: {exc}"

        tool_names = self.extract_tool_names_from_defs(tools)
        observed_tool_names = self.extract_observed_tool_names(sample)
        original_question = self.extract_original_user_question(sample)
        tool_call_count = self.count_tool_calls(sample)

        return {
            "sample_idx": sample_idx,
            "sample": sample,
            "tools": tools,
            "tool_names": tool_names,
            "tool_definition_count": len(tools),
            "observed_tool_names": observed_tool_names,
            "observed_tool_count": len(observed_tool_names),
            "tool_call_count": tool_call_count,
            "original_question": original_question,
            "messages": sample.get("messages", []) if isinstance(sample, dict) else [],
            "tool_parse_error": tool_parse_error,
        }

    def should_process(self, info: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if info.get("tool_parse_error"):
            return False, "filtered_by_invalid_tools"

        if info["tool_definition_count"] < self.config.min_tool_definitions:
            return False, "filtered_by_min_tool_definitions"

        if info["tool_call_count"] < self.config.min_tool_calls:
            return False, "filtered_by_min_tool_calls"

        if self.config.require_user_question and not info["original_question"]:
            return False, "filtered_by_missing_user_question"

        return True, None

    def prepare_input_samples(
        self,
        data: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        candidate: List[Dict[str, Any]] = []
        stats = {
            "filtered_by_invalid_tools": 0,
            "filtered_by_min_tool_definitions": 0,
            "filtered_by_min_tool_calls": 0,
            "filtered_by_missing_user_question": 0,
            "filtered_by_resume_existing": 0,
            "filtered_by_max_samples": 0,
        }

        existing_review_ids = set()
        if self.config.resume and os.path.exists(self.config.review_path):
            existing_review_ids = JsonIO.read_existing_sample_ids_from_jsonl(self.config.review_path)

        for idx, sample in enumerate(data):
            info = self.inspect_sample(sample, idx)
            should_use, reason = self.should_process(info)
            if not should_use:
                stats[reason] += 1
                continue
            if self.config.resume and info["sample_idx"] in existing_review_ids:
                stats["filtered_by_resume_existing"] += 1
                continue
            candidate.append(info)

        if self.config.max_samples is not None:
            if len(candidate) > self.config.max_samples:
                stats["filtered_by_max_samples"] = len(candidate) - self.config.max_samples
            candidate = candidate[:self.config.max_samples]

        return candidate, stats

    def process_one(self, info: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        sample_idx = info["sample_idx"]
        source_sample = info["sample"]
        review_record: Dict[str, Any] = {
            "sample_idx": sample_idx,
            "status": "init",
            "error": None,
            "tool_names": info["tool_names"],
            "observed_tool_names": info["observed_tool_names"],
            "tool_definition_count": info["tool_definition_count"],
            "tool_call_count": info["tool_call_count"],
        }

        try:
            prompt = build_agent_syn_select_prompt_with_context(info)
            raw_response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                request_name=f"sample {sample_idx} agent_syn_select",
            )
            selection_report = SelectionResponseParser.parse(raw_response)
            accepted = selection_report["decision"] == "accept"

            enriched_record = dict(source_sample)
            enriched_record["sample_idx"] = sample_idx
            enriched_record["tool_names"] = info["tool_names"]
            enriched_record["observed_tool_names"] = info["observed_tool_names"]
            enriched_record["tool_definition_count"] = info["tool_definition_count"]
            enriched_record["tool_call_count"] = info["tool_call_count"]
            enriched_record["original_question"] = info["original_question"]
            enriched_record["agent_syn_select"] = selection_report

            if self.config.save_raw_response:
                enriched_record["agent_syn_select_raw_response"] = raw_response

            review_record.update({
                "status": "ok",
                "accepted": accepted,
                "decision": selection_report["decision"],
                "confidence": selection_report["confidence"],
                "summary": selection_report["summary"],
                "fatal_issues": selection_report["fatal_issues"],
                "minor_issues": selection_report["minor_issues"],
                "dimension_scores": selection_report["dimension_scores"],
                "suggested_task_pattern": selection_report["suggested_task_pattern"],
                "suggested_answer_type": selection_report["suggested_answer_type"],
                "elapsed_sec": round(time.time() - start_time, 3),
            })

            if self.config.save_raw_response:
                review_record["raw_response"] = raw_response

            return {
                "accepted_record": enriched_record if accepted else None,
                "review_record": review_record,
            }

        except Exception as exc:
            review_record["status"] = "error"
            review_record["error"] = f"{type(exc).__name__}: {exc}"
            review_record["traceback"] = traceback.format_exc(limit=5)
            review_record["elapsed_sec"] = round(time.time() - start_time, 3)
            return {
                "accepted_record": None,
                "review_record": review_record,
            }

    def write_result(self, result: Dict[str, Any]) -> None:
        accepted_record = result.get("accepted_record")
        review_record = result.get("review_record") or {}

        if accepted_record is not None:
            self.output_writer.append(accepted_record)

        self.review_writer.append(review_record)

    def run_all(self) -> Dict[str, Any]:
        if self.config.overwrite_output and os.path.exists(self.config.output_path):
            os.remove(self.config.output_path)
        if self.config.overwrite_review and os.path.exists(self.config.review_path):
            os.remove(self.config.review_path)

        data = JsonIO.load_json_or_jsonl(self.config.input_path)
        if not isinstance(data, list):
            raise ValueError("input data must be a list-like json/jsonl content")

        total_input = len(data)
        candidate, filter_stats = self.prepare_input_samples(data)

        submitted = len(candidate)
        accepted = 0
        rejected = 0
        error_count = 0
        input_filtered = sum(filter_stats.values())

        if submitted == 0:
            return {
                "input_total": total_input,
                "submitted": 0,
                "accepted": 0,
                "rejected": 0,
                "errors": 0,
                "filtered_before_submit": input_filtered,
                **filter_stats,
                "output_path": self.config.output_path,
                "review_path": self.config.review_path,
            }

        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_idx = {
                executor.submit(self.process_one, info): info["sample_idx"]
                for info in candidate
            }

            for processed_count, future in enumerate(
                tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="agent_syn_select"),
                start=1,
            ):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "accepted_record": None,
                        "review_record": {
                            "sample_idx": idx,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    }

                self.write_result(result)

                review_record = result.get("review_record") or {}
                if review_record.get("status") == "error":
                    error_count += 1
                elif review_record.get("accepted"):
                    accepted += 1
                else:
                    rejected += 1

                if processed_count % 10 == 0 or processed_count == submitted:
                    print(
                        "processed/submitted: "
                        f"{processed_count}/{submitted} | "
                        f"accepted: {accepted} | "
                        f"rejected: {rejected} | "
                        f"errors: {error_count}",
                        flush=True,
                    )

        return {
            "input_total": total_input,
            "submitted": submitted,
            "accepted": accepted,
            "rejected": rejected,
            "errors": error_count,
            "filtered_before_submit": input_filtered,
            **filter_stats,
            "output_path": self.config.output_path,
            "review_path": self.config.review_path,
            "resume": self.config.resume,
            "num_workers": self.config.num_workers,
            "min_tool_definitions": self.config.min_tool_definitions,
            "min_tool_calls": self.config.min_tool_calls,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="/dev/shm/ye/data/tool_use_no_think_v2.jsonl",
    )
    parser.add_argument(
        "--output-path",
        default="/dev/shm/ye/data/tool_use_no_think_v2.agent_syn_selected.jsonl",
    )
    parser.add_argument(
        "--review-path",
        default="/dev/shm/ye/data/tool_use_no_think_v2.agent_syn_select_review.jsonl",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:6031/v1",
    )
    parser.add_argument(
        "--model",
        default="/opt/users/Qwen/Qwen3.5-397B",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--min-tool-definitions",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--min-tool-calls",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--max-request-retries",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--request-retry-backoff-sec",
        type=float,
        default=3.0,
    )
    args = parser.parse_args()

    config = SelectionConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        review_path=args.review_path,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        min_tool_definitions=args.min_tool_definitions,
        min_tool_calls=args.min_tool_calls,
        request_timeout_sec=args.request_timeout_sec,
        max_request_retries=args.max_request_retries,
        request_retry_backoff_sec=args.request_retry_backoff_sec,
    )

    pipeline = AgentSynSelectPipeline(config)
    summary = pipeline.run_all()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
