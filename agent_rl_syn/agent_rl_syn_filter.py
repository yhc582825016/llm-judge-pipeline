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

from openai import OpenAI
from tqdm import tqdm

try:
    from .prompt import build_filter_prompt_with_context
except ImportError:
    from prompt import build_filter_prompt_with_context


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
class FilterConfig:
    input_path: str
    output_path: str
    review_path: str

    base_url: str = "http://127.0.0.1:6031/v1"
    api_key: str = "EMPTY"
    model: str = "/opt/users/Qwen/Qwen3.5-397B"

    temperature: float = 0.0
    max_tokens: int = 2048
    request_timeout_sec: float = 180.0
    max_request_retries: int = 2
    request_retry_backoff_sec: float = 3.0

    num_workers: int = 8
    max_samples: Optional[int] = None
    resume: bool = True
    overwrite_output: bool = False
    overwrite_review: bool = False
    save_raw_response: bool = True
    require_status_ok: bool = True
    require_test_ok: bool = True


class LocalLLMClient:
    def __init__(self, config: FilterConfig):
        self.config = config
        self._local = threading.local()

    def _get_client(self) -> OpenAI:
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


class FilterResponseParser:
    JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

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
            raise ValueError("filter response must be a JSON object")

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
            result.append(str(item).strip())
        return [item for item in result if item]

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

        normalized = {
            "decision": decision,
            "confidence": cls._to_int(obj.get("confidence", 0), default=0, low=0, high=100),
            "summary": str(obj.get("summary", "")).strip(),
            "fatal_issues": cls._to_string_list(obj.get("fatal_issues")),
            "minor_issues": cls._to_string_list(obj.get("minor_issues")),
            "dimension_scores": {
                "mock_quality": cls._to_int(scores.get("mock_quality", 0), default=0, low=0, high=5),
                "qa_consistency": cls._to_int(scores.get("qa_consistency", 0), default=0, low=0, high=5),
                "answerability": cls._to_int(scores.get("answerability", 0), default=0, low=0, high=5),
                "task_realism": cls._to_int(scores.get("task_realism", 0), default=0, low=0, high=5),
                "mode_fit": cls._to_int(scores.get("mode_fit", 0), default=0, low=0, high=5),
            },
        }

        if normalized["decision"] == "accept" and normalized["fatal_issues"]:
            raise ValueError("accepted sample must not contain fatal_issues")

        return normalized


class LLMFilterPipeline:
    def __init__(self, config: FilterConfig):
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

    def should_process(self, record: Dict[str, Any]) -> bool:
        if self.config.require_status_ok and record.get("status") not in {None, "", "ok"}:
            return False

        if self.config.require_test_ok and "test_report" in record:
            test_report = record.get("test_report") or {}
            if not test_report.get("ok", False):
                return False

        return True

    def prepare_input_samples(
        self,
        data: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[str, int]]:
        candidate = []
        stats = {
            "filtered_by_status_or_test": 0,
            "filtered_by_resume_existing": 0,
            "filtered_by_max_samples": 0,
        }

        existing_review_ids = set()
        if self.config.resume and os.path.exists(self.config.review_path):
            existing_review_ids = JsonIO.read_existing_sample_ids_from_jsonl(self.config.review_path)

        for idx, record in enumerate(data):
            sample_idx = record.get("sample_idx", idx)
            if not self.should_process(record):
                stats["filtered_by_status_or_test"] += 1
                continue
            if self.config.resume and sample_idx in existing_review_ids:
                stats["filtered_by_resume_existing"] += 1
                continue
            candidate.append((idx, record))

        if self.config.max_samples is not None:
            if len(candidate) > self.config.max_samples:
                stats["filtered_by_max_samples"] = len(candidate) - self.config.max_samples
            candidate = candidate[:self.config.max_samples]

        return candidate, stats

    def process_one(self, source_record: Dict[str, Any], fallback_idx: int) -> Dict[str, Any]:
        start_time = time.time()
        sample_idx = source_record.get("sample_idx", fallback_idx)
        review_record: Dict[str, Any] = {
            "sample_idx": sample_idx,
            "status": "init",
            "error": None,
            "qa_mode": source_record.get("qa_mode"),
            "tool_names": source_record.get("tool_names"),
        }

        try:
            prompt = build_filter_prompt_with_context(source_record)
            raw_response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                request_name=f"sample {sample_idx} llm_filter",
            )
            filter_report = FilterResponseParser.parse(raw_response)
            accepted = filter_report["decision"] == "accept"

            enriched_record = dict(source_record)
            enriched_record["llm_filter"] = filter_report
            if self.config.save_raw_response:
                enriched_record["llm_filter_raw_response"] = raw_response

            review_record.update({
                "status": "ok",
                "accepted": accepted,
                "decision": filter_report["decision"],
                "confidence": filter_report["confidence"],
                "summary": filter_report["summary"],
                "fatal_issues": filter_report["fatal_issues"],
                "minor_issues": filter_report["minor_issues"],
                "dimension_scores": filter_report["dimension_scores"],
                "elapsed_sec": round(time.time() - start_time, 3),
            })

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
        input_filtered = (
            filter_stats["filtered_by_status_or_test"]
            + filter_stats["filtered_by_resume_existing"]
            + filter_stats["filtered_by_max_samples"]
        )

        if submitted == 0:
            return {
                "input_total": total_input,
                "submitted": 0,
                "accepted": 0,
                "rejected": 0,
                "errors": 0,
                "filtered_before_submit": input_filtered,
                "filtered_by_status_or_test": filter_stats["filtered_by_status_or_test"],
                "filtered_by_resume_existing": filter_stats["filtered_by_resume_existing"],
                "filtered_by_max_samples": filter_stats["filtered_by_max_samples"],
                "output_path": self.config.output_path,
                "review_path": self.config.review_path,
            }

        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_idx = {
                executor.submit(self.process_one, record, idx): idx
                for idx, record in candidate
            }

            for processed_count, future in enumerate(
                tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="llm_filtering"),
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
            "filtered_by_status_or_test": filter_stats["filtered_by_status_or_test"],
            "filtered_by_resume_existing": filter_stats["filtered_by_resume_existing"],
            "filtered_by_max_samples": filter_stats["filtered_by_max_samples"],
            "output_path": self.config.output_path,
            "review_path": self.config.review_path,
            "resume": self.config.resume,
            "num_workers": self.config.num_workers,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="/dev/shm/ye/rl-data/agent_syn_data/recall/synthetic_mock_success_only_3.jsonl",
    )
    parser.add_argument(
        "--output-path",
        default="/dev/shm/ye/rl-data/agent_syn_data/recall/synthetic_mock_success_only_3_llm_filtered.jsonl",
    )
    parser.add_argument(
        "--review-path",
        default="/dev/shm/ye/rl-data/agent_syn_data/recall/synthetic_mock_filter_review.jsonl",
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

    config = FilterConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        review_path=args.review_path,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        request_timeout_sec=args.request_timeout_sec,
        max_request_retries=args.max_request_retries,
        request_retry_backoff_sec=args.request_retry_backoff_sec,
    )

    pipeline = LLMFilterPipeline(config)
    summary = pipeline.run_all()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
