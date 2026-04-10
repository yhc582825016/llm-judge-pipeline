import argparse
import ast
import os
import re
import sys
import json
import time
import queue
import traceback
import tempfile
import threading
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm
try:
    from .prompt import build_mock_prompt_with_context, build_qa_prompt_with_context
except ImportError:
    from prompt import build_mock_prompt_with_context, build_qa_prompt_with_context


# ============================================================
# 基础 IO
# ============================================================

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
    def append_jsonl(record: Dict[str, Any], path: str):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def read_existing_sample_ids_from_jsonl(path: str, key: str = "sample_idx") -> Set[int]:
        """
        从已有 jsonl 中读取已存在的 sample_idx，用于断点续跑。
        默认读取成功导出文件 output_path，因为它是最终成功样本的“权威来源”。
        """
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


# ============================================================
# 配置
# ============================================================

@dataclass
class PipelineConfig:
    input_path: str
    output_path: str                         # 只保存最终成功样本
    progress_path: str                       # 保存全部过程记录（成功 / 失败 / 错误）

    base_url: str = "http://127.0.0.1:6031/v1"
    api_key: str = "EMPTY"
    model: str = "/opt/users/Qwen/Qwen3.5-397B"

    temperature: float = 1.0
    mock_max_tokens: int = 4096
    qa_max_tokens: int = 1024
    repetition_penalty: float = 1.05
    enable_thinking: bool = False
    request_timeout_sec: float = 180.0
    max_request_retries: int = 2
    request_retry_backoff_sec: float = 3.0

    min_tool_calls: int = 0                  # 若只保留多工具样本，可改为 2
    max_samples: Optional[int] = None

    qa_mode: str = "extra_difficult"               # "extra_difficult" / "boundary_missing" / "difficult" / "easy"
    use_original_question_for_generation: bool = False

    test_timeout_sec: int = 30

    overwrite_output: bool = False           # 全量跑建议 False，避免误删
    overwrite_progress: bool = False
    save_raw_response: bool = True

    # 新增
    num_workers: int = 16                    # 多线程并发数
    resume: bool = True                      # 是否断点续跑


# ============================================================
# LLM 客户端（线程安全：每个线程一个 client）
# ============================================================

class LocalLLMClient:
    def __init__(self, config: PipelineConfig):
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


# ============================================================
# 解析器
# ============================================================

class ResponseParser:
    MODULE_START = "===PYTHON_MODULE_START==="
    MODULE_END = "===PYTHON_MODULE_END==="
    TEST_START = "===PYTHON_TEST_START==="
    TEST_END = "===PYTHON_TEST_END==="

    MODULE_PATTERN = re.compile(
        r"===PYTHON_MODULE_START===\s*(.*?)\s*===PYTHON_MODULE_END===",
        re.DOTALL,
    )
    TEST_PATTERN = re.compile(
        r"===PYTHON_TEST_START===\s*(.*?)\s*===PYTHON_TEST_END===",
        re.DOTALL,
    )
    QA_PATTERN = re.compile(
        r"PROMPT:\s*(?P<prompt>.*?)\s*"
        r"OUTPUT_FORMAT:\s*(?P<fmt>.*?)\s*"
        r"ANSWER:\s*(?P<answer>.*?)\s*$",
        re.DOTALL,
    )

    @classmethod
    def extract_python_blocks(cls, text: str) -> Dict[str, str]:
        for marker in [cls.MODULE_START, cls.MODULE_END, cls.TEST_START, cls.TEST_END]:
            if text.count(marker) != 1:
                raise ValueError(f"marker count invalid: {marker}")

        m_mod = cls.MODULE_PATTERN.search(text)
        m_test = cls.TEST_PATTERN.search(text)
        if not m_mod or not m_test:
            raise ValueError("failed to extract python module/test blocks")

        return {
            "module_code": m_mod.group(1).strip(),
            "test_code": m_test.group(1).strip(),
        }

    @classmethod
    def extract_qa(cls, text: str) -> Dict[str, str]:
        m = cls.QA_PATTERN.search(text.strip())
        if not m:
            raise ValueError("failed to extract PROMPT / OUTPUT_FORMAT / ANSWER")
        return {
            "PROMPT": m.group("prompt").strip(),
            "OUTPUT_FORMAT": m.group("fmt").strip(),
            "ANSWER": m.group("answer").strip(),
        }


class MockContextBuilder:
    @staticmethod
    def build(module_code: str, max_functions: int = 8) -> str:
        try:
            tree = ast.parse(module_code)
        except Exception:
            return module_code[:4000]

        lines: List[str] = []
        fn_count = 0
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_") or node.name == "run_demo":
                continue

            fn_count += 1
            if fn_count > max_functions:
                lines.append(f"... 其余函数省略，共 {fn_count} 个以上函数")
                break

            args = []
            for arg in node.args.args:
                if arg.arg == "self":
                    continue
                args.append(arg.arg)

            lines.append(f"- 函数: {node.name}({', '.join(args)})")

            doc = ast.get_docstring(node)
            if doc:
                lines.append(f"  说明: {doc.strip()[:200]}")

            sample_returns: List[str] = []
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return):
                    try:
                        val = ast.literal_eval(inner.value)
                    except Exception:
                        continue
                    if isinstance(val, (dict, list, str, int, float, bool)) and val not in sample_returns:
                        sample_returns.append(json.dumps(val, ensure_ascii=False)[:300])
                    if len(sample_returns) >= 2:
                        break

            for idx, sample in enumerate(sample_returns, 1):
                lines.append(f"  样例返回{idx}: {sample}")

        return "\n".join(lines).strip() or module_code[:4000]


class MockTestRunner:
    """
    改进版：
    1. module / test 分文件
    2. 子进程隔离执行
    3. 结构化测试报告
    """

    HARNESS_CODE = r'''
import os
import sys
import json
import traceback
import importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

REPORT = {
    "ok": False,
    "syntax_ok": True,
    "test_functions": [],
    "passed": [],
    "failed": [],
}

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    mock_mod = load_module("generated_mock_module", os.path.join(ROOT, "generated_mock_module.py"))
    test_mod = load_module("generated_mock_test", os.path.join(ROOT, "generated_mock_test.py"))
except Exception:
    REPORT["syntax_ok"] = False
    REPORT["failed"].append({
        "name": "__import__",
        "error": traceback.format_exc(),
    })
    print(json.dumps(REPORT, ensure_ascii=False))
    raise SystemExit(0)

test_names = sorted(
    name for name in dir(test_mod)
    if name.startswith("test_") and callable(getattr(test_mod, name))
)
REPORT["test_functions"] = test_names

for name in test_names:
    fn = getattr(test_mod, name)
    try:
        fn()
        REPORT["passed"].append(name)
    except Exception:
        REPORT["failed"].append({
            "name": name,
            "error": traceback.format_exc(),
        })

REPORT["ok"] = REPORT["syntax_ok"] and len(REPORT["failed"]) == 0
print(json.dumps(REPORT, ensure_ascii=False))
'''.strip()

    def __init__(self, timeout_sec: int = 30):
        self.timeout_sec = timeout_sec

    def _syntax_check(self, code: str, filename: str) -> Optional[str]:
        try:
            compile(code, filename, "exec")
            return None
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    def _ensure_test_import(self, test_code: str) -> str:
        import_line = "from generated_mock_module import *"
        if import_line in test_code:
            return test_code
        return import_line + "\n\n" + test_code

    def run(self, module_code: str, test_code: str) -> Dict[str, Any]:
        module_syntax_error = self._syntax_check(module_code, "generated_mock_module.py")
        test_syntax_error = self._syntax_check(test_code, "generated_mock_test.py")

        if module_syntax_error or test_syntax_error:
            return {
                "ok": False,
                "syntax_ok": False,
                "module_syntax_error": module_syntax_error,
                "test_syntax_error": test_syntax_error,
                "test_functions": [],
                "passed": [],
                "failed": [],
            }

        wrapped_test_code = self._ensure_test_import(test_code)

        with tempfile.TemporaryDirectory(prefix="mock_runner_") as tmpdir:
            root = Path(tmpdir)
            module_path = root / "generated_mock_module.py"
            test_path = root / "generated_mock_test.py"
            harness_path = root / "run_harness.py"

            module_path.write_text(module_code, encoding="utf-8")
            test_path.write_text(wrapped_test_code, encoding="utf-8")
            harness_path.write_text(self.HARNESS_CODE, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [sys.executable, str(harness_path)],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
            except subprocess.TimeoutExpired:
                return {
                    "ok": False,
                    "syntax_ok": True,
                    "timeout": True,
                    "test_functions": [],
                    "passed": [],
                    "failed": [{"name": "__timeout__", "error": f"timeout > {self.timeout_sec}s"}],
                }

            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()

            if not stdout:
                return {
                    "ok": False,
                    "syntax_ok": True,
                    "timeout": False,
                    "test_functions": [],
                    "passed": [],
                    "failed": [{"name": "__empty_stdout__", "error": stderr or "empty stdout"}],
                    "stderr": stderr,
                }

            last_line = stdout.splitlines()[-1]
            try:
                report = json.loads(last_line)
            except Exception:
                report = {
                    "ok": False,
                    "syntax_ok": True,
                    "test_functions": [],
                    "passed": [],
                    "failed": [{"name": "__parse_report__", "error": stdout}],
                }

            report["stdout"] = stdout[-4000:]
            report["stderr"] = stderr[-4000:]
            report["timeout"] = False
            return report


# ============================================================
# 主流水线
# ============================================================

class ToolSynthesisPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._ensure_output_file_parent(config.output_path, "output_path")
        self._ensure_output_file_parent(config.progress_path, "progress_path")
        self.llm = LocalLLMClient(config)
        self.test_runner = MockTestRunner(timeout_sec=config.test_timeout_sec)

        self.output_writer = ThreadSafeJsonlWriter(config.output_path)
        self.progress_writer = ThreadSafeJsonlWriter(config.progress_path)

    @staticmethod
    def _ensure_output_file_parent(path: str, label: str) -> None:
        if os.path.isdir(path):
            raise IsADirectoryError(f"{label} must be a file path, got directory: {path}")

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    @staticmethod
    def ensure_tools_obj(tools_field: Union[str, List[Dict[str, Any]], None]) -> List[Dict[str, Any]]:
        if tools_field is None:
            return []
        if isinstance(tools_field, str):
            tools_field = tools_field.strip()
            return json.loads(tools_field) if tools_field else []
        if isinstance(tools_field, list):
            return tools_field
        raise TypeError(f"unsupported tools type: {type(tools_field)}")

    @staticmethod
    def count_tool_calls(sample: Dict[str, Any]) -> int:
        msgs = sample.get("messages", []) if isinstance(sample, dict) else []
        total = 0

        for m in msgs:
            if not isinstance(m, dict):
                continue

            if m.get("role") == "tool_call":
                content = m.get("content")
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

            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list):
                total += len(tool_calls)

        return total

    @staticmethod
    def extract_original_user_question(sample: Dict[str, Any]) -> str:
        msgs = sample.get("messages", []) if isinstance(sample, dict) else []
        if not isinstance(msgs, list):
            return ""

        for m in msgs:
            if not isinstance(m, dict):
                continue
            if m.get("role") != "user":
                continue

            content = m.get("content")
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

    def should_process(self, sample: Dict[str, Any]) -> bool:
        return self.count_tool_calls(sample) >= self.config.min_tool_calls

    @staticmethod
    def is_meaningless_letter_concat_question(prompt_text: str) -> bool:
        """
        过滤掉“查若干文本后取首字母拼接”这类无意义 QA。
        """
        text = (prompt_text or "").lower()
        patterns = [
            r"首字母",
            r"第一个单词的首字母",
            r"拼在一起",
            r"拼接(成)?字符串",
            r"initial letter",
            r"first letter",
            r"first word",
            r"acronym",
        ]
        if not any(re.search(p, text) for p in patterns):
            return False

        context_hints = [
            r"标题",
            r"新闻",
            r"title",
            r"headline",
        ]
        return any(re.search(p, text) for p in context_hints)

    def process_one(self, sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
        start_time = time.time()

        record: Dict[str, Any] = {
            "sample_idx": sample_idx,
            "status": "init",
            "error": None,
            "tool_call_count": self.count_tool_calls(sample),
        }

        try:
            tools = self.ensure_tools_obj(sample.get("tools"))
            original_question = self.extract_original_user_question(sample)
            record["tools"] = tools
            record["tool_names"] = [t.get("function", {}).get("name", "") for t in tools]
            record["used_original_question_for_generation"] = self.config.use_original_question_for_generation
            if original_question:
                record["original_question"] = original_question

            # 1) 生成 mock module + test
            mock_prompt = build_mock_prompt_with_context(
                tools,
                self.config.use_original_question_for_generation,
                original_question,
            )
            messages = [{"role": "user", "content": mock_prompt}]
            mock_response = self.llm.chat(
                messages,
                max_tokens=self.config.mock_max_tokens,
                request_name=f"sample {sample_idx} mock_generation",
            )
            messages.append({"role": "assistant", "content": mock_response})

            parsed_code = ResponseParser.extract_python_blocks(mock_response)
            module_code = parsed_code["module_code"]
            test_code = parsed_code["test_code"]

            # 2) 解耦生成 QA：只给出原始问题和 mocked 环境摘要，不续写 mock 代码生成对话
            mock_context = MockContextBuilder.build(module_code)
            qa_prompt = build_qa_prompt_with_context(
                self.config.qa_mode,
                self.config.use_original_question_for_generation,
                original_question,
                mock_context=mock_context,
            )
            qa_response = ""
            qa_info: Dict[str, str] = {}
            qa_retry_blocked = False
            for qa_try in range(2):
                extra_guard = ""
                if qa_try > 0:
                    extra_guard = (
                        "\n\n额外约束：严禁生成“先查多条文本/标题再取首字母拼接字符串”的题型。"
                        "这类题目没有实际业务意义，必须重写为真实查询/筛选/比对任务。"
                    )
                qa_messages = [{"role": "user", "content": qa_prompt + extra_guard}]
                qa_response = self.llm.chat(
                    qa_messages,
                    max_tokens=self.config.qa_max_tokens,
                    request_name=f"sample {sample_idx} qa_generation_try_{qa_try + 1}",
                )
                qa_info = ResponseParser.extract_qa(qa_response)
                if not self.is_meaningless_letter_concat_question(qa_info.get("PROMPT", "")):
                    break
                qa_retry_blocked = True
            if qa_retry_blocked and self.is_meaningless_letter_concat_question(qa_info.get("PROMPT", "")):
                raise ValueError("qa prompt is meaningless letter-concat style after retry")

            # 3) 执行 mock 测试
            test_report = self.test_runner.run(module_code, test_code)
            test_ok = test_report.get("ok", False)
            final_ok = test_ok

            record.update({
                "status": "ok" if final_ok else "test_failed",
                "qa_mode": self.config.qa_mode,
                "qa": qa_info,
                "test_report": test_report,
                "mock_context": mock_context,
                "elapsed_sec": round(time.time() - start_time, 3),
            })

            if self.config.save_raw_response:
                record["mock_response_raw"] = mock_response
                record["qa_response_raw"] = qa_response
                record["module_code"] = module_code
                record["test_code"] = test_code

            return record

        except Exception as e:
            record["status"] = "error"
            record["error"] = f"{type(e).__name__}: {e}"
            record["traceback"] = traceback.format_exc(limit=5)
            record["elapsed_sec"] = round(time.time() - start_time, 3)
            return record

    def prepare_input_samples(self, data: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[str, int]]:
        """
        1. 先做样本过滤（min_tool_calls）
        2. 再做断点续跑过滤（跳过已成功导出的 sample_idx）
        3. 最后按 max_samples 截断
        """
        candidate = []
        stats = {
            "filtered_by_min_tool_calls": 0,
            "filtered_by_resume_existing": 0,
            "filtered_by_max_samples": 0,
        }

        existing_success_ids = set()
        if self.config.resume and os.path.exists(self.config.output_path):
            existing_success_ids = JsonIO.read_existing_sample_ids_from_jsonl(self.config.output_path)

        for idx, sample in enumerate(data):
            if not self.should_process(sample):
                stats["filtered_by_min_tool_calls"] += 1
                continue
            if self.config.resume and idx in existing_success_ids:
                stats["filtered_by_resume_existing"] += 1
                continue
            candidate.append((idx, sample))

        if self.config.max_samples is not None:
            if len(candidate) > self.config.max_samples:
                stats["filtered_by_max_samples"] = len(candidate) - self.config.max_samples
            candidate = candidate[:self.config.max_samples]

        return candidate, stats

    def write_result(self, record: Dict[str, Any]):
        """
        写入策略：
        1. progress_path：全部都写
        2. output_path：只写最终成功样本
        为了断点续跑稳妥，成功样本先写 output，再写 progress
        """
        if record["status"] == "ok":
            self.output_writer.append(record)

        self.progress_writer.append({
            "sample_idx": record.get("sample_idx"),
            "status": record.get("status"),
            "error": record.get("error"),
            "elapsed_sec": record.get("elapsed_sec"),
            "tool_call_count": record.get("tool_call_count"),
            "tool_names": record.get("tool_names"),
        })

    def run_all(self) -> Dict[str, Any]:
        # 文件初始化
        if self.config.overwrite_output and os.path.exists(self.config.output_path):
            os.remove(self.config.output_path)
        if self.config.overwrite_progress and os.path.exists(self.config.progress_path):
            os.remove(self.config.progress_path)

        data = JsonIO.load_json_or_jsonl(self.config.input_path)
        if not isinstance(data, list):
            raise ValueError("input data must be a list-like json/jsonl content")

        total_input = len(data)
        candidate, filter_stats = self.prepare_input_samples(data)

        submitted = len(candidate)
        success = 0
        failed = 0
        test_ok_count = 0
        error_count = 0
        input_filtered = (
            filter_stats["filtered_by_min_tool_calls"]
            + filter_stats["filtered_by_resume_existing"]
            + filter_stats["filtered_by_max_samples"]
        )

        if submitted == 0:
            return {
                "input_total": total_input,
                "submitted": 0,
                "success": 0,
                "failed": 0,
                "filtered_before_submit": input_filtered,
                "filtered_by_min_tool_calls": filter_stats["filtered_by_min_tool_calls"],
                "filtered_by_resume_existing": filter_stats["filtered_by_resume_existing"],
                "filtered_by_max_samples": filter_stats["filtered_by_max_samples"],
                "passed_after_submit": 0,
                "filtered_after_submit": 0,
                "skipped_or_already_done": total_input,
                "output_path": self.config.output_path,
                "progress_path": self.config.progress_path,
            }

        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_idx = {
                executor.submit(self.process_one, sample, idx): idx
                for idx, sample in candidate
            }

            for processed_count, future in enumerate(
                tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="synthesizing"),
                start=1,
            ):
                idx = future_to_idx[future]
                try:
                    record = future.result()
                except Exception as e:
                    record = {
                        "sample_idx": idx,
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                    }

                self.write_result(record)

                test_ok = bool(record.get("test_report", {}).get("ok", False))

                if record.get("status") == "error":
                    error_count += 1
                if test_ok:
                    test_ok_count += 1

                if record["status"] == "ok":
                    success += 1
                else:
                    failed += 1

                if processed_count % 10 == 0 or processed_count == submitted:
                    print(
                        "processed/submitted: "
                        f"{processed_count}/{submitted} | "
                        f"test_ok: {test_ok_count} | "
                        f"final_ok: {success} | "
                        f"errors: {error_count}",
                        flush=True,
                    )

        skipped_or_already_done = total_input - submitted
        passed_after_submit = success
        filtered_after_submit = failed

        return {
            "input_total": total_input,
            "submitted": submitted,
            "success": success,
            "failed": failed,
            "filtered_before_submit": input_filtered,
            "filtered_by_min_tool_calls": filter_stats["filtered_by_min_tool_calls"],
            "filtered_by_resume_existing": filter_stats["filtered_by_resume_existing"],
            "filtered_by_max_samples": filter_stats["filtered_by_max_samples"],
            "passed_after_submit": passed_after_submit,
            "filtered_after_submit": filtered_after_submit,
            "skipped_or_already_done": skipped_or_already_done,
            "output_path": self.config.output_path,
            "progress_path": self.config.progress_path,
            "resume": self.config.resume,
            "num_workers": self.config.num_workers,
        }


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="/dev/shm/ye/rl-data/agent_syn_data/tool_use_no_think_v2_first_2w.jsonl",
    )
    parser.add_argument(
        "--output-path",
        default="/dev/shm/ye/rl-data/agent_syn_data/recall/synthetic_mock_success_only_3.jsonl",
    )
    parser.add_argument(
        "--progress-path",
        default="/dev/shm/ye/rl-data/agent_syn_data/recall/synthetic_mock_progress.jsonl",
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
        "--qa-mode",
        default="extra_difficult",
        choices=["extra_difficult", "boundary_missing", "difficult", "easy"],
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=300000,
    )
    parser.add_argument(
        "--min-tool-calls",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--mock-max-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--qa-max-tokens",
        type=int,
        default=1024,
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

    config = PipelineConfig(
        input_path=args.input_path,

        # 只保存最终成功样本
        output_path=args.output_path,

        # 保存全部过程记录，便于排查和断点续跑观察
        progress_path=args.progress_path,

        base_url=args.base_url,
        model=args.model,

        # 如果你只想处理并行 / 多工具样本，就改成 2
        min_tool_calls=args.min_tool_calls,

        # 全量跑
        max_samples=args.max_samples,

        qa_mode=args.qa_mode,
        use_original_question_for_generation=True,

        mock_max_tokens=args.mock_max_tokens,
        qa_max_tokens=args.qa_max_tokens,
        request_timeout_sec=args.request_timeout_sec,
        max_request_retries=args.max_request_retries,
        request_retry_backoff_sec=args.request_retry_backoff_sec,

        # 全量生产建议别覆盖，避免误删历史结果
        overwrite_output=False,
        overwrite_progress=False,

        save_raw_response=True,

        # 新增能力
        num_workers=args.num_workers,
        resume=True,
        test_timeout_sec=300,
    )

    pipeline = ToolSynthesisPipeline(config)
    summary = pipeline.run_all()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
