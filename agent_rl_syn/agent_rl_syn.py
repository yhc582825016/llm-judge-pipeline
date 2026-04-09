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
    max_tokens: int = 12000
    repetition_penalty: float = 1.05
    enable_thinking: bool = False

    min_tool_calls: int = 0                  # 若只保留多工具样本，可改为 2
    max_samples: Optional[int] = None

    qa_mode: str = "difficult"               # "extra_difficult" / "boundary_missing" / "difficult" / "easy"
    use_original_question_for_generation: bool = False

    test_timeout_sec: int = 30

    overwrite_output: bool = False           # 全量跑建议 False，避免误删
    overwrite_progress: bool = False
    save_raw_response: bool = True

    # 新增
    num_workers: int = 4                     # 多线程并发数
    resume: bool = True                      # 是否断点续跑
    success_only_output: bool = True         # output_path 只写成功样本
    require_quality_ok: bool = True          # 成功样本必须通过质量门槛
    enable_rule_verifier: bool = True        # 启用规则 verifier


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
            )
        return self._local.client

    def chat(self, messages: List[Dict[str, str]]) -> str:
        client = self._get_client()
        completion = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
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


class RuleVerifier:
    FORBIDDEN_PROMPT_TERMS = (
        "测试数据", "样例数据", "当前数据集", "本地数据", "演示环境", "run_demo",
        "mock", "mocked", "工具调用", "函数调用", "评测",
    )

    @staticmethod
    def _extract_function_names(module_code: str) -> List[str]:
        try:
            tree = ast.parse(module_code)
        except Exception:
            return []

        names = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_") and node.name != "run_demo":
                names.append(node.name)
        return names

    @staticmethod
    def _detect_required_tool_count(prompt_text: str) -> int:
        text = prompt_text or ""
        if any(token in text for token in ("然后", "最后", "分别", "同时", "先", "再", "并")):
            return 2
        return 1

    @staticmethod
    def _parse_diag(answer_text: str) -> Dict[str, str]:
        text = (answer_text or "").strip()
        match = re.fullmatch(r"//diag\{(.*)\}", text, flags=re.DOTALL)
        if not match:
            return {}

        body = match.group(1).strip()
        result: Dict[str, str] = {}
        for part in body.split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            result[key.strip()] = value.strip()
        return result

    @classmethod
    def verify(cls, qa_info: Dict[str, str], module_code: str, qa_mode: str) -> Dict[str, Any]:
        issues: List[str] = []
        prompt_text = qa_info.get("PROMPT", "").strip()
        output_format = qa_info.get("OUTPUT_FORMAT", "").strip()
        answer = qa_info.get("ANSWER", "").strip()

        if not prompt_text:
            issues.append("empty prompt")
        if not output_format:
            issues.append("empty output_format")
        if not answer:
            issues.append("empty answer")

        for term in cls.FORBIDDEN_PROMPT_TERMS:
            if term.lower() in prompt_text.lower():
                issues.append(f"prompt leaks generation context: {term}")
                break

        if len(prompt_text) > (150 if qa_mode == "extra_difficult" else 120):
            issues.append("prompt too long for selected qa_mode")

        if output_format.startswith("//box{") and not answer.startswith("//box{"):
            issues.append("answer does not match //box output format")
        if output_format.startswith("//diag{") and not answer.startswith("//diag{"):
            issues.append("answer does not match //diag output format")

        if prompt_text and answer and prompt_text == answer:
            issues.append("prompt and answer are identical")

        function_names = cls._extract_function_names(module_code)
        if not function_names:
            issues.append("no callable mocked functions extracted from module")

        if qa_mode == "extra_difficult" and len(function_names) < 2:
            issues.append("extra_difficult sample has fewer than 2 mocked functions")

        if qa_mode == "boundary_missing":
            fmt_diag = cls._parse_diag(output_format)
            ans_diag = cls._parse_diag(answer)
            if not fmt_diag:
                issues.append("boundary_missing output_format is not valid //diag")
            if not ans_diag:
                issues.append("boundary_missing answer is not valid //diag")

            if fmt_diag.get("tag") != "BOUNDARY_CASE_V1" or ans_diag.get("tag") != "BOUNDARY_CASE_V1":
                issues.append("boundary_missing tag must be BOUNDARY_CASE_V1")

            allowed_cases = {"missing_parameters", "missing_functions"}
            allowed_actions = {"clarifying_question", "graceful_decline"}
            answer_case = ans_diag.get("case")
            answer_action = ans_diag.get("expected_action")
            if answer_case not in allowed_cases:
                issues.append("boundary_missing case is invalid")
            if answer_action not in allowed_actions:
                issues.append("boundary_missing expected_action is invalid")
            if answer_case == "missing_parameters" and answer_action != "clarifying_question":
                issues.append("missing_parameters must map to clarifying_question")
            if answer_case == "missing_functions" and answer_action != "graceful_decline":
                issues.append("missing_functions must map to graceful_decline")
            if ans_diag.get("answer") != "<TBD>" and ans_diag.get("answer") != "TBD":
                issues.append("boundary_missing answer placeholder must stay TBD")
        else:
            required_tool_count = cls._detect_required_tool_count(prompt_text)
            if required_tool_count > len(function_names):
                issues.append("prompt implies more steps than available mocked functions")

        if "首字母" in prompt_text or "acronym" in prompt_text.lower():
            issues.append("meaningless letter-concat style prompt")

        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "mock_function_count": len(function_names),
            "mock_function_names": function_names,
        }


# ============================================================
# Mock 测试器
# ============================================================

class MockTestRunner:
    """
    改进版：
    1. module / test 分文件
    2. 子进程隔离执行
    3. 结构化测试报告
    4. 增加一些“质量检查”，避免只靠 exec
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

    def _quality_check(self, module_code: str, test_code: str) -> Dict[str, Any]:
        issues = []

        banned_phrases = [
            "unsupported mocked input",
            "No mock defined",
            "not implemented for this input",
            "only specific predefined combinations are supported",
        ]
        for phrase in banned_phrases:
            if phrase in module_code:
                issues.append(f"module code contains banned fallback phrase: {phrase}")

        if "assert " not in test_code and "assert(" not in test_code:
            issues.append("test code does not contain assert")

        if "def test_" not in test_code:
            issues.append("test code does not contain any test_* function")

        empty_return_patterns = [
            r"return\s+\[\s*\]",
            r"return\s+\"\"",
            r"return\s+''",
            r"return\s+\{\s*\}",
            r"return\s+None",
        ]
        empty_return_hits = 0
        for p in empty_return_patterns:
            empty_return_hits += len(re.findall(p, module_code))
        if empty_return_hits >= 4:
            issues.append(
                f"module code contains too many empty fallback returns ({empty_return_hits} hits)"
            )

        return {
            "quality_ok": len(issues) == 0,
            "quality_issues": issues,
        }

    def run(self, module_code: str, test_code: str) -> Dict[str, Any]:
        quality_report = self._quality_check(module_code, test_code)

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
                **quality_report,
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
                    **quality_report,
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
                    **quality_report,
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
            report.update(quality_report)
            return report


# ============================================================
# 主流水线
# ============================================================

class ToolSynthesisPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm = LocalLLMClient(config)
        self.test_runner = MockTestRunner(timeout_sec=config.test_timeout_sec)

        self.output_writer = ThreadSafeJsonlWriter(config.output_path)
        self.progress_writer = ThreadSafeJsonlWriter(config.progress_path)

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
            record["tool_names"] = [t.get("function", {}).get("name", "") for t in tools]
            record["used_original_question_for_generation"] = self.config.use_original_question_for_generation
            if self.config.use_original_question_for_generation:
                record["original_question"] = original_question

            # 1) 生成 mock module + test
            mock_prompt = build_mock_prompt_with_context(
                tools,
                self.config.use_original_question_for_generation,
                original_question,
            )
            messages = [{"role": "user", "content": mock_prompt}]
            mock_response = self.llm.chat(messages)
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
                qa_response = self.llm.chat(qa_messages)
                qa_info = ResponseParser.extract_qa(qa_response)
                if not self.is_meaningless_letter_concat_question(qa_info.get("PROMPT", "")):
                    break
                qa_retry_blocked = True
            if qa_retry_blocked and self.is_meaningless_letter_concat_question(qa_info.get("PROMPT", "")):
                raise ValueError("qa prompt is meaningless letter-concat style after retry")

            # 3) 执行 mock 测试
            test_report = self.test_runner.run(module_code, test_code)
            verifier_report = (
                RuleVerifier.verify(qa_info, module_code, self.config.qa_mode)
                if self.config.enable_rule_verifier
                else {"ok": True, "issues": [], "mock_function_count": 0, "mock_function_names": []}
            )
            quality_ok = test_report.get("quality_ok", True)
            verifier_ok = verifier_report.get("ok", True)
            test_ok = test_report.get("ok", False)
            final_ok = test_ok and verifier_ok
            if self.config.require_quality_ok:
                final_ok = final_ok and quality_ok

            record.update({
                "status": "ok" if final_ok else "test_failed",
                "qa_mode": self.config.qa_mode,
                "qa": qa_info,
                "test_report": test_report,
                "verifier_report": verifier_report,
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

    def prepare_input_samples(self, data: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
        """
        1. 先做样本过滤（min_tool_calls）
        2. 再做断点续跑过滤（跳过已成功导出的 sample_idx）
        3. 最后按 max_samples 截断
        """
        candidate = []

        existing_success_ids = set()
        if self.config.resume and os.path.exists(self.config.output_path):
            existing_success_ids = JsonIO.read_existing_sample_ids_from_jsonl(self.config.output_path)

        for idx, sample in enumerate(data):
            if not self.should_process(sample):
                continue
            if self.config.resume and idx in existing_success_ids:
                continue
            candidate.append((idx, sample))

        if self.config.max_samples is not None:
            candidate = candidate[:self.config.max_samples]

        return candidate

    def write_result(self, record: Dict[str, Any]):
        """
        写入策略：
        1. progress_path：全部都写
        2. output_path：只写最终成功样本
        为了断点续跑稳妥，成功样本先写 output，再写 progress
        """
        if record["status"] == "ok":
            if self.config.success_only_output:
                self.output_writer.append(record)
            else:
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
        candidate = self.prepare_input_samples(data)

        submitted = len(candidate)
        success = 0
        failed = 0

        if submitted == 0:
            return {
                "input_total": total_input,
                "submitted": 0,
                "success": 0,
                "failed": 0,
                "skipped_or_already_done": total_input,
                "output_path": self.config.output_path,
                "progress_path": self.config.progress_path,
            }

        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_idx = {
                executor.submit(self.process_one, sample, idx): idx
                for idx, sample in candidate
            }

            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="synthesizing"):
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

                if record["status"] == "ok":
                    success += 1
                else:
                    failed += 1

        skipped_or_already_done = total_input - submitted

        return {
            "input_total": total_input,
            "submitted": submitted,
            "success": success,
            "failed": failed,
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
        "--qa-mode",
        default="extra_difficult",
        choices=["extra_difficult", "boundary_missing", "difficult", "easy"],
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--min-tool-calls",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    config = PipelineConfig(
        input_path=args.input_path,

        # 只保存最终成功样本
        output_path=args.output_path,

        # 保存全部过程记录，便于排查和断点续跑观察
        progress_path=args.progress_path,

        # 如果你只想处理并行 / 多工具样本，就改成 2
        min_tool_calls=args.min_tool_calls,

        # 全量跑
        max_samples=args.max_samples,

        qa_mode=args.qa_mode,
        use_original_question_for_generation=True,

        # 全量生产建议别覆盖，避免误删历史结果
        overwrite_output=False,
        overwrite_progress=False,

        save_raw_response=True,

        # 新增能力
        num_workers=100,
        resume=True,
        success_only_output=True,

        test_timeout_sec=30,
    )

    pipeline = ToolSynthesisPipeline(config)
    summary = pipeline.run_all()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
