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

    base_url: str = "http://127.0.0.1:6032/v1"
    api_key: str = "EMPTY"
    model: str = "/opt/users/Qwen/Qwen3.5-397B"

    temperature: float = 1.0
    max_tokens: int = 12000
    repetition_penalty: float = 1.05
    enable_thinking: bool = False

    min_tool_calls: int = 0                  # 若只保留多工具样本，可改为 2
    max_samples: Optional[int] = None

    qa_mode: str = "difficult"               # "difficult" / "easy"
    use_original_question_for_generation: bool = False

    test_timeout_sec: int = 30

    overwrite_output: bool = False           # 全量跑建议 False，避免误删
    overwrite_progress: bool = False
    save_raw_response: bool = True

    # 新增
    num_workers: int = 4                     # 多线程并发数
    resume: bool = True                      # 是否断点续跑
    success_only_output: bool = True         # output_path 只写成功样本


# ============================================================
# Prompt 模板
# ============================================================

class PromptRepository:
    MOCK_PROMPT = '''
你现在是一个“工具函数模拟器生成器”。

我会提供一组工具定义。你的任务不是实现真实工具能力，而是为每一个工具生成一个“可执行的 mock 函数”，用于本地测试、单元测试或流程联调。

【核心目标】
请为每一个工具合成一个可执行函数。由于真实输入空间无限大，你不需要覆盖所有输入，只需要：
1. 为每个工具挑选若干组“你能确定正确”的输入；
2. 对这些固定输入返回固定输出；
3. 对所有未覆盖输入，明确抛出异常或返回“unsupported input”错误；
4. 保证代码可以直接运行；
5. 保证返回值结构与该工具的接口风格一致。

【实现要求】
1. 使用 Python 实现。
2. 每个工具生成一个独立函数，函数名尽量与工具名对应。
3. 每个函数内部只做“固定输入 -> 固定输出”的映射，不要调用外部网络、数据库、系统命令或真实 API。
4. 必须保证 deterministic（同样输入永远返回同样输出）。
5. 对于未覆盖输入，统一：
   - 抛出 `NotImplementedError` 或 `ValueError`
   - 错误信息中写明“unsupported mocked input”
6. 如果一个工具输入是 JSON/dict，则优先基于“规范化后的 JSON 字符串”或关键字段匹配来判断。
7. 如果某些工具本身很复杂，也不要省略，至少给出 1~3 组固定样例。
8. 所有函数放在同一个 Python 文件中。
9. 生成一个 `run_demo()` 函数，演示每个工具至少一组输入输出。
10. 生成简单测试代码，使用 `assert` 校验固定样例。

【输出格式要求】
你必须严格只输出以下两段内容，不能多写任何解释、说明、前言、后记。

第一段：正式函数代码区
必须以这一行开始：
===PYTHON_MODULE_START===

必须以这一行结束：
===PYTHON_MODULE_END===

第二段：测试代码区
必须以这一行开始：
===PYTHON_TEST_START===

必须以这一行结束：
===PYTHON_TEST_END===

【强制格式要求】
1. 两段都必须是纯 Python 代码内容，不要再嵌套 markdown 代码块。
2. 输出中除这四个标记和两段 Python 代码外，不能有任何其他文字。
3. 正式函数代码区必须可以单独保存为一个 `.py` 文件并被 import。
4. 测试代码区必须可以单独提取出来用于测试。
5. 整个回答中，这四个标记各自只能出现一次。

下面是工具定义：
'''.strip()

    DIFFICULT_PROMPT = '''
请基于你生成的工具集合，设计一道“必须依赖这些能力才能得到答案”的标准问答题。

这里的“难”指的是：需要在同一个真实任务中做信息定位、条件筛选、一步到两步的组合推理；不是把多个无关领域的事实强行拼在一起。

要求如下：
1. 只生成 1 道题。
2. `PROMPT` 必须只包含一段自然、真实的用户提问，像聊天或业务场景里的正常询问，不要像测试脚本，不要出现“工具”“函数”“接口”“调用”等字样。
3. 问题必须围绕同一个主题、对象或任务场景展开，所有查询步骤都要服务于同一个明确目标。
4. 问题不能只靠常识直接回答，必须结合可获取的信息或可执行能力才能得出结果。
5. 难度主要来自“筛选条件、字段理解、结果比对、一步到两步计算或归纳”，不要来自生造背景、堆砌限制或跨领域混搭。
6. 最终答案必须是简短、唯一、可验证的客观答案，例如一个数字、一个日期、一个名称，或一个非常短的结构化结果。
7. 答案必须可以被明确校验，不能是开放式回答，不能模糊，不能带解释。
8. 最多只允许 2 到 3 个紧密相关的求解步骤；不要把多个彼此独立的小问题硬拼成一道题。
9. 禁止把体育、金融、农业、地理、医疗、娱乐等无关领域强行组合；禁止为了制造难度而引入与主任务无关的背景设定。
10. 禁止无意义的字符串加工，例如截取首字母、拼接代号、强制大小写变换、凑固定花哨格式；除非这本身就是任务目标中自然且必要的一部分。
11. 输出格式应尽量简单自然，优先使用单个值，或最多 2 个字段的短结构；不要设计冗长格式约束。
12. 不要输出解题过程，不要输出分析，不要输出额外说明。
13. 你必须严格按照我指定的三段格式输出，且字段名必须完全一致。
14. 题目应当使用部分或全部可用能力，但必须体现真实用户意图，不能显得像“为了覆盖能力而覆盖能力”。
15. PROMPT 请控制在 120 个汉字以内；如果超过，说明题目设计得过于复杂，请重写得更直接。
16. `PROMPT` 中禁止出现任何“mock / mocked / 模拟 / 测试数据 / 样例数据 / 当前数据集 / 本地数据 / 演示环境 / run_demo”之类暴露生成背景的话。
17. 不要在 `PROMPT` 里要求回答者“按某种格式作答”，不要出现“请将最终答案写成”“输出为”“返回为”等格式指令；这些只允许放在 `OUTPUT_FORMAT` 字段。
18. 题目必须确实需要借助外部能力检索或计算后才能回答。若工具之间存在依赖关系，可设计为串行；若存在可独立查询的子问题，可设计为并行；也可以是串并行混合。但这些求解结构不要明说在题面里。
19. 优先生成带有真实生活或业务语境的问题，例如查询、比对、筛选、排期、核验、推荐、定位、统计，不要生成“为了验证系统而提问”的句子。

请严格按照下面格式输出：

PROMPT:
<你生成的自然语言问题>

OUTPUT_FORMAT:
<该问题要求回答者使用的最终输出格式，例如：//box{值}>

ANSWER:
<该问题唯一且可验证的标准答案，必须严格符合 OUTPUT_FORMAT 中定义的格式>
'''.strip()

    EASY_PROMPT = '''
请基于我提供的工具集合，设计一道“必须依赖这些能力才能得到答案”的标准问答题。

要求如下：
1. 只生成 1 道题。
2. `PROMPT` 必须只包含一段自然、真实的用户提问，像聊天或业务场景里的正常询问，不要像测试脚本，不要出现“工具”“函数”“接口”“调用”等字样。
3. 问题必须围绕同一个主题、对象或任务场景展开，不要跨领域拼接无关信息。
4. 问题不能只靠常识直接回答，必须结合可获取的信息或可执行能力才能得出结果。
5. 最终答案必须是简短、唯一、可验证的客观答案，例如一个数字、一个日期、一个名称，或一个非常短的结构化结果。
6. 答案必须可以被明确校验，不能是开放式回答，不能模糊，不能带解释。
7. 问题难度适中，不要设计成超长推理题，也不要依赖主观判断。
8. 最多只允许 1 到 2 个紧密相关的求解步骤，不要堆砌条件，不要做无意义字符串拼接。
9. 输出格式应尽量简单自然，优先使用单个值，或最多 2 个字段的短结构。
10. 不要输出解题过程，不要输出分析，不要输出额外说明。
11. 你必须严格按照我指定的三段格式输出，且字段名必须完全一致。
12. 题目应当使用部分或全部可用能力，但必须体现真实用户意图，不能显得像“为了覆盖能力而覆盖能力”。
13. `PROMPT` 中禁止出现任何“mock / mocked / 模拟 / 测试数据 / 样例数据 / 当前数据集 / 本地数据 / 演示环境 / run_demo”之类暴露生成背景的话。
14. 不要在 `PROMPT` 里要求回答者“按某种格式作答”，不要出现“请将最终答案写成”“输出为”“返回为”等格式指令；这些只允许放在 `OUTPUT_FORMAT` 字段。
15. 题目必须确实需要借助外部能力检索或计算后才能回答；根据工具关系，可以是串行、并行或串并行混合求解，但不要把这种结构直接写进题面。
16. 优先生成带有真实生活或业务语境的问题，例如查询、比对、筛选、排期、核验、推荐、定位、统计，不要生成“为了验证系统而提问”的句子。

请严格按照下面格式输出：

PROMPT:
<你生成的自然语言问题>

OUTPUT_FORMAT:
<该问题要求回答者使用的最终输出格式，例如：//box{值}>

ANSWER:
<该问题唯一且可验证的标准答案，必须严格符合 OUTPUT_FORMAT 中定义的格式>
'''.strip()

    @classmethod
    def get_qa_prompt(cls, mode: str) -> str:
        return cls.DIFFICULT_PROMPT if mode == "difficult" else cls.EASY_PROMPT

    @classmethod
    def build_original_question_context(cls, original_question: Optional[str]) -> str:
        if not original_question:
            return ""
        return (
            "\n\n【原始用户问题（仅作语义参考，不可照抄）】\n"
            f"{original_question.strip()}\n"
        )


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
    "run_demo_found": False,
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

ordered_names = []
if hasattr(test_mod, "run_demo") and callable(getattr(test_mod, "run_demo")):
    REPORT["run_demo_found"] = True
    ordered_names.append("run_demo")

test_names = sorted(
    name for name in dir(test_mod)
    if name.startswith("test_") and callable(getattr(test_mod, name))
)
REPORT["test_functions"] = test_names
ordered_names.extend(test_names)

for name in ordered_names:
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

        if "unsupported mocked input" not in module_code:
            issues.append("module code does not contain 'unsupported mocked input'")

        if "assert " not in test_code and "assert(" not in test_code:
            issues.append("test code does not contain assert")

        if "def run_demo" not in test_code:
            issues.append("test code does not contain run_demo")

        if "def test_" not in test_code:
            issues.append("test code does not contain any test_* function")

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
                "run_demo_found": False,
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
                    "run_demo_found": False,
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
                    "run_demo_found": False,
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
                    "run_demo_found": False,
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

    def build_mock_prompt(self, tools: List[Dict[str, Any]]) -> str:
        lines: List[str] = [PromptRepository.MOCK_PROMPT, "", "【可用工具】"]

        if not tools:
            lines.append("无")
            return "\n".join(lines)

        for i, t in enumerate(tools, 1):
            fn = t.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            required = set(params.get("required", []))
            props = params.get("properties", {})

            lines.append(f"{i}. {name}")
            if desc:
                lines.append(f"   描述: {desc}")

            if props:
                lines.append("   参数:")
                for p_name, p_info in props.items():
                    p_type = p_info.get("type", "any")
                    p_desc = p_info.get("description", "")
                    req_mark = " [必填]" if p_name in required else ""
                    line = f"   - {p_name}: {p_type}{req_mark}"
                    if p_desc:
                        line += f" - {p_desc}"
                    lines.append(line)
            else:
                lines.append("   参数: 无")

        return "\n".join(lines)

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

    def build_mock_prompt_with_context(
        self,
        tools: List[Dict[str, Any]],
        original_question: str = "",
    ) -> str:
        base_prompt = self.build_mock_prompt(tools)
        if not (self.config.use_original_question_for_generation and original_question):
            return base_prompt

        guidance = '''
【原始问题增强要求】
下面我会额外提供一条原始用户问题，目的是帮助你理解这些工具最可能被怎样组合使用。
请据此优化你生成的 mock 函数和测试样例：
1. 优先覆盖原始问题中最核心、最可能出现的参数组合与查询路径。
2. 如果原始问题涉及多步查询，请尽量让固定样例能支撑这类串行、并行或串并行混合求解。
3. 仍然只生成 deterministic 的固定映射，不要实现真实外部能力。
4. 原始问题只是帮助你挑选更贴近真实场景的 mocked 输入输出，不代表你必须逐字复现其中每个字段。
5. 不要在输出代码或错误信息中泄露“原始问题增强”“参考问题”等字样。
'''.strip()

        return (
            base_prompt
            + "\n"
            + guidance
            + PromptRepository.build_original_question_context(original_question)
        )

    def build_qa_prompt_with_context(self, original_question: str = "") -> str:
        base_prompt = PromptRepository.get_qa_prompt(self.config.qa_mode)
        if not (self.config.use_original_question_for_generation and original_question):
            return base_prompt

        guidance = '''
【原始问题增强要求】
下面我会额外提供一条原始用户问题，目的是帮助你生成更高质量的新题目。
请严格遵守以下要求：
1. 将它视为“语义风格与真实任务意图”的参考，而不是要你原样改写。
2. 你生成的是一道新的题目，必须仍然以我给定的 `PROMPT / OUTPUT_FORMAT / ANSWER` 三段格式输出。
3. 新题目应尽量继承原始问题的真实感、任务目标和信息组织方式，但要结合当前你已经生成的 mock 工具能力，确保答案可被当前 mock 数据唯一支撑。
4. 允许你对原始问题做收缩、重组、具体化或轻度改写，让它更适合产出唯一、可校验、便于 RL 使用的最终答案。
5. 如果原始问题本身没有约束最终答案格式，请你自行把结果收束成简短唯一的目标，并把格式要求只写在 `OUTPUT_FORMAT` 中，不要写进 `PROMPT`。
6. 不要在 `PROMPT` 中提及原始问题、mock、测试、样例、数据集等生成背景。
'''.strip()

        return (
            base_prompt
            + "\n"
            + guidance
            + PromptRepository.build_original_question_context(original_question)
        )

    def should_process(self, sample: Dict[str, Any]) -> bool:
        return self.count_tool_calls(sample) >= self.config.min_tool_calls

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
            mock_prompt = self.build_mock_prompt_with_context(tools, original_question)
            messages = [{"role": "user", "content": mock_prompt}]
            mock_response = self.llm.chat(messages)
            messages.append({"role": "assistant", "content": mock_response})

            parsed_code = ResponseParser.extract_python_blocks(mock_response)
            module_code = parsed_code["module_code"]
            test_code = parsed_code["test_code"]

            # 2) 基于已有上下文生成 QA
            qa_prompt = self.build_qa_prompt_with_context(original_question)
            qa_messages = messages + [{"role": "user", "content": qa_prompt}]
            qa_response = self.llm.chat(qa_messages)
            qa_info = ResponseParser.extract_qa(qa_response)

            # 3) 执行 mock 测试
            test_report = self.test_runner.run(module_code, test_code)

            record.update({
                "status": "ok" if test_report.get("ok") else "test_failed",
                "qa_mode": self.config.qa_mode,
                "qa": qa_info,
                "test_report": test_report,
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
    config = PipelineConfig(
        input_path="/mnt/code/yehangcheng/all_data/sft_data/Nemotron-Post-Training-Dataset-v1/tool_use_no_think_v2.jsonl",

        # 只保存最终成功样本
        output_path="/mnt/code/yehangcheng/Intruct_augment/gen_data/agent_syn_data/synthetic_mock_success_only_2.jsonl",

        # 保存全部过程记录，便于排查和断点续跑观察
        progress_path="/mnt/code/yehangcheng/Intruct_augment/gen_data/agent_syn_data/synthetic_mock_progress.jsonl",

        # 如果你只想处理并行 / 多工具样本，就改成 2
        min_tool_calls=2,

        # 全量跑
        max_samples=5000,

        qa_mode="difficult",
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
