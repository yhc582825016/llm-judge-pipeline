# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# from __future__ import annotations

# import argparse
# import json
# import logging
# import time
# import re
# from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import Any, Dict, Optional
# import random

# import pandas as pd
# import requests
# from tqdm import tqdm

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger("sglang_runner_simple")

# # 默认 prompt_map（可被 --prompt-map-file 覆盖）
# DEFAULT_PROMPT_MAP = {
#     "default": "请判断下面文本是否违规，直接返回 JSON 或简短文本：\n\n{text}",
#     "文本分类": "请判断下面文本的类别，仅返回 JSON：{\"label\": \"类别名\"}\n\n文本:\n{text}",
#     "打分": "请对下面文本从0到100进行打分，并返回 JSON：{\"score\": <0-100>}\n\n文本:\n{text}"
# }

# # 最大 prompt 文本长度（字符），可在运行时由 main() 调整
# DEFAULT_MAX_TEXT_LENGTH = 4000

# def load_prompt_map_from_py(file_path: Optional[str]) -> Dict[str, str]:
#     """
#     从指定的 Python 文件中动态导入 prompt 内容。
#     要求文件中定义若干字符串变量，例如：
#         base_prompt = xxx
#         classify_prompt = xxx
#     最终返回字典 {"base_prompt": "...", "classify_prompt": "..."}
#     """
#     pm = DEFAULT_PROMPT_MAP.copy()
#     if not file_path:
#         return pm
#     try:
#         import importlib.util
#         import pathlib

#         file_path = str(pathlib.Path(file_path).resolve())
#         spec = importlib.util.spec_from_file_location("prompt_module", file_path)
#         module = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(module)  # 执行文件

#         # 收集文件中所有字符串变量
#         for k, v in vars(module).items():
#             if not k.startswith("_") and isinstance(v, str):
#                 pm[k] = v
#         logger.info("从 %s 加载 %d 个 prompt 模板变量", file_path, len(pm))
#     except Exception as e:
#         logger.warning("加载 Python prompt 文件失败，使用默认模板: %s", e)
#     return pm


# def extract_text_from_orig(orig: Any, field: str = "prompt") -> Optional[str]:
#     if orig is None:
#         return None
#     if isinstance(orig, str):
#         try:
#             orig = json.loads(orig)
#         except Exception:
#             # 字符串不是 JSON，直接作为文本返回
#             return orig
#     convs = None
#     if isinstance(orig, dict):
#         convs = orig.get("conversations") or orig.get("conversation") or orig.get("messages") or orig.get("dialogue")
#     if convs is None:
#         if isinstance(orig, dict) and "content" in orig and "role" in orig:
#             return orig.get("content")
#         # 不是会话结构，尝试取常见字段
#         if isinstance(orig, dict):
#             for k in ("text", "message", "content", "prompt", "input"):
#                 if k in orig and isinstance(orig[k], (str,)):
#                     return orig[k]
#         return None
#     last_user = None
#     last_assistant = None
#     # print(convs)
#     for item in convs:
#         if not isinstance(item, dict):
#             continue
#         role = item.get("role") or item.get("sender") or item.get("from")
#         content = item.get("content") or item.get("text") or item.get("message")
#         if role is None:
#             continue
#         r = str(role).lower()
#         if r in ("user", "human", "usr", "客户", "用户"):
#             last_user = content
#         elif r in ("assistant", "agent", "bot", "客服"):
#             last_assistant = content
#     return last_user if field == "prompt" else last_assistant

# def sanitize_and_truncate_text(text: Any, max_len: Optional[int] = None) -> str:
#     """
#     将输入转换为字符串、去首尾空白并按 max_len 截断。
#     如果 max_len 为 None，则使用模块级 DEFAULT_MAX_TEXT_LENGTH（可在运行时修改）。
#     """
#     if text is None:
#         return ""
#     if not isinstance(text, str):
#         text = str(text)
#     text = text.strip()
#     effective_max = max_len if (max_len is not None) else DEFAULT_MAX_TEXT_LENGTH
#     try:
#         if len(text) > effective_max:
#             return text[:effective_max] + "...(truncated)"
#     except Exception:
#         # 如果 len() 出错（极少），退回原始字符串
#         return text
#     return text

# def build_prompt(prompt_key: str, row: pd.Series, prompt_map: Dict[str,str], extra: Optional[str]=None) -> str:
#     """
#     使用安全替换方式：只替换模板中的 {text} 占位符，避免模板中出现 JSON 花括号引发 str.format 的问题。
#     若模板没有 {text} 占位符，则在模板后追加文本。
#     """
#     if prompt_key not in prompt_map:
#         prompt_key = "default"
#     template = prompt_map[prompt_key]
    

#     prompt = extract_text_from_orig(row.get("orig"), field="prompt") or ""
#     response = json.loads(row.get("orig"))['response']
#     # print('response',row.get("orig"))
#     prompt = sanitize_and_truncate_text(prompt)
#     response = sanitize_and_truncate_text(response)

#     # 如果模板包含显式的 {text} 占位符，直接替换（不会触发 format 对其它花括号的解析）
#     if "{text}" in template:
#         try:
#             return template.replace("{text}", prompt)
#         except Exception:
#             # 保险回退
#             return template + "\n\n" + prompt
    
#     elif "{question}" in template and "{answer}" in template:
#         return template.replace("{question}", prompt).replace("{answer}", response)
#     elif "{question}" in template and "{answer}" not in template:
#         return template.replace("{question}", prompt)
#     else:
#         return template + "\n\n" + prompt

# def try_parse_json_from_text(text: Optional[str]) -> Optional[Dict[str,Any]]:
#     if text is None:
#         return None
#     if isinstance(text, dict):
#         return text
#     t = text.strip()
#     try:
#         parsed = json.loads(t)
#         if isinstance(parsed, dict):
#             return parsed
#     except Exception:
#         pass
#     m = re.search(r"\{.*\}", t, flags=re.DOTALL)
#     if m:
#         candidate = m.group(0)
#         try:
#             parsed = json.loads(candidate)
#             if isinstance(parsed, dict):
#                 return parsed
#         except Exception:
#             # 尝试修正单引号
#             try:
#                 parsed = json.loads(candidate.replace("'", '"'))
#                 if isinstance(parsed, dict):
#                     return parsed
#             except Exception:
#                 return None
#     return None

# def forward_local_api_openai(message: str, api_key: Optional[str], base_url: str, model: str,
#                              temperature: float, max_retries:int, sleep_duration:float, extra_body:Optional[dict]):
#     try:
#         from openai import OpenAI
#     except Exception as e:
#         return {"status":"import_error","text":None,"raw":str(e)}
#     base = base_url if base_url.endswith("/v1") else base_url.rstrip("/") + "/v1"
#     client = OpenAI(api_key=api_key, base_url=base)
#     stop_tokens = ["<|eot_id|>","<|im_end|>","</s>","<|endoftext|>","</answer>"]
#     extra = extra_body or {"repetition_penalty":1.05,"chat_template_kwargs":{"enable_thinking":False}}
#     attempt = 0
#     while attempt <= max_retries:
#         print('message',message)
#         try:
#             completion = client.chat.completions.create(
#                 model=model,
#                 messages=[{"role":"user","content":message}],
#                 temperature=temperature,
#                 stop=stop_tokens,
#                 max_tokens=12000,
#                 extra_body=extra
#             )
#             try:
#                 text_out = completion.choices[0].message.content
#             except Exception:
#                 try:
#                     text_out = (completion.get("choices",[{}])[0].get("message",{}) or {}).get("content")
#                 except Exception:
#                     text_out = str(completion)
#             if text_out and '</think>' in text_out:
#                 text_out = text_out.split('</think>')[-1]
#             return {"status":"ok","text":text_out,"raw":completion}
#         except Exception as e:
#             logger.warning("openai client 调用异常: %s", e)
#             attempt += 1
#             if attempt>max_retries: break
#             time.sleep(sleep_duration)
#     return {"status":"failed","text":None,"raw":f"failed_after_{max_retries}"}

# def forward_local_api_requests(message: str, endpoint: str, headers:Dict[str,str], timeout:float, max_retries:int, sleep_duration:float):
#     payload = {"text": message}
#     attempt = 0
#     while attempt <= max_retries:
#         try:
#             resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
#             try:
#                 raw = resp.json()
#             except:
#                 raw = resp.text
#             if resp.status_code==200:
#                 text = None
#                 if isinstance(raw, dict):
#                     text = raw.get("text") or raw.get("result") or raw.get("output") or raw.get("reply")
#                 if text is None:
#                     text = str(raw)
#                 return {"status":"ok","text":text,"raw":raw}
#             else:
#                 logger.warning("非200响应 %s", resp.status_code)
#                 attempt += 1
#                 time.sleep(sleep_duration)
#         except Exception as e:
#             logger.warning("requests 调用异常: %s", e)
#             attempt += 1
#             time.sleep(sleep_duration)
#     return {"status":"failed","text":None,"raw":f"failed_after_{max_retries}"}

# def worker(row_idx:int, row:pd.Series, field:str, use_openai:bool, api_key:Optional[str], base_url:str, model:str,
#            timeout:float, max_retries:int, sleep_duration:float, headers:Dict[str,str], prompt_key_arg:str, prompt_map:Dict[str,str], prompt_extra:Optional[str], extra_body:Optional[dict]):
#     # extract text existence check
#     orig = row.get("orig")
#     text = extract_text_from_orig(orig, field=field)
#     # print(orig)
#     if text is None or (isinstance(text,str) and text.strip()==""):
#         return (row_idx, None)
#     # determine prompt_key per row
#     row_prompt_key = row.get("prompt_key") if ("prompt_key" in row.index and row.get("prompt_key") not in (None,"")) else prompt_key_arg
#     prompt = build_prompt(row_prompt_key, row, prompt_map, extra=prompt_extra)
#     # print(prompt)
#     logger.debug("Prompt for row %s: %s", row_idx, prompt)
#     # forward
#     if use_openai:
#         res = forward_local_api_openai(message=prompt, api_key=api_key, base_url=base_url, model=model, temperature=0, max_retries=max_retries, sleep_duration=sleep_duration, extra_body=extra_body)
#         # print('res',res)
#     else:
#         res = forward_local_api_requests(message=prompt, endpoint=base_url, headers=headers, timeout=timeout, max_retries=max_retries, sleep_duration=sleep_duration)
#     text_out = res.get("text")
#     # try parse json first (prefer structured result)
#     # parsed = try_parse_json_from_text(text_out) if isinstance(text_out,str) else None
#     # if parsed is not None:
#     #     # store parsed JSON as compact string
#     #     try:
#     #         return (row_idx, json.dumps(parsed, ensure_ascii=False))
#     #     except Exception:
#     #         return (row_idx, str(parsed))
#     # fallback to raw text_out
#     if text_out is not None:
#         return (row_idx, text_out)
#     # fallback to raw response dump
#     raw = res.get("raw")
#     try:
#         return (row_idx, json.dumps(raw, ensure_ascii=False))
#     except Exception:
#         return (row_idx, str(raw))

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--input","-i", required=True)
#     p.add_argument("--output","-o", required=True)
#     p.add_argument("--field", choices=["prompt","response"], default="prompt")
#     p.add_argument("--use-openai-client", action="store_true")
#     p.add_argument("--api-key", default=None)
#     p.add_argument("--base-url", default="http://localhost:8035/v1")
#     p.add_argument("--model", default="Qwen3-30B-A3B-Base-general_totals_202w_concat_lmsys_278w-ep3")
#     p.add_argument("--concurrency", type=int, default=8)
#     p.add_argument("--timeout", type=float, default=30.0)
#     p.add_argument("--max-retries", type=int, default=5)
#     p.add_argument("--sleep-duration", type=float, default=5.0)
#     p.add_argument("--header", action="append")
#     p.add_argument("--sample", type=int, default=0)
#     p.add_argument("--prompt-key", type=str, default="default")
#     p.add_argument("--prompt-map-file", type=str, default=None)
#     p.add_argument("--prompt-extra", type=str, default=None)
#     p.add_argument("--extra-body", type=str, default=None)
#     p.add_argument("--max-text-len", type=int, default=DEFAULT_MAX_TEXT_LENGTH)
#     args = p.parse_args()

#     headers = {}
#     if args.header:
#         for h in args.header:
#             if ":" in h:
#                 k,v = h.split(":",1)
#                 headers[k.strip()] = v.strip()
#     extra_body = None
#     if args.extra_body:
#         try:
#             extra_body = json.loads(args.extra_body)
#         except Exception:
#             extra_body = None

#     prompt_map = load_prompt_map_from_py(args.prompt_map_file)

#     logger.info("读取 parquet：%s", args.input)
#     df = pd.read_parquet(args.input, engine="pyarrow")
#     logger.info("数据行数：%d", len(df))
#     if args.sample and args.sample>0:
#         df = df.iloc[:args.sample].copy()
#         logger.info("sample 模式，处理前 %d 行", len(df))
#     df = df.reset_index(drop=True)   # 放在 sample 之后更稳
#     # 使用 globals() 动态设置模块级默认值，避免在函数中声明 global 导致的 SyntaxError
#     globals()["DEFAULT_MAX_TEXT_LENGTH"] = args.max_text_len

#     n = len(df)
#     results = [None] * n

#     logger.info("启动线程池：%d 并发", args.concurrency)
#     with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
#         futures = {}
#         for i,row in df.iterrows():
#             fut = exe.submit(worker, i, row, args.field, args.use_openai_client, args.api_key, args.base_url, args.model,
#                              args.timeout, args.max_retries, args.sleep_duration, headers, args.prompt_key, prompt_map, args.prompt_extra, extra_body)
#             futures[fut] = i
#         # 使用 tqdm 显示处理进度（基于 futures 完成数）
#         with tqdm(total=len(futures), desc="Processing", unit="row") as pbar:
#             for fut in as_completed(futures):
#                 idx = futures[fut]
#                 try:
#                     r = fut.result()
#                     if r is None:
#                         results[idx] = None
#                     else:
#                         ri, val = r
#                         results[ri] = val
#                 except Exception as e:
#                     logger.exception("future 异常 %s", e)
#                     results[idx] = None
#                 finally:
#                     pbar.update(1)

#     logger.info("将结果写回 sglang_result 字段并保存到：%s", args.output)
#     df["sglang_result"] = results
#     df.to_parquet(args.output, engine="pyarrow", index=False)
#     logger.info("完成，输出文件：%s", args.output)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sglang_runner_simple")

# 默认 prompt_map（可被 --prompt-map-file 覆盖）
DEFAULT_PROMPT_MAP = {
    "default": "请判断下面文本是否违规，直接返回 JSON 或简短文本：\n\n{text}",
    "文本分类": "请判断下面文本的类别，仅返回 JSON：{\"label\": \"类别名\"}\n\n文本:\n{text}",
    "打分": "请对下面文本从0到100进行打分，并返回 JSON：{\"score\": <0-100>}\n\n文本:\n{text}",
    "问答判断": "请阅读下面的问答对，并判断回答是否切题、是否有帮助。仅返回 JSON。\n\n{text}",
    "问答打分": "请对下面问答对的回答质量从0到100打分，并返回 JSON：{\"score\": <0-100>}\n\n{text}"
}

# 最大 prompt 文本长度（字符），可在运行时由 main() 调整
DEFAULT_MAX_TEXT_LENGTH = 4000


def load_prompt_map_from_py(file_path: Optional[str]) -> Dict[str, str]:
    """
    从指定的 Python 文件中动态导入 prompt 内容。
    要求文件中定义若干字符串变量，例如：
        base_prompt = xxx
        classify_prompt = xxx
    最终返回字典 {"base_prompt": "...", "classify_prompt": "..."}
    """
    pm = DEFAULT_PROMPT_MAP.copy()
    if not file_path:
        return pm
    try:
        import importlib.util
        import pathlib

        file_path = str(pathlib.Path(file_path).resolve())
        spec = importlib.util.spec_from_file_location("prompt_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # 执行文件

        # 收集文件中所有字符串变量
        for k, v in vars(module).items():
            if not k.startswith("_") and isinstance(v, str):
                pm[k] = v
        logger.info("从 %s 加载 %d 个 prompt 模板变量", file_path, len(pm))
    except Exception as e:
        logger.warning("加载 Python prompt 文件失败，使用默认模板: %s", e)
    return pm


def maybe_json_loads(obj: Any) -> Any:
    """
    如果 obj 是 JSON 字符串则解析，否则原样返回。
    """
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return obj
        try:
            return json.loads(s)
        except Exception:
            return obj
    return obj


def ndarray_to_list_if_needed(obj: Any) -> Any:
    """
    兼容 numpy.ndarray(dtype=object) -> list
    不强依赖 numpy，避免环境没有 numpy 时出错。
    """
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return obj


def get_messages_like_list(obj: Any) -> Optional[list]:
    """
    将不同格式统一解析成消息列表：
    1. 直接就是 list
    2. numpy array(dtype=object)
    3. JSON 字符串，解析后是 list/dict
    4. dict 里包含 conversations/conversation/messages/dialogue
    """
    if obj is None:
        return None

    obj = maybe_json_loads(obj)
    obj = ndarray_to_list_if_needed(obj)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for k in ("conversations", "conversation", "messages", "dialogue"):
            if k in obj:
                v = obj.get(k)
                v = maybe_json_loads(v)
                v = ndarray_to_list_if_needed(v)
                if isinstance(v, list):
                    return v

    return None


def normalize_role(role: Any) -> Optional[str]:
    if role is None:
        return None
    r = str(role).strip().lower()

    if r in ("user", "human", "usr", "客户", "用户"):
        return "user"
    if r in ("assistant", "agent", "bot", "客服", "助手", "assistant_response", "gpt", "model"):
        return "assistant"
    if r in ("system", "sys"):
        return "system"
    return r


def normalize_message_item(item: Any) -> Optional[Dict[str, str]]:
    """
    将一条消息统一成:
    {
        "role": "user/assistant/system/...",
        "content": "..."
    }
    """
    if item is None:
        return None

    item = maybe_json_loads(item)
    item = ndarray_to_list_if_needed(item)

    if isinstance(item, dict):
        role = item.get("role") or item.get("sender") or item.get("from")
        content = item.get("content") or item.get("text") or item.get("message")
        if content is None:
            return None
        return {
            "role": normalize_role(role) or "",
            "content": str(content)
        }

    if isinstance(item, str):
        return {
            "role": "",
            "content": item
        }

    return None


def get_normalized_messages(obj: Any) -> List[Dict[str, str]]:
    """
    将各种 messages/orig/conversations 格式统一成标准消息列表。
    """
    convs = get_messages_like_list(obj)
    if convs is None:
        return []

    out = []
    for item in convs:
        item = normalize_message_item(item)
        if item is not None:
            out.append(item)
    return out


def extract_text_from_orig(orig: Any, field: str = "prompt") -> Optional[str]:
    """
    兼容以下格式：
    1. orig 本身就是 conversations/messages/dialogue 的 dict
    2. orig 是 JSON 字符串
    3. orig 是单条消息 dict，如 {"role": "...", "content": "..."}
    4. orig 是普通 dict，包含 text/message/content/prompt/input/response/solution 等字段

    field:
    - prompt   -> 最后一个 user
    - response -> 最后一个 assistant
    """
    if orig is None:
        return None

    orig_loaded = maybe_json_loads(orig)
    orig_loaded = ndarray_to_list_if_needed(orig_loaded)

    msgs = get_normalized_messages(orig_loaded)
    if msgs:
        last_user = None
        last_assistant = None

        for item in msgs:
            role = item.get("role")
            content = item.get("content")
            if role == "user":
                last_user = content
            elif role == "assistant":
                last_assistant = content

        return last_user if field == "prompt" else last_assistant

    if isinstance(orig_loaded, dict) and "content" in orig_loaded and "role" in orig_loaded:
        role = normalize_role(orig_loaded.get("role"))
        content = orig_loaded.get("content")
        if content is None:
            return None
        if field == "prompt" and role == "user":
            return str(content)
        if field == "response" and role == "assistant":
            return str(content)
        return None

    if isinstance(orig_loaded, dict):
        if field == "prompt":
            for k in ("text", "message", "content", "prompt", "input", "question"):
                if k in orig_loaded and isinstance(orig_loaded[k], str):
                    return orig_loaded[k]
        else:
            for k in ("response", "answer", "output", "solution", "label"):
                if k in orig_loaded and isinstance(orig_loaded[k], str):
                    return orig_loaded[k]

    if isinstance(orig_loaded, str):
        return orig_loaded if field == "prompt" else None

    return None


def extract_last_user_assistant_pair(obj: Any) -> Optional[Dict[str, Optional[str]]]:
    """
    提取最后一轮 user + assistant：
    从后往前找到最后一个 assistant，
    再往前找到离它最近的 user。

    返回:
        {
            "user": ...,
            "assistant": ...
        }

    若没有 assistant，则返回最后一个 user + None
    """
    msgs = get_normalized_messages(obj)
    if not msgs:
        return None

    pairs: List[Tuple[str, str]] = []
    for item in msgs:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and content is not None:
            pairs.append((role, content))

    if not pairs:
        return None

    last_assistant_idx = None
    for i in range(len(pairs) - 1, -1, -1):
        if pairs[i][0] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        for i in range(len(pairs) - 1, -1, -1):
            if pairs[i][0] == "user":
                return {"user": pairs[i][1], "assistant": None}
        return None

    assistant_text = pairs[last_assistant_idx][1]
    user_text = None

    for i in range(last_assistant_idx - 1, -1, -1):
        if pairs[i][0] == "user":
            user_text = pairs[i][1]
            break

    return {"user": user_text, "assistant": assistant_text}


def format_last_turn_pair(pair: Optional[Dict[str, Optional[str]]]) -> Optional[str]:
    if pair is None:
        return None

    user_text = pair.get("user") or ""
    assistant_text = pair.get("assistant") or ""

    if user_text and assistant_text:
        return f"用户: {user_text}\n助手: {assistant_text}"
    if user_text:
        return f"用户: {user_text}"
    if assistant_text:
        return f"助手: {assistant_text}"
    return None


def extract_text_from_row(row: pd.Series, field: str = "prompt") -> Optional[str]:
    """
    优先从整行数据中提取文本，兼容：
    1. row["messages"]
    2. row["orig"]
    3. row["solution"] / 其它常见字段

    field:
    - prompt
    - response
    - last_turn
    """
    if field == "last_turn":
        if "messages" in row.index:
            pair = extract_last_user_assistant_pair(row.get("messages"))
            text = format_last_turn_pair(pair)
            if text not in (None, ""):
                return text

        if "orig" in row.index:
            pair = extract_last_user_assistant_pair(row.get("orig"))
            text = format_last_turn_pair(pair)
            if text not in (None, ""):
                return text

        # 兜底：若没有 messages/orig，可尝试普通字段拼接
        prompt_text = None
        response_text = None

        for k in ("prompt", "input", "question", "text", "content"):
            if k in row.index and pd.notna(row.get(k)):
                prompt_text = str(row.get(k))
                break

        for k in ("response", "answer", "output", "solution", "label"):
            if k in row.index and pd.notna(row.get(k)):
                response_text = str(row.get(k))
                break

        if prompt_text and response_text:
            return f"用户: {prompt_text}\n助手: {response_text}"
        if prompt_text:
            return f"用户: {prompt_text}"
        if response_text:
            return f"助手: {response_text}"
        return None

    # field == prompt / response
    if "messages" in row.index:
        val = extract_text_from_orig(row.get("messages"), field=field)
        if val not in (None, ""):
            return val

    if "orig" in row.index:
        val = extract_text_from_orig(row.get("orig"), field=field)
        if val not in (None, ""):
            return val

    if field == "prompt":
        for k in ("prompt", "input", "question", "text", "content"):
            if k in row.index and pd.notna(row.get(k)):
                return str(row.get(k))
    else:
        for k in ("response", "answer", "output", "solution", "label"):
            if k in row.index and pd.notna(row.get(k)):
                return str(row.get(k))

    return None


def sanitize_and_truncate_text(text: Any, max_len: Optional[int] = None) -> str:
    """
    将输入转换为字符串、去首尾空白并按 max_len 截断。
    如果 max_len 为 None，则使用模块级 DEFAULT_MAX_TEXT_LENGTH（可在运行时修改）。
    """
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


def build_prompt(prompt_key: str, row: pd.Series, field: str, prompt_map: Dict[str, str], extra: Optional[str] = None) -> str:
    """
    支持占位符：
    - {text}
    - {question}
    - {answer}

    其中：
    - field=prompt    时，{text} 默认取最后一个 user
    - field=response  时，{text} 默认取最后一个 assistant
    - field=last_turn 时，{text} 默认取最后一轮 user+assistant
    """
    if prompt_key not in prompt_map:
        prompt_key = "default"
    template = prompt_map[prompt_key]

    main_text = extract_text_from_row(row, field=field) or ""
    question = extract_text_from_row(row, field="prompt") or ""
    answer = extract_text_from_row(row, field="response") or ""

    main_text = sanitize_and_truncate_text(main_text)
    question = sanitize_and_truncate_text(question)
    answer = sanitize_and_truncate_text(answer)

    if extra:
        extra = str(extra).strip()
        if extra:
            main_text = f"{main_text}\n\n{extra}" if main_text else extra

    if "{question}" in template or "{answer}" in template:
        out = template
        out = out.replace("{question}", question)
        out = out.replace("{answer}", answer)
        return out

    if "{text}" in template:
        return template.replace("{text}", main_text)

    return template + "\n\n" + main_text


def try_parse_json_from_text(text: Optional[str]) -> Optional[Dict[str, Any]]:
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
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            try:
                parsed = json.loads(candidate.replace("'", '"'))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
    return None


def forward_local_api_openai(
    message: str,
    api_key: Optional[str],
    base_url: str,
    model: str,
    temperature: float,
    max_retries: int,
    sleep_duration: float,
    extra_body: Optional[dict]
):
    try:
        from openai import OpenAI
    except Exception as e:
        return {"status": "import_error", "text": None, "raw": str(e)}

    base = base_url if base_url.endswith("/v1") else base_url.rstrip("/") + "/v1"
    client = OpenAI(api_key=api_key, base_url=base)
    stop_tokens = ["<|eot_id|>", "<|im_end|>", "</s>", "<|endoftext|>", "</answer>"]
    extra = extra_body or {"repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}}

    attempt = 0
    while attempt <= max_retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
                stop=stop_tokens,
                max_tokens=12000,
                extra_body=extra
            )
            try:
                text_out = completion.choices[0].message.content
            except Exception:
                try:
                    text_out = (completion.get("choices", [{}])[0].get("message", {}) or {}).get("content")
                except Exception:
                    text_out = str(completion)

            if text_out and "</think>" in text_out:
                text_out = text_out.split("</think>")[-1]
            return {"status": "ok", "text": text_out, "raw": completion}
        except Exception as e:
            logger.warning("openai client 调用异常: %s", e)
            attempt += 1
            if attempt > max_retries:
                break
            time.sleep(sleep_duration)

    return {"status": "failed", "text": None, "raw": f"failed_after_{max_retries}"}


def forward_local_api_requests(
    message: str,
    endpoint: str,
    headers: Dict[str, str],
    timeout: float,
    max_retries: int,
    sleep_duration: float
):
    payload = {"text": message}
    attempt = 0
    while attempt <= max_retries:
        try:
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
            logger.warning("requests 调用异常: %s", e)
            attempt += 1
            time.sleep(sleep_duration)

    return {"status": "failed", "text": None, "raw": f"failed_after_{max_retries}"}


def worker(
    row_idx: int,
    row: pd.Series,
    field: str,
    use_openai: bool,
    api_key: Optional[str],
    base_url: str,
    model: str,
    timeout: float,
    max_retries: int,
    sleep_duration: float,
    headers: Dict[str, str],
    prompt_key_arg: str,
    prompt_map: Dict[str, str],
    prompt_extra: Optional[str],
    extra_body: Optional[dict]
):
    # 从整行中提取文本，兼容 messages / orig / solution 等格式
    text = extract_text_from_row(row, field=field)
    if text is None or (isinstance(text, str) and text.strip() == ""):
        return (row_idx, None)

    row_prompt_key = row.get("prompt_key") if ("prompt_key" in row.index and row.get("prompt_key") not in (None, "")) else prompt_key_arg
    prompt = build_prompt(row_prompt_key, row, field, prompt_map, extra=prompt_extra)

    logger.debug("Prompt for row %s: %s", row_idx, prompt)

    if use_openai:
        res = forward_local_api_openai(
            message=prompt,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0,
            max_retries=max_retries,
            sleep_duration=sleep_duration,
            extra_body=extra_body
        )
    else:
        res = forward_local_api_requests(
            message=prompt,
            endpoint=base_url,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            sleep_duration=sleep_duration
        )

    text_out = res.get("text")

    # 如你后续想优先保存成结构化 JSON，可以把下面注释放开
    # parsed = try_parse_json_from_text(text_out) if isinstance(text_out, str) else None
    # if parsed is not None:
    #     try:
    #         return (row_idx, json.dumps(parsed, ensure_ascii=False))
    #     except Exception:
    #         return (row_idx, str(parsed))

    if text_out is not None:
        return (row_idx, text_out)

    raw = res.get("raw")
    try:
        return (row_idx, json.dumps(raw, ensure_ascii=False))
    except Exception:
        return (row_idx, str(raw))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--field", choices=["prompt", "response", "last_turn"], default="prompt")
    p.add_argument("--use-openai-client", action="store_true")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default="http://localhost:8035/v1")
    p.add_argument("--model", default="Qwen3-30B-A3B-Base-general_totals_202w_concat_lmsys_278w-ep3")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--sleep-duration", type=float, default=5.0)
    p.add_argument("--header", action="append")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--prompt-key", type=str, default="default")
    p.add_argument("--prompt-map-file", type=str, default=None)
    p.add_argument("--prompt-extra", type=str, default=None)
    p.add_argument("--extra-body", type=str, default=None)
    p.add_argument("--max-text-len", type=int, default=DEFAULT_MAX_TEXT_LENGTH)
    args = p.parse_args()

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

    prompt_map = load_prompt_map_from_py(args.prompt_map_file)

    logger.info("读取 parquet：%s", args.input)
    df = pd.read_parquet(args.input, engine="pyarrow")
    logger.info("数据行数：%d", len(df))

    if args.sample and args.sample > 0:
        df = df.iloc[:args.sample].copy()
        logger.info("sample 模式，处理前 %d 行", len(df))

    df = df.reset_index(drop=True)

    # 动态设置模块级默认值
    globals()["DEFAULT_MAX_TEXT_LENGTH"] = args.max_text_len

    n = len(df)
    results = [None] * n

    logger.info("启动线程池：%d 并发", args.concurrency)
    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = {}
        for i, row in df.iterrows():
            fut = exe.submit(
                worker,
                i,
                row,
                args.field,
                args.use_openai_client,
                args.api_key,
                args.base_url,
                args.model,
                args.timeout,
                args.max_retries,
                args.sleep_duration,
                headers,
                args.prompt_key,
                prompt_map,
                args.prompt_extra,
                extra_body
            )
            futures[fut] = i

        with tqdm(total=len(futures), desc="Processing", unit="row") as pbar:
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    r = fut.result()
                    if r is None:
                        results[idx] = None
                    else:
                        ri, val = r
                        results[ri] = val
                except Exception as e:
                    logger.exception("future 异常 %s", e)
                    results[idx] = None
                finally:
                    pbar.update(1)

    logger.info("将结果写回 sglang_result 字段并保存到：%s", args.output)
    df["sglang_result"] = results
    df.to_parquet(args.output, engine="pyarrow", index=False)
    logger.info("完成，输出文件：%s", args.output)


if __name__ == "__main__":
    main()