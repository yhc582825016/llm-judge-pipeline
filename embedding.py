#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_embeddings_shards_runbydate_parquet_minimal_tqdm.py (vllm embed, 支持 tensor_parallel_size)

兼容说明:
- 支持两类常见输入 record:
  1) {"conversations":[{"role":"user","content":...}, ...], ...}
  2) {"instruction": "...", "input": "...", "output": "...", ...}
- 优先提取 conversations 中最后一条 user content；否则按 instruction -> input -> prompt 抽取文本。
- 支持从每条 record 提取 per-row source 字段(--source-field)，若无则使用全局 --source。
- 新增：--concat-user-turns 开关，打开时会把所有 role="user" 的内容按【turn1】...格式拼接后统一做 embedding。
"""
from __future__ import annotations

import os
import json
import argparse
import traceback
from typing import List, Tuple, Optional, Union
import datetime
import tempfile
import uuid
import time
import math
import gc
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

# pyarrow for parquet
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm  # progress bars

# Try import vllm (preferred). If not available, we'll fallback to sentence_transformers (if installed).
try:
    from vllm import LLM
    _HAS_VLLM = True
except Exception:
    LLM = None
    _HAS_VLLM = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except Exception:
    SentenceTransformer = None
    _HAS_SBERT = False

# -------------------- Helper utilities --------------------
def print_cuda_mem(tag: str):
    try:
        if torch.cuda.is_available():
            print(f"[{tag}] allocated={torch.cuda.memory_allocated():,} reserved={torch.cuda.memory_reserved():,}")
        else:
            print(f"[{tag}] cpu mode")
    except Exception:
        print(f"[{tag}] torch 查询显存失败")

def _get_char_truncated_texts(texts: List[str], max_tokens: int = 2048):
    # 保守的字符截断，1 token 约 4 chars 的经验比例
    max_chars = max_tokens * 4
    return [t if len(t) <= max_chars else t[:max_chars] for t in texts]

# -------------------- vllm embedding utility --------------------
def compute_embeddings_vllm(
    texts: List[str],
    vllm_model,
    batch_size: int = 64,
    max_input_tokens: int = 2048,
    verbose: bool = True,
    sync_after_batch: bool = False,
):
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    texts = _get_char_truncated_texts(texts, max_tokens=max_input_tokens)
    embs_list = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    batch_starts = list(range(0, len(texts), batch_size))

    for bidx, start in enumerate(tqdm(batch_starts, total=n_batches, desc="vllm embed batches", unit="batch")):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        if verbose:
            print(f"[vllm] Batch {bidx+1}/{n_batches}: items {start}:{end}")
        print_cuda_mem(f"before vllm embed batch {bidx+1}")

        try:
            outputs = vllm_model.embed(batch)
            batch_embs = []
            for o in outputs:
                emb_t = None
                if hasattr(o, "outputs") and hasattr(o.outputs, "embedding"):
                    emb_t = o.outputs.embedding
                elif hasattr(o, "embedding"):
                    emb_t = o.embedding
                else:
                    try:
                        emb_t = o["outputs"]["embedding"]
                    except Exception:
                        emb_t = None
                if emb_t is None:
                    raise RuntimeError("无法从 vllm 输出对象中提取 embedding (字段路径未知)，请打印输出结构检查。")
                if isinstance(emb_t, torch.Tensor):
                    emb_np = emb_t.detach().cpu().numpy()
                else:
                    emb_np = np.asarray(emb_t, dtype=np.float32)
                if emb_np.ndim == 1:
                    emb_np = emb_np.reshape(1, -1)
                batch_embs.append(emb_np)
            if len(batch_embs) == 0:
                continue
            batch_arr = np.vstack(batch_embs)
            embs_list.append(batch_arr)
        except Exception as e:
            print(f"[vllm] batch {bidx+1} embed 失败: {e}\n尝试逐条回退")
            for i, txt in enumerate(batch):
                try:
                    o_iter = vllm_model.embed([txt])
                    o = next(iter(o_iter))
                    emb_t = None
                    if hasattr(o, "outputs") and hasattr(o.outputs, "embedding"):
                        emb_t = o.outputs.embedding
                    elif hasattr(o, "embedding"):
                        emb_t = o.embedding
                    else:
                        try:
                            emb_t = o["outputs"]["embedding"]
                        except Exception:
                            emb_t = None
                    if emb_t is None:
                        raise RuntimeError("无法从 vllm 输出对象中提取 embedding (单条回退)。")
                    if isinstance(emb_t, torch.Tensor):
                        emb_np = emb_t.detach().cpu().numpy()
                    else:
                        emb_np = np.asarray(emb_t, dtype=np.float32)
                    if emb_np.ndim == 1:
                        emb_np = emb_np.reshape(1, -1)
                    embs_list.append(emb_np)
                except Exception as e2:
                    print(f"[vllm] 回退单条失败 idx={start+i}: {e2}")
                    dim = embs_list[0].shape[1] if embs_list else 0
                    embs_list.append(np.zeros((1, dim), dtype=np.float32))

        print_cuda_mem(f"after vllm embed batch {bidx+1}")
        if sync_after_batch and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    if not embs_list:
        return np.zeros((len(texts), 0), dtype=np.float32)
    embs = np.vstack(embs_list)
    if embs.shape[0] > len(texts):
        embs = embs[: len(texts), :]
    return embs.astype(np.float32)

# -------------------- SentenceTransformer fallback --------------------
def compute_embeddings_sbert(
    texts: List[str],
    model: "SentenceTransformer",
    batch_size: int = 64,
    max_input_tokens: int = 2048,
    verbose: bool = True,
    sync_after_batch: bool = False,
):
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    texts = _get_char_truncated_texts(texts, max_tokens=max_input_tokens)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(embs, dtype=np.float32)

# -------------------- IO / data utils (增强兼容性) --------------------
def items_from_data_last_user(
    data: List[dict],
    source_field: Optional[str] = None,
    concat_user_turns: bool = False,
) -> List[Tuple[int, str, dict, Optional[str]]]:
    """
    返回 list of tuples: (idx, text, orig_dict, per_item_source)
    兼容多种输入格式：
    - conversations -> last user content（默认）
    - instruction / input / prompt
    per_item_source: 如果 source_field 给定并且 record 中存在该字段，则返回对应值（字符串），否则为 None。
    当 concat_user_turns=True 时，会将 conversations 中所有 role="user" 的内容按顺序拼接：
    【turn1】xxx\n【turn2】yyy\n...
    """
    items = []
    for i, rec in enumerate(data):
        # determine per-item source if available
        per_source = None
        if source_field and isinstance(rec, dict) and source_field in rec:
            try:
                per_source = str(rec.get(source_field, "")).strip()
            except Exception:
                per_source = None
        elif isinstance(rec, dict) and "source" in rec:
            # fallback common 'source' key
            try:
                per_source = str(rec.get("source", "")).strip()
            except Exception:
                per_source = None

        text = None
        # 1) conversations format
        if isinstance(rec, dict) and "conversations" in rec and isinstance(rec["conversations"], list):
            convs = rec.get("conversations", [])
            user_msgs = [c.get("content") for c in convs
                         if isinstance(c, dict) and c.get("role") == "user" and c.get("content")]
            if user_msgs:
                if concat_user_turns:
                    # 拼接所有 user 轮次
                    lines = []
                    for t_idx, msg in enumerate(user_msgs, start=1):
                        try:
                            msg_str = msg if isinstance(msg, str) else json.dumps(msg, ensure_ascii=False)
                        except Exception:
                            msg_str = str(msg)
                        lines.append(f"【turn{t_idx}】{msg_str}")
                    text = "\n".join(lines)
                else:
                    # 只取最后一条 user
                    text = user_msgs[-1]
        ## messages format
        if isinstance(rec, dict) and "messages" in rec and isinstance(rec["messages"], list):
            convs = rec.get("messages", [])
            user_msgs = [c.get("content") for c in convs
                         if isinstance(c, dict) and c.get("role") == "user" and c.get("content")]
            if user_msgs:
                if concat_user_turns:
                    # 拼接所有 user 轮次
                    lines = []
                    for t_idx, msg in enumerate(user_msgs, start=1):
                        try:
                            msg_str = msg if isinstance(msg, str) else json.dumps(msg, ensure_ascii=False)
                        except Exception:
                            msg_str = str(msg)
                        lines.append(f"【turn{t_idx}】{msg_str}")
                    text = "\n".join(lines)
                else:
                    # 只取最后一条 user
                    text = user_msgs[-1]
        
        # 2) instruction / input / prompt style
        if text is None and isinstance(rec, dict):
            # prefer 'instruction'
            for k in ("instruction", "input", "prompt"):
                if k in rec and rec.get(k) not in (None, ""):
                    cand = rec.get(k)
                    # some datasets put instruction as dict or list; coerce to str
                    if isinstance(cand, (list, dict)):
                        try:
                            cand = json.dumps(cand, ensure_ascii=False)
                        except Exception:
                            cand = str(cand)
                    text = str(cand)
                    break
        # 3) if text still None and record has top-level 'text' or 'query'
        if text is None and isinstance(rec, dict):
            for k in ("text", "query", "prompt_text"):
                if k in rec and rec.get(k) not in (None, ""):
                    text = str(rec.get(k))
                    break

        if text is None:
            # skip records that we cannot extract text from
            continue

        # ensure orig is the full original dict (for writing back)
        orig = rec
        items.append((i, text, orig, per_source))
    return items

def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def partition_items_with_idx(idxs_texts, n_shards):
    shards = [[] for _ in range(n_shards)]
    for i, it in enumerate(idxs_texts):
        shards[i % n_shards].append(it)
    return shards

# -------------------- atomic helpers --------------------
def _next_available_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    base_dir = os.path.dirname(base_path) or "."
    base_name = os.path.basename(base_path)
    name, ext = os.path.splitext(base_name)
    i = 1
    while True:
        candidate = os.path.join(base_dir, f"{name}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1

# -------------------- Parquet save helper (支持 per-row source list 或 单一 source) --------------------
def save_parquet_shard_minimal(
    out_dir: str,
    shard_id: int,
    uids: List[str],
    idxs: List[int],
    embeddings: np.ndarray,   # shape (N, D), dtype float32
    origs: Optional[List[dict]] = None,
    source: Optional[Union[str, List[str]]] = None,
    compression: str = "zstd",
    row_group_size: int = 65536,
):
    os.makedirs(out_dir, exist_ok=True)
    base_name = f"shard_{shard_id}.parquet"
    out_path = os.path.join(out_dir, base_name)
    out_path = _next_available_path(out_path)

    N = len(uids)
    # build schema with source field
    if N == 0:
        schema = pa.schema([
            pa.field("uid", pa.string()),
            pa.field("idx", pa.int64()),
            pa.field("embedding", pa.list_(pa.float32())),
            pa.field("orig", pa.string()),
            pa.field("source", pa.string()),
        ])
        table = pa.Table.from_arrays([
                                     pa.array([], type=pa.string()),
                                     pa.array([], type=pa.int64()),
                                     pa.array([], type=pa.list_(pa.float32())),
                                     pa.array([], type=pa.string()),
                                     pa.array([], type=pa.string()),
                                     ],
                                     schema=schema)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet", prefix="tmp_parquet_", dir=os.path.dirname(out_path) or ".")
        os.close(tmp_fd)
        try:
            pq.write_table(table, tmp_path, compression=compression, row_group_size=row_group_size)
            os.replace(tmp_path, out_path)
            return out_path
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

    emb = np.ascontiguousarray(embeddings.astype(np.float32))
    if emb.ndim != 2:
        raise ValueError("embeddings must be 2D array (N, D)")
    N_e, D = emb.shape
    if N_e != N:
        raise ValueError(f"embeddings rows {N_e} != number of uids {N}")

    try:
        uid_arr = pa.array(uids, type=pa.string())
        idx_arr = pa.array([int(x) for x in idxs], type=pa.int64())
        flat_vals = pa.array(emb.reshape(-1).tolist(), type=pa.float32())
        emb_arr = pa.FixedSizeListArray.from_arrays(flat_vals, D)

        if origs:
            orig_jsons = []
            for o in origs:
                try:
                    orig_jsons.append(json.dumps(o, ensure_ascii=False))
                except Exception:
                    orig_jsons.append(str(o))
        else:
            orig_jsons = [""] * N
        orig_arr = pa.array(orig_jsons, type=pa.string())

        # source can be a single string or list of strings
        if isinstance(source, list):
            if len(source) != N:
                # fallback: pad or truncate
                srcs = [str(s) if s is not None else "" for s in (source + [""] * max(0, N - len(source)))]
                srcs = srcs[:N]
            else:
                srcs = [str(s) if s is not None else "" for s in source]
        else:
            srcs = [str(source) if source is not None else ""] * N
        source_arr = pa.array(srcs, type=pa.string())

        arrays = [uid_arr, idx_arr, emb_arr, orig_arr, source_arr]
        names = ["uid", "idx", "embedding", "orig", "source"]
        table = pa.Table.from_arrays(arrays, names=names)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet", prefix="tmp_parquet_", dir=os.path.dirname(out_path) or ".")
        os.close(tmp_fd)
        pq.write_table(table, tmp_path, compression=compression, row_group_size=row_group_size)
        os.replace(tmp_path, out_path)
        return out_path
    except Exception as e:
        raise

# -------------------- worker --------------------
def worker_compute_save(
    shard_id: int,
    shard_items: List[Tuple],  # (idx, text, orig, maybe_per_source)
    model_path: str,
    gpu_group: str,            # 逗号分隔的物理 GPU id，如 "4,5"
    tp_size: int,              # tensor_parallel_size (应等于 len(gpu_group.split(',')))
    emb_batch_size: int,
    max_input_tokens: int,
    empty_cache_every_n_batches: Optional[int],
    sync_after_batch: bool,
    out_dir: str,
    global_source: Optional[str] = None,
):
    try:
        if gpu_group is not None and gpu_group != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_group)
        print(f"[worker {shard_id}] start, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, tp_size={tp_size}, items={len(shard_items)}, global_source={global_source!r}")

        if len(shard_items) == 0:
            try:
                save_parquet_shard_minimal(out_dir, shard_id, [], [], np.zeros((0, 0), dtype=np.float32), origs=[], source=global_source)
                print(f"[worker {shard_id}] 無 items，已寫空 parquet 並退出")
            except Exception as e:
                print(f"[worker {shard_id}] 寫空 parquet 失敗: {e}")
            return

        # 解析 shard_items（支持 2/3/4 长度元组）
        idxs = []
        texts = []
        origs = []
        per_sources = []
        for it in shard_items:
            if len(it) == 2:
                i, txt = it
                idxs.append(i)
                texts.append(txt)
                origs.append(None)
                per_sources.append(None)
            elif len(it) == 3:
                i, txt, o = it
                idxs.append(i)
                texts.append(txt)
                origs.append(o)
                per_sources.append(None)
            else:
                # (i, txt, o, per_source)
                i, txt, o, ps = it
                idxs.append(i)
                texts.append(txt)
                origs.append(o)
                per_sources.append(ps if ps not in (None, "") else None)

        # uids
        uids = [uuid.uuid4().hex for _ in range(len(texts))]

        emb = None
        # 优先使用 vllm（GPU 路径）
        if _HAS_VLLM and (os.environ.get("CUDA_VISIBLE_DEVICES") is not None and torch.cuda.device_count() > 0):
            visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
            try:
                visible_count = len([g for g in visible_gpus if g.strip() != ""])
            except Exception:
                visible_count = 0
            if visible_count <= 0:
                print(f"[worker {shard_id}] 未检测到可见 GPU（visible_count={visible_count}），跳过 vllm，尝试回退到 SentenceTransformer（若可用）")
            else:
                if tp_size != visible_count:
                    print(f"[worker {shard_id}] WARNING: tp_size ({tp_size}) 与该子进程可见 GPU 数量 ({visible_count}) 不一致；将以可见卡数为准。")
                    tp_used = visible_count
                else:
                    tp_used = tp_size
                try:
                    vllm_model = LLM(model=model_path, task="embed", tensor_parallel_size=tp_used)
                except Exception as e:
                    print(f"[worker {shard_id}] vllm LLM 加载失败: {e}\n{traceback.format_exc()}")
                    vllm_model = None

                if vllm_model is not None:
                    try:
                        emb = compute_embeddings_vllm(
                            texts,
                            vllm_model=vllm_model,
                            batch_size=emb_batch_size,
                            max_input_tokens=max_input_tokens,
                            verbose=True,
                            sync_after_batch=sync_after_batch,
                        )
                    except Exception as e:
                        print(f"[worker {shard_id}] vllm embed 失败: {e}\n{traceback.format_exc()}")
                        emb = None
                    finally:
                        try:
                            vllm_model.close()
                        except Exception:
                            try:
                                vllm_model.shutdown()
                            except Exception:
                                pass

        # fallback to sbert
        if emb is None:
            if _HAS_SBERT:
                try:
                    print(f"[worker {shard_id}] 使用 SentenceTransformer 回退生成 embeddings（可能在 CPU）")
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                        sb_model = SentenceTransformer(model_path) if os.path.exists(model_path) else SentenceTransformer("all-MiniLM-L6-v2")
                    else:
                        sb_model = SentenceTransformer("all-MiniLM-L6-v2")
                    emb = compute_embeddings_sbert(texts, model=sb_model, batch_size=emb_batch_size, max_input_tokens=max_input_tokens)
                except Exception as e:
                    print(f"[worker {shard_id}] SentenceTransformer 回退也失败: {e}\n{traceback.format_exc()}")
                    emb = None
            else:
                print(f"[worker {shard_id}] 没有可用的 embed 后备方案（vllm 未成功且没有安装 sentence-transformers），将尝试生成零向量占位")
                emb = np.zeros((len(texts), 512), dtype=np.float32)

        if emb is None:
            raise RuntimeError(f"[worker {shard_id}] 未能生成 embedding（emb is None）")

        # determine per-row source list: prefer per_sources if not None else global_source
        source_list = []
        for ps in per_sources:
            if ps not in (None, ""):
                source_list.append(ps)
            elif global_source not in (None, ""):
                source_list.append(global_source)
            else:
                source_list.append("")

        # save parquet (传入 per-row source list)
        os.makedirs(out_dir, exist_ok=True)
        try:
            path = save_parquet_shard_minimal(out_dir, shard_id, uids, idxs, emb, origs=origs, source=source_list)
            print(f"[worker {shard_id}] 已保存 parquet 到 {path} （rows={len(texts)})")
        except Exception as e:
            print(f"[worker {shard_id}] 保存 parquet 失败: {e}\n{traceback.format_exc()}")
            # 降级：回退写入
            try:
                emb_path = os.path.join(out_dir, f"shard_{shard_id}_fallback.npy")
                uids_path = os.path.join(out_dir, f"shard_{shard_id}_fallback_uids.npy")
                meta_path = os.path.join(out_dir, f"shard_{shard_id}_fallback_meta.json")
                np.save(emb_path, emb)
                np.save(uids_path, np.array(uids, dtype=object))
                meta_records = []
                for uid, i, txt, o, s in zip(uids, idxs, texts, origs, source_list):
                    meta_records.append({"uid": uid, "idx": int(i), "text": txt, "orig": o, "source": s})
                with open(meta_path, "w", encoding="utf-8") as wf:
                    json.dump(meta_records, wf, ensure_ascii=False, indent=2)
                print(f"[worker {shard_id}] 回退寫入到 {emb_path}, {uids_path}, {meta_path}")
            except Exception as e2:
                print(f"[worker {shard_id}] 回退也失败: {e2}")

        # 清理
        try:
            del emb
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    except Exception as e:
        print(f"[worker {shard_id}] 出错: {e}\n{traceback.format_exc()}")

# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入 jsonl 文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="vllm 模型名或本地路径（如 Qwen/Qwen3-Embedding-8B）")
    parser.add_argument("--out_dir", type=str, default="./emb_parquet_shards", help="保存 parquet 的目录")
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--empty_cache_every_n_batches", type=int, default=10)
    parser.add_argument("--sync_after_batch", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="每个 vllm 实例使用多少张 GPU 做 tensor parallelism（>=1）")
    parser.add_argument("--source", type=str, default="", help="写入 parquet 的全局 source 字段（每行相同，若不指定可通过 --source-field 从 record 中抽取）")
    parser.add_argument("--source-field", type=str, default="", help="（可选）从每条 record 中提取 per-row source 的字段名，例如 'source' 或 'split'。若指定，优先使用记录内的该字段值作为每行 source")
    parser.add_argument("--concat-user-turns", action="store_true",
                        help="当输入为 conversations 时，将所有 role='user' 的内容按【turn1】...格式拼接后再做单条 embedding")
    args = parser.parse_args()

    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_env:
        physical_gpus = [g.strip() for g in cuda_env.split(",") if g.strip() != ""]
    else:
        n = torch.cuda.device_count()
        physical_gpus = [str(i) for i in range(n)]

    total_gpus = len(physical_gpus)
    tp = int(args.tensor_parallel_size)
    if tp < 1:
        raise ValueError("--tensor_parallel_size must be >= 1")

    if total_gpus == 0:
        print("未检测到 GPU。脚本将尝试在 CPU 上或使用 sentence-transformers 回退（若已安装）。")
        n_shards = 1
        gpu_groups = [""]
    else:
        if total_gpus % tp != 0:
            raise ValueError(f"total_gpus ({total_gpus}) 必须能被 tensor_parallel_size ({tp}) 整除")
        n_shards = total_gpus // tp
        gpu_groups = []
        for i in range(n_shards):
            group = physical_gpus[i*tp:(i+1)*tp]
            gpu_groups.append(",".join(group))
    if args.input.endswith('json') or args.input.endswith('jsonl'):
        data = load_jsonl(args.input)
    else:
        print('load parquet data')
        data = pd.read_parquet(args.input)
        data = json.loads(data.to_json(orient="records", force_ascii=False))
    print('len data',len(data))
    items = items_from_data_last_user(
        data,
        source_field=args.source_field if args.source_field else None,
        concat_user_turns=args.concat_user_turns,
    )
    if not items:
        raise ValueError("数据中没有找到可提取的文本 (conversations/instruction/input/prompt)")
    print(f"共提取 {len(items)} 条文本")

    print("Physical GPUs:", physical_gpus)
    print("GPU groups for workers:", gpu_groups)
    print("n_shards (workers):", n_shards)
    print("global source:", args.source, "source_field:", args.source_field)

    # 分 shard（按 worker 数量）
    shards = partition_items_with_idx(items, n_shards)
    print("每 shard 文本数:", [len(s) for s in shards])

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    procs = []
    for sid in range(n_shards):
        p = mp.Process(
            target=worker_compute_save,
            args=(
                sid,
                shards[sid],
                args.model_path,
                gpu_groups[sid],
                tp,
                args.embedding_batch_size,
                args.max_input_tokens,
                args.empty_cache_every_n_batches,
                args.sync_after_batch,
                args.out_dir,
                args.source,  # global fallback
            ),
        )
        p.start()
        procs.append(p)
        print(f"启动 worker shard {sid} pid={p.pid} gpu_group={gpu_groups[sid]} global_source={args.source!r}")

    # 等待 shards 完成并显示 tqdm
    with tqdm(total=len(procs), desc="waiting shards", unit="shard") as pj:
        for p in procs:
            p.join()
            print(f"worker pid={p.pid} 结束，exitcode={p.exitcode}")
            pj.update(1)

if __name__ == "__main__":
    main()
