# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# faisst_filter_v2_torch_full_parquet.py
# 说明：
# - 支持 json/.npy 对、以及 Parquet 输入（单文件或目录）
# - 不会修改原始 parquet 文件；写出时仅去掉重复的行并保留所有原始列
# - kept / removed 会以 parquet 格式写出（embedding 列以 list 存储）
# - self_dedup / pair 两种模式（torch topk；无 GPU 回退 numpy）
# """
# import argparse
# import json
# import os
# import sys
# import glob
# import csv
# import ast
# from typing import List, Dict, Any, Tuple, Optional

# import numpy as np
# import torch
# from tqdm import tqdm

# try:
#     import pandas as pd
# except Exception as e:
#     print("[ERROR] pandas required. pip install pandas pyarrow", file=sys.stderr)
#     raise

# # ---------------------------
# # parse embedding helper
# # ---------------------------
# def parse_embedding_field(val) -> Optional[np.ndarray]:
#     if val is None:
#         return None
#     if isinstance(val, np.ndarray):
#         return val.astype(np.float32)
#     if isinstance(val, (list, tuple)):
#         try:
#             return np.asarray(val, dtype=np.float32)
#         except Exception:
#             return None
#     if isinstance(val, str):
#         s = val.strip()
#         if s == "":
#             return None
#         try:
#             arr = json.loads(s)
#             if isinstance(arr, (list, tuple)):
#                 return np.asarray(arr, dtype=np.float32)
#         except Exception:
#             try:
#                 arr = ast.literal_eval(s)
#                 if isinstance(arr, (list, tuple)):
#                     return np.asarray(arr, dtype=np.float32)
#             except Exception:
#                 return None
#     return None

# # ---------------------------
# # resolve parquet paths helper
# # ---------------------------
# def resolve_parquet_paths(path_or_dir: str) -> List[str]:
#     """返回匹配的 parquet 文件的完整路径（按字母序）"""
#     if path_or_dir is None:
#         return []
#     if os.path.isfile(path_or_dir) and path_or_dir.lower().endswith(".parquet"):
#         return [os.path.abspath(path_or_dir)]
#     if os.path.isdir(path_or_dir):
#         ps = sorted([os.path.join(path_or_dir, p) for p in os.listdir(path_or_dir) if p.lower().endswith(".parquet")])
#         return [os.path.abspath(p) for p in ps]
#     # glob pattern
#     ps = sorted(glob.glob(path_or_dir))
#     ps = [p for p in ps if p.lower().endswith(".parquet")]
#     return [os.path.abspath(p) for p in ps]

# # ---------------------------
# # parquet loader -> records (包含 emb numpy)
# # ---------------------------
# def load_parquet_files_to_records(path_or_dir: str) -> List[Dict[str, Any]]:
#     paths = resolve_parquet_paths(path_or_dir)
#     records: List[Dict[str, Any]] = []
#     if not paths:
#         return records

#     for p in paths:
#         try:
#             df = pd.read_parquet(p)
#         except Exception as e:
#             print(f"[WARN] failed to read parquet {p}: {e}", file=sys.stderr)
#             continue

#         cols = df.columns.tolist()
#         idx_col = None
#         for c in ("idx", "global_idx", "id", "index"):
#             if c in cols:
#                 idx_col = c
#                 break

#         for i, row in df.iterrows():
#             try:
#                 emb = None
#                 if "embedding" in cols:
#                     emb = parse_embedding_field(row["embedding"])
#                 for alt in ("emb", "vector", "embedding_vector"):
#                     if emb is None and alt in cols:
#                         emb = parse_embedding_field(row[alt])
#                         if emb is not None:
#                             break
#                 if emb is None:
#                     # skip if no embedding
#                     continue
#                 if idx_col is not None:
#                     try:
#                         gi = int(row[idx_col])
#                     except Exception:
#                         gi = None
#                 else:
#                     gi = None

#                 # local_pos: 尝试以行标签（index）保存为 int（若 index 为 int）
#                 local_pos = None
#                 if isinstance(i, (int, np.integer)):
#                     local_pos = int(i)

#                 rec: Dict[str, Any] = {
#                     "global_idx": int(gi) if gi is not None else None,
#                     "emb": emb.astype(np.float32),
#                     "text": row["text"] if "text" in cols and not pd.isna(row["text"]) else "",
#                     "src_parquet": os.path.basename(p),
#                     "src_parquet_full": os.path.abspath(p),
#                     "local_pos": local_pos,
#                 }
#                 # carry optional fields
#                 for opt in ("uid", "model_name", "run_name", "run_date", "timestamp", "score", "label", "orig_input_json", "kmeans_label"):
#                     if opt in cols:
#                         val = row[opt]
#                         try:
#                             json.dumps(val)
#                             rec[opt] = val
#                         except Exception:
#                             rec[opt] = str(val)
#                 records.append(rec)
#             except Exception as e:
#                 print(f"[WARN] skip row {i} in {p}: {e}", file=sys.stderr)
#                 continue
#     records.sort(key=lambda r: (r.get("src_parquet", ""), r.get("local_pos", 0) if r.get("local_pos") is not None else 0))
#     return records

# # ---------------------------
# # original json/.npy loader (fallback)
# # ---------------------------
# def load_emb_dir_to_records(emb_dir: str) -> List[Dict[str, Any]]:
#     # 如果是 parquet 路径，直接走 parquet loader（以确保 src_parquet_full 保存）
#     if emb_dir is None:
#         return []
#     parquet_paths = resolve_parquet_paths(emb_dir)
#     if parquet_paths:
#         return load_parquet_files_to_records(emb_dir)
#     if not os.path.isdir(emb_dir):
#         raise RuntimeError(f"emb_dir not found: {emb_dir}")
#     records: List[Dict[str, Any]] = []
#     json_files = sorted([f for f in os.listdir(emb_dir) if f.lower().endswith(".json")])
#     for json_name in json_files:
#         json_path = os.path.join(emb_dir, json_name)
#         base = os.path.splitext(json_name)[0]
#         emb_path = os.path.join(emb_dir, base + ".npy")
#         if not os.path.exists(emb_path):
#             print(f"[WARN] skip {json_path}, missing {emb_path}", file=sys.stderr)
#             continue
#         try:
#             with open(json_path, "r", encoding="utf-8") as rf:
#                 meta = json.load(rf)
#         except Exception as e:
#             print(f"[WARN] failed to load meta {json_path}: {e}", file=sys.stderr)
#             continue

#         idxs = meta.get("idx", [])
#         texts = meta.get("texts", [])
#         inputs = meta.get("inputs", None)
#         try:
#             emb = np.load(emb_path, allow_pickle=False)
#         except Exception as e:
#             print(f"[WARN] failed to load emb {emb_path}: {e}", file=sys.stderr)
#             continue

#         if emb.ndim == 1:
#             emb = emb.reshape(1, -1)
#         n_meta = min(len(idxs), len(texts))
#         n_emb = emb.shape[0]
#         if n_emb == 0 or n_meta == 0:
#             continue
#         n_use = min(n_meta, n_emb)
#         for i_local in range(n_use):
#             try:
#                 gi = int(idxs[i_local])
#             except Exception:
#                 gi = None
#             emb_row = emb[i_local].astype(np.float32)
#             txt = texts[i_local] if i_local < len(texts) else ""
#             orig_input = None
#             if inputs is not None and isinstance(inputs, (list, tuple)) and i_local < len(inputs):
#                 orig_input = inputs[i_local]
#             records.append({
#                 "global_idx": gi,
#                 "emb": emb_row,
#                 "text": txt,
#                 "inputs": orig_input,
#                 "src_json": json_name,
#                 "local_pos": i_local
#             })
#     records.sort(key=lambda r: (r.get("src_json", ""), r.get("local_pos", 0)))
#     return records

# # ---------------------------
# # normalize
# # ---------------------------
# def _ensure_normalized(arr: np.ndarray) -> np.ndarray:
#     if arr.size == 0:
#         return arr
#     norms = np.linalg.norm(arr, axis=1, keepdims=True)
#     norms[norms == 0] = 1.0
#     return (arr / norms).astype(np.float32)

# # ---------------------------
# # build arrays & keep emb in orig_records (so kept can be written back)
# # ---------------------------
# def build_array_from_emb_dir(emb_dir: str) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
#     records = load_emb_dir_to_records(emb_dir)
#     if not records:
#         return np.zeros((0,0), dtype=np.float32), [], []
#     X = np.vstack([r["emb"] for r in records]).astype(np.float32)
#     texts = [r.get("text", "") for r in records]
#     orig_records = []
#     for r in records:
#         rec = {}
#         # preserve many useful fields and include emb (as numpy)
#         for k in ("uid", "global_idx", "idx", "text", "src_parquet", "src_parquet_full", "src_json", "local_pos",
#                   "model_name", "run_name", "run_date", "timestamp", "score", "label", "orig_input_json", "kmeans_label", "inputs"):
#             if k in r:
#                 rec[k] = r[k]
#         # include raw embedding as numpy (keep as np for dedup), will convert to list when writing parquet if needed
#         rec["emb"] = r["emb"]
#         orig_records.append(rec)
#     return X, texts, orig_records

# # ---------------------------
# # topk torch/numpy (unchanged)
# # ---------------------------
# def topk_sim_torch(Xq: np.ndarray, Xdb: np.ndarray, topk: int = 1, device: str = "cuda", chunk_db: int = 20000, chunk_q: int = 256) -> Tuple[np.ndarray, np.ndarray]:
#     assert Xq.dtype == np.float32 and Xdb.dtype == np.float32
#     nq, d = Xq.shape
#     nd = Xdb.shape[0]
#     k_eff = min(topk, nd) if nd > 0 else 0
#     if k_eff <= 0:
#         return np.zeros((nq, 0), dtype=np.float32), np.zeros((nq, 0), dtype=np.int64)

#     if device == "cpu" or not torch.cuda.is_available():
#         sims_full = np.matmul(Xq, Xdb.T)
#         idxs_part = np.argpartition(-sims_full, range(k_eff), axis=1)[:, :k_eff]
#         topk_sims = np.take_along_axis(sims_full, idxs_part, axis=1)
#         order = np.argsort(-topk_sims, axis=1)
#         topk_sims = np.take_along_axis(topk_sims, order, axis=1)
#         idxs_sorted = np.take_along_axis(idxs_part, order, axis=1)
#         return topk_sims.astype(np.float32), idxs_sorted.astype(np.int64)

#     dev = torch.device(device)
#     Xdb_t = torch.from_numpy(Xdb).to(dev)
#     result_sims = np.zeros((nq, k_eff), dtype=np.float32)
#     result_idxs = np.zeros((nq, k_eff), dtype=np.int64)
#     with torch.no_grad():
#         for qstart in range(0, nq, chunk_q):
#             qend = min(nq, qstart + chunk_q)
#             Xq_chunk = torch.from_numpy(Xq[qstart:qend]).to(dev)

#             topk_vals = None
#             topk_idxs = None

#             for dbstart in range(0, nd, chunk_db):
#                 dbend = min(nd, dbstart + chunk_db)
#                 Xdb_chunk = Xdb_t[dbstart:dbend]
#                 sims_chunk = torch.matmul(Xq_chunk, Xdb_chunk.t())
#                 k_cur = min(k_eff, sims_chunk.size(1))
#                 vals, idxs = torch.topk(sims_chunk, k_cur, dim=1)
#                 idxs = idxs + dbstart

#                 if topk_vals is None:
#                     topk_vals = vals
#                     topk_idxs = idxs
#                 else:
#                     cat_vals = torch.cat([topk_vals, vals], dim=1)
#                     cat_idxs = torch.cat([topk_idxs, idxs], dim=1)
#                     k_choose = min(k_eff, cat_vals.size(1))
#                     vals2, pos = torch.topk(cat_vals, k_choose, dim=1)
#                     idxs2 = torch.gather(cat_idxs, 1, pos)
#                     topk_vals = vals2
#                     topk_idxs = idxs2

#                 del sims_chunk, vals, idxs
#                 torch.cuda.empty_cache()

#             result_sims[qstart:qend] = topk_vals.cpu().numpy().astype(np.float32)
#             result_idxs[qstart:qend] = topk_idxs.cpu().numpy().astype(np.int64)
#             del Xq_chunk, topk_vals, topk_idxs
#             torch.cuda.empty_cache()
#     return result_sims, result_idxs

# # ---------------------------
# # self_dedup (preserve first occurrence)
# # ---------------------------
# def self_dedup_records(records: List[Dict[str, Any]], threshold: float, topk: int = 2, device: str = "cuda", chunk_db: int = 20000, chunk_q: int = 256) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
#     N = len(records)
#     if N == 0:
#         return [], []

#     X = np.vstack([r["emb"] for r in records]).astype(np.float32)
#     X = np.ascontiguousarray(_ensure_normalized(X), dtype=np.float32)

#     k_eff = min(max(2, int(topk)), N)
#     sims, idxs = topk_sim_torch(X, X, topk=k_eff, device=device, chunk_db=chunk_db, chunk_q=chunk_q)

#     keep_flags = np.ones((N,), dtype=bool)
#     for i in range(N):
#         if not keep_flags[i]:
#             continue
#         neighbors = idxs[i].tolist()
#         sims_row = sims[i].tolist()
#         for nb_pos, nb in enumerate(neighbors):
#             if nb == i:
#                 continue
#             nb_sim = sims_row[nb_pos]
#             if nb_sim >= threshold:
#                 if nb < i:
#                     keep_flags[i] = False
#                 else:
#                     keep_flags[nb] = False
#                 break

#     kept = [records[i] for i in range(N) if keep_flags[i]]
#     removed = []
#     for i in range(N):
#         if not keep_flags[i]:
#             neighbors = idxs[i].tolist()
#             sims_row = sims[i].tolist()
#             cause = None
#             for nb_pos, nb in enumerate(neighbors):
#                 if nb == i:
#                     continue
#                 if keep_flags[nb] and sims_row[nb_pos] >= threshold:
#                     cause = {"best_sim": float(sims_row[nb_pos]), "best_idx": int(nb)}
#                     break
#             removed_rec = {
#                 "global_idx": int(records[i].get("global_idx")) if records[i].get("global_idx") is not None else None,
#                 "text": records[i].get("text", ""),
#                 "inputs": records[i].get("inputs"),
#                 "src_parquet": records[i].get("src_parquet"),
#                 "src_parquet_full": records[i].get("src_parquet_full"),
#                 "local_pos": records[i].get("local_pos"),
#                 "reason": cause,
#             }
#             for opt in ("uid", "model_name", "run_name", "run_date", "timestamp", "score", "label", "orig_input_json", "kmeans_label"):
#                 if opt in records[i]:
#                     removed_rec[opt] = records[i][opt]
#             removed.append(removed_rec)
#     return kept, removed

# # ---------------------------
# # pair mode
# # ---------------------------
# def filter_pair_emb_dirs_torch(train_emb_dir: str, test_emb_dir: str, threshold: float, topk: int = 1, device: str = "cuda", chunk_db: int = 20000, chunk_q: int = 256):
#     X_train, train_texts, train_recs = build_array_from_emb_dir(train_emb_dir)
#     X_test, test_texts, test_recs = build_array_from_emb_dir(test_emb_dir)

#     if X_train.size == 0:
#         return [], []
#     if X_test.size == 0:
#         kept = train_recs.copy()
#         removed = []
#         return kept, removed

#     X_train = np.ascontiguousarray(_ensure_normalized(X_train), dtype=np.float32)
#     X_test = np.ascontiguousarray(_ensure_normalized(X_test), dtype=np.float32)

#     nd = X_test.shape[0]
#     k_eff = min(int(topk), nd) if nd > 0 else 0

#     sims, idxs = topk_sim_torch(X_train, X_test, topk=k_eff, device=device, chunk_db=chunk_db, chunk_q=chunk_q)

#     best_arg = np.argmax(sims, axis=1)
#     best_sims = sims[np.arange(sims.shape[0]), best_arg]
#     best_idxs = idxs[np.arange(idxs.shape[0]), best_arg]
#     keep_mask = best_sims < threshold

#     kept = [train_recs[i] for i, keep in enumerate(keep_mask) if keep]
#     removed = []
#     for i, (keep, sim_row, idx_row, best_sim, best_idx) in enumerate(zip(keep_mask, sims, idxs, best_sims, best_idxs)):
#         if keep:
#             continue
#         rec = train_recs[i]
#         removed.append({
#             "train_pos": i,
#             "record": rec,
#             "train_text": train_texts[i] if i < len(train_texts) else "",
#             "topk_sims": [float(x) for x in sim_row.tolist()],
#             "topk_test_indices": [int(x) for x in idx_row.tolist()],
#             "best_sim": float(best_sim),
#             "best_test_index": int(best_idx),
#             "best_test_user": (test_texts[int(best_idx)] if 0 <= int(best_idx) < len(test_texts) else None)
#         })
#     return kept, removed

# # ---------------------------
# # records -> dataframe fallback (for json/npy sources)
# # ---------------------------
# def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
#     if not records:
#         return pd.DataFrame()
#     rows = []
#     # unify keys
#     all_keys = set()
#     for r in records:
#         all_keys.update(r.keys())
#     # construct rows, convert emb numpy -> list for parquet
#     for r in records:
#         row = {}
#         for k in all_keys:
#             v = r.get(k, None)
#             if k == "emb" and v is not None:
#                 # convert numpy to python list (float)
#                 try:
#                     row["embedding"] = (v.tolist() if isinstance(v, np.ndarray) else list(v))
#                 except Exception:
#                     row["embedding"] = str(v)
#             else:
#                 row[k] = v
#         rows.append(row)
#     df = pd.DataFrame.from_records(rows)
#     return df

# # ---------------------------
# # write parquet while preserving original columns when possible
# # ---------------------------
# def write_kept_removed_preserve_parquet(parquet_input_paths: List[str],
#                                         kept_records: List[Dict[str, Any]],
#                                         removed_records: List[Dict[str, Any]],
#                                         out_kept_path: str,
#                                         out_removed_path: Optional[str] = None):
#     """
#     对于来自 parquet 的记录，尽量从原始 parquet 读取并基于 local_pos/index 精确筛选行并写出（保持所有列）。
#     对于来自 json/npy（没有源 parquet）的记录，作为 fallback，用 records_to_dataframe 写出。
#     """
#     # build mapping basename -> full path
#     basename_to_full = {os.path.basename(p): p for p in parquet_input_paths}

#     def group_by_src(records: List[Dict[str, Any]]):
#         by_src = {}
#         fallback = []
#         for r in records:
#             src = r.get("src_parquet")
#             src_full = r.get("src_parquet_full")
#             local_pos = r.get("local_pos")
#             if src is None and src_full is None:
#                 # json/npy fallback
#                 fallback.append(r)
#                 continue
#             # prefer full path if available
#             if src_full:
#                 key_full = os.path.abspath(src_full)
#                 key_basename = os.path.basename(src_full)
#             else:
#                 key_basename = src
#                 key_full = basename_to_full.get(key_basename)
#             if key_full is None:
#                 # cannot locate original parquet -> fallback
#                 fallback.append(r)
#                 continue
#             s = by_src.get(key_full)
#             if s is None:
#                 s = set()
#                 by_src[key_full] = s
#             if local_pos is not None:
#                 s.add(local_pos)
#             else:
#                 # if no local_pos, fallback to marking whole file? safer: fallback record
#                 fallback.append(r)
#         return by_src, fallback

#     kept_by_src, kept_fallback = group_by_src(kept_records)
#     removed_by_src, removed_fallback = group_by_src(removed_records)

#     # write kept: iterate parquet_input_paths in order, filter rows belonging to kept_by_src
#     kept_dfs = []
#     for p in parquet_input_paths:
#         p_abs = os.path.abspath(p)
#         keep_set = kept_by_src.get(p_abs)
#         if keep_set is None:
#             continue
#         try:
#             df = pd.read_parquet(p_abs)
#         except Exception as e:
#             print(f"[WARN] failed to read parquet {p_abs} while writing kept: {e}", file=sys.stderr)
#             continue
#         if df.shape[0] == 0:
#             continue
#         # build boolean mask by matching index labels or by position
#         mask = [False] * len(df)
#         idx_index = df.index
#         for pos in keep_set:
#             try:
#                 # 优先按 index label 匹配
#                 loc = idx_index.get_indexer([pos])[0]
#                 if loc != -1:
#                     mask[loc] = True
#                     continue
#             except Exception:
#                 pass
#             # fallback 按位置匹配（若 pos 在合法范围）
#             try:
#                 if 0 <= int(pos) < len(df):
#                     mask[int(pos)] = True
#             except Exception:
#                 pass
#         df_keep = df[pd.Series(mask).values]
#         if not df_keep.empty:
#             kept_dfs.append(df_keep)
#     # append fallback kept records (json/npy or unmatched)
#     if kept_fallback:
#         df_fb = records_to_dataframe(kept_fallback)
#         if not df_fb.empty:
#             kept_dfs.append(df_fb)

#     if kept_dfs:
#         df_all_kept = pd.concat(kept_dfs, ignore_index=True, axis=0)
#     else:
#         df_all_kept = pd.DataFrame()
#     # write kept parquet
#     try:
#         if not df_all_kept.empty:
#             df_all_kept.to_parquet(out_kept_path, index=False)
#         else:
#             # write empty parquet (so downstream tools can read)
#             pd.DataFrame().to_parquet(out_kept_path, index=False)
#         print(f"[INFO] parquet written to {out_kept_path}", file=sys.stderr)
#     except Exception as e:
#         print(f"[WARN] parquet write failed ({e}), trying csv fallback for kept", file=sys.stderr)
#         try:
#             df_all_kept.to_csv(out_kept_path + ".csv", index=False)
#             print(f"[INFO] csv fallback written to {out_kept_path}.csv", file=sys.stderr)
#         except Exception as e2:
#             print(f"[ERROR] fallback write failed for kept: {e2}", file=sys.stderr)

#     # write removed if requested
#     if out_removed_path:
#         rem_dfs = []
#         for p in parquet_input_paths:
#             p_abs = os.path.abspath(p)
#             rem_set = removed_by_src.get(p_abs)
#             if rem_set is None:
#                 continue
#             try:
#                 df = pd.read_parquet(p_abs)
#             except Exception as e:
#                 print(f"[WARN] failed to read parquet {p_abs} while writing removed: {e}", file=sys.stderr)
#                 continue
#             if df.shape[0] == 0:
#                 continue
#             mask = [False] * len(df)
#             idx_index = df.index
#             for pos in rem_set:
#                 try:
#                     loc = idx_index.get_indexer([pos])[0]
#                     if loc != -1:
#                         mask[loc] = True
#                         continue
#                 except Exception:
#                     pass
#                 try:
#                     if 0 <= int(pos) < len(df):
#                         mask[int(pos)] = True
#                 except Exception:
#                     pass
#             df_rem = df[pd.Series(mask).values]
#             if not df_rem.empty:
#                 # we may want to add reason columns, but user要求不新增/减少列 -> 所以不添加
#                 rem_dfs.append(df_rem)
#         if removed_fallback:
#             df_fb = records_to_dataframe(removed_fallback)
#             if not df_fb.empty:
#                 rem_dfs.append(df_fb)
#         if rem_dfs:
#             df_all_rem = pd.concat(rem_dfs, ignore_index=True, axis=0)
#         else:
#             df_all_rem = pd.DataFrame()
#         try:
#             if not df_all_rem.empty:
#                 df_all_rem.to_parquet(out_removed_path, index=False)
#             else:
#                 pd.DataFrame().to_parquet(out_removed_path, index=False)
#             print(f"[INFO] removed parquet written to {out_removed_path}", file=sys.stderr)
#         except Exception as e:
#             print(f"[WARN] parquet write failed ({e}), trying csv fallback for removed", file=sys.stderr)
#             try:
#                 df_all_rem.to_csv(out_removed_path + ".csv", index=False)
#                 print(f"[INFO] csv fallback written to {out_removed_path}.csv", file=sys.stderr)
#             except Exception as e2:
#                 print(f"[ERROR] fallback write failed for removed: {e2}", file=sys.stderr)

# # ---------------------------
# # argparse & main
# # ---------------------------
# def parse_args():
#     ap = argparse.ArgumentParser(description="Parquet-aware dedup (self_dedup/pair). Outputs kept/removed as parquet; does not change input.")
#     ap.add_argument("--train", nargs="+")
#     ap.add_argument("--test", nargs="+")
#     ap.add_argument("--out", required=True, help="输出 kept parquet 文件路径（例如 kept.parquet）")
#     ap.add_argument("--dump_removed", type=str, default=None, help="（可选）写 removed.parquet")
#     ap.add_argument("--threshold", type=float, default=0.99)
#     ap.add_argument("--mode", type=str, choices=["self_dedup", "pair"], default="self_dedup")
#     ap.add_argument("--topk", type=int, default=1)
#     ap.add_argument("--train_emb_dir", type=str, default=None)
#     ap.add_argument("--test_emb_dir", type=str, default=None)
#     ap.add_argument("--device", type=str, default="cuda")
#     ap.add_argument("--chunk_db", type=int, default=20000)
#     ap.add_argument("--chunk_q", type=int, default=256)
#     return ap.parse_args()

# def main():
#     args = parse_args()
#     mode = args.mode
#     threshold = float(args.threshold)
#     out_path = args.out
#     dump_removed = args.dump_removed

#     if mode == "self_dedup":
#         emb_dir = args.train_emb_dir or (args.test_emb_dir if args.test_emb_dir else None)
#         if emb_dir is None and args.train:
#             candidate = args.train[0]
#             if os.path.isdir(candidate) or (os.path.isfile(candidate) and candidate.lower().endswith(".parquet")):
#                 emb_dir = candidate
#         if emb_dir is None:
#             raise RuntimeError("self_dedup 模式需要指定 --train_emb_dir 或 --train 指向 emb 目录或 parquet 文件")
#         print(f"[INFO] self_dedup on {emb_dir}", file=sys.stderr)
#         records = load_emb_dir_to_records(emb_dir)
#         if not records:
#             print("[WARN] no records found", file=sys.stderr)
#             # still create empty parquet
#             pd.DataFrame().to_parquet(out_path, index=False)
#             return

#         kept, removed = self_dedup_records(records, threshold=threshold, topk=2, device=args.device, chunk_db=args.chunk_db, chunk_q=args.chunk_q)

#         # 获取原始 parquet 路径列表（用于按源回写）
#         parquet_paths = resolve_parquet_paths(emb_dir)

#         # 写出时尽量保留原始 parquet 的列：只删除重复行
#         write_kept_removed_preserve_parquet(parquet_paths, kept, removed, out_path, out_removed_path=dump_removed)

#         print(f"[INFO] self_dedup done. kept={len(kept)}, removed={len(removed)}", file=sys.stderr)

#     else:
#         # pair mode
#         train_dir = args.train_emb_dir or (args.train[0] if args.train and (os.path.isdir(args.train[0]) or args.train[0].lower().endswith(".parquet")) else None)
#         test_dir = args.test_emb_dir or (args.test[0] if args.test and (os.path.isdir(args.test[0]) or args.test[0].lower().endswith(".parquet")) else None)
#         if not train_dir or not test_dir:
#             raise RuntimeError("pair 模式需要提供 --train_emb_dir 与 --test_emb_dir（或通过 --train/--test 指向 emb 目录或 parquet 文件）")
#         print(f"[INFO] pair mode: train={train_dir}, test={test_dir}", file=sys.stderr)
#         kept, removed = filter_pair_emb_dirs_torch(train_dir, test_dir, threshold=threshold, topk=args.topk, device=args.device, chunk_db=args.chunk_db, chunk_q=args.chunk_q)

#         # 获取训练集原始 parquet 路径列表（用于按源回写）
#         parquet_paths = resolve_parquet_paths(train_dir)

#         # note: removed items for pair mode are dicts with 'record' inside; normalize to record dicts
#         removed_norm = []
#         for r in removed:
#             rec = {}
#             if isinstance(r.get("record"), dict):
#                 rec.update(r["record"])
#             # keep existing metadata if any (but we will try not to add new columns to original parquet)
#             # keep local_pos / src_parquet / src_parquet_full if present
#             for k in ("train_pos", "topk_sims", "topk_test_indices", "best_sim", "best_test_index", "best_test_user", "train_text"):
#                 if k in r:
#                     # avoid adding these to original-file-backed removed outputs (user要求不新增列)
#                     # instead they will be present only in fallback rows (if we must write them as separate records)
#                     # so we append them into rec only if no src_parquet available (handled in fallback)
#                     pass
#             removed_norm.append(rec)

#         # 写出 kept/removed（尽量保留原始 parquet 的列）
#         write_kept_removed_preserve_parquet(parquet_paths, kept, removed_norm, out_path, out_removed_path=dump_removed)

#         print(f"[INFO] pair done. kept={len(kept)}, removed={len(removed)}", file=sys.stderr)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faisst_filter_v2_torch_full_parquet.py
说明：
- 支持 json/.npy 对、以及 Parquet 输入（单文件或目录）
- 不会修改原始 parquet 文件；写出时仅去掉重复的行并保留所有原始列
- kept / removed 会以 parquet 格式写出（embedding 列以 list 存储）
- self_dedup / pair 两种模式（torch topk；无 GPU 回退 numpy）

本版修改（针对你的诉求）：
1) pair 模式 dump_removed：逐行写入最相似的 test 侧 prompt/orig（你数据里是 orig）
   新增列：matched_test_index / matched_test_orig / matched_test_prompt / matched_test_sim
2) 修复 matched_test_orig 为 None 的问题：把 parquet 的 orig/source 字段带入 records，并在提取时优先用 orig
3) 写 removed.parquet 时：先从源 parquet 精确抽取行，再把 matched_test_* 列按行顺序写回（逐行对齐）
"""
import argparse
import json
import os
import sys
import glob
import ast
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch

try:
    import pandas as pd
except Exception as e:
    print("[ERROR] pandas required. pip install pandas pyarrow", file=sys.stderr)
    raise


# ---------------------------
# parse embedding helper
# ---------------------------
def parse_embedding_field(val) -> Optional[np.ndarray]:
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val.astype(np.float32)
    if isinstance(val, (list, tuple)):
        try:
            return np.asarray(val, dtype=np.float32)
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return None
        try:
            arr = json.loads(s)
            if isinstance(arr, (list, tuple)):
                return np.asarray(arr, dtype=np.float32)
        except Exception:
            try:
                arr = ast.literal_eval(s)
                if isinstance(arr, (list, tuple)):
                    return np.asarray(arr, dtype=np.float32)
            except Exception:
                return None
    return None


# ---------------------------
# resolve parquet paths helper
# ---------------------------
def resolve_parquet_paths(path_or_dir: str) -> List[str]:
    """返回匹配的 parquet 文件的完整路径（按字母序）"""
    if path_or_dir is None:
        return []
    if os.path.isfile(path_or_dir) and path_or_dir.lower().endswith(".parquet"):
        return [os.path.abspath(path_or_dir)]
    if os.path.isdir(path_or_dir):
        ps = sorted(
            [os.path.join(path_or_dir, p) for p in os.listdir(path_or_dir) if p.lower().endswith(".parquet")]
        )
        return [os.path.abspath(p) for p in ps]
    # glob pattern
    ps = sorted(glob.glob(path_or_dir))
    ps = [p for p in ps if p.lower().endswith(".parquet")]
    return [os.path.abspath(p) for p in ps]


# ---------------------------
# parquet loader -> records (包含 emb numpy)
# ---------------------------
def load_parquet_files_to_records(path_or_dir: str) -> List[Dict[str, Any]]:
    paths = resolve_parquet_paths(path_or_dir)
    records: List[Dict[str, Any]] = []
    if not paths:
        return records

    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[WARN] failed to read parquet {p}: {e}", file=sys.stderr)
            continue

        cols = df.columns.tolist()
        idx_col = None
        for c in ("idx", "global_idx", "id", "index"):
            if c in cols:
                idx_col = c
                break

        for i, row in df.iterrows():
            try:
                emb = None
                if "embedding" in cols:
                    emb = parse_embedding_field(row["embedding"])
                for alt in ("emb", "vector", "embedding_vector"):
                    if emb is None and alt in cols:
                        emb = parse_embedding_field(row[alt])
                        if emb is not None:
                            break
                if emb is None:
                    continue

                if idx_col is not None:
                    try:
                        gi = int(row[idx_col])
                    except Exception:
                        gi = None
                else:
                    gi = None

                # local_pos: 尝试以行标签（index）保存为 int（若 index 为 int）
                local_pos = None
                if isinstance(i, (int, np.integer)):
                    local_pos = int(i)

                rec: Dict[str, Any] = {
                    "global_idx": int(gi) if gi is not None else None,
                    "emb": emb.astype(np.float32),
                    # 如果 parquet 没有 text，就留空；后续会用 orig 兜底
                    "text": row["text"] if "text" in cols and not pd.isna(row["text"]) else "",
                    "src_parquet": os.path.basename(p),
                    "src_parquet_full": os.path.abspath(p),
                    "local_pos": local_pos,
                }

                # NEW: 你的数据列里有 orig/source，这里显式带上
                if "orig" in cols and not pd.isna(row["orig"]):
                    rec["orig"] = row["orig"]
                if "source" in cols and not pd.isna(row["source"]):
                    rec["source"] = row["source"]

                # carry optional fields
                for opt in (
                    "uid",
                    "model_name",
                    "run_name",
                    "run_date",
                    "timestamp",
                    "score",
                    "label",
                    "orig_input_json",
                    "kmeans_label",
                    "idx",
                ):
                    if opt in cols:
                        val = row[opt]
                        try:
                            json.dumps(val, ensure_ascii=False)
                            rec[opt] = val
                        except Exception:
                            rec[opt] = str(val)

                records.append(rec)

            except Exception as e:
                print(f"[WARN] skip row {i} in {p}: {e}", file=sys.stderr)
                continue

    records.sort(
        key=lambda r: (
            r.get("src_parquet", ""),
            r.get("local_pos", 0) if r.get("local_pos") is not None else 0,
        )
    )
    return records


# ---------------------------
# original json/.npy loader (fallback)
# ---------------------------
def load_emb_dir_to_records(emb_dir: str) -> List[Dict[str, Any]]:
    # 如果是 parquet 路径，直接走 parquet loader（以确保 src_parquet_full 保存）
    if emb_dir is None:
        return []
    parquet_paths = resolve_parquet_paths(emb_dir)
    if parquet_paths:
        return load_parquet_files_to_records(emb_dir)
    if not os.path.isdir(emb_dir):
        raise RuntimeError(f"emb_dir not found: {emb_dir}")

    records: List[Dict[str, Any]] = []
    json_files = sorted([f for f in os.listdir(emb_dir) if f.lower().endswith(".json")])
    for json_name in json_files:
        json_path = os.path.join(emb_dir, json_name)
        base = os.path.splitext(json_name)[0]
        emb_path = os.path.join(emb_dir, base + ".npy")
        if not os.path.exists(emb_path):
            print(f"[WARN] skip {json_path}, missing {emb_path}", file=sys.stderr)
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as rf:
                meta = json.load(rf)
        except Exception as e:
            print(f"[WARN] failed to load meta {json_path}: {e}", file=sys.stderr)
            continue

        idxs = meta.get("idx", [])
        texts = meta.get("texts", [])
        inputs = meta.get("inputs", None)
        try:
            emb = np.load(emb_path, allow_pickle=False)
        except Exception as e:
            print(f"[WARN] failed to load emb {emb_path}: {e}", file=sys.stderr)
            continue

        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        n_meta = min(len(idxs), len(texts))
        n_emb = emb.shape[0]
        if n_emb == 0 or n_meta == 0:
            continue
        n_use = min(n_meta, n_emb)

        for i_local in range(n_use):
            try:
                gi = int(idxs[i_local])
            except Exception:
                gi = None
            emb_row = emb[i_local].astype(np.float32)
            txt = texts[i_local] if i_local < len(texts) else ""
            orig_input = None
            if inputs is not None and isinstance(inputs, (list, tuple)) and i_local < len(inputs):
                orig_input = inputs[i_local]
            records.append(
                {
                    "global_idx": gi,
                    "emb": emb_row,
                    "text": txt,
                    "inputs": orig_input,
                    "src_json": json_name,
                    "local_pos": i_local,
                }
            )

    records.sort(key=lambda r: (r.get("src_json", ""), r.get("local_pos", 0)))
    return records


# ---------------------------
# normalize
# ---------------------------
def _ensure_normalized(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


# ---------------------------
# prompt/orig extraction helpers
# ---------------------------
def _safe_to_str_or_json(v):
    """尽量保留可写入 parquet 的类型；复杂对象转 json 字符串。"""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def extract_prompt_or_orig(rec: Dict[str, Any]) -> Tuple[str, Any]:
    """
    你的数据里通常用 orig 来存原始输入，所以这里优先用 orig。
    - prompt：优先 text，其次 orig
    - orig：优先 orig，其次 orig_input_json，再次 inputs
    """
    prompt = rec.get("text", "") or ""
    if not prompt:
        prompt = rec.get("orig", "") or ""
    prompt = json.loads(prompt)['conversations'][-1]['content']
    orig = rec.get("orig", None)
    if orig is None:
        orig = rec.get("orig_input_json", None)
    if orig is None:
        orig = rec.get("inputs", None)

    return prompt, _safe_to_str_or_json(orig)


# ---------------------------
# build arrays & keep emb in orig_records (so kept can be written back)
# ---------------------------
def build_array_from_emb_dir(emb_dir: str) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    records = load_emb_dir_to_records(emb_dir)
    if not records:
        return np.zeros((0, 0), dtype=np.float32), [], []

    X = np.vstack([r["emb"] for r in records]).astype(np.float32)
    texts = [r.get("text", "") for r in records]

    orig_records = []
    for r in records:
        rec = {}
        for k in (
            "uid",
            "global_idx",
            "idx",
            "text",
            "orig",        # NEW
            "source",      # NEW
            "src_parquet",
            "src_parquet_full",
            "src_json",
            "local_pos",
            "model_name",
            "run_name",
            "run_date",
            "timestamp",
            "score",
            "label",
            "orig_input_json",
            "kmeans_label",
            "inputs",
        ):
            if k in r:
                rec[k] = r[k]

        rec["emb"] = r["emb"]
        orig_records.append(rec)

    return X, texts, orig_records


# ---------------------------
# topk torch/numpy (unchanged)
# ---------------------------
def topk_sim_torch(
    Xq: np.ndarray,
    Xdb: np.ndarray,
    topk: int = 1,
    device: str = "cuda",
    chunk_db: int = 20000,
    chunk_q: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    assert Xq.dtype == np.float32 and Xdb.dtype == np.float32
    nq, d = Xq.shape
    nd = Xdb.shape[0]
    k_eff = min(topk, nd) if nd > 0 else 0
    if k_eff <= 0:
        return np.zeros((nq, 0), dtype=np.float32), np.zeros((nq, 0), dtype=np.int64)

    if device == "cpu" or not torch.cuda.is_available():
        sims_full = np.matmul(Xq, Xdb.T)
        idxs_part = np.argpartition(-sims_full, range(k_eff), axis=1)[:, :k_eff]
        topk_sims = np.take_along_axis(sims_full, idxs_part, axis=1)
        order = np.argsort(-topk_sims, axis=1)
        topk_sims = np.take_along_axis(topk_sims, order, axis=1)
        idxs_sorted = np.take_along_axis(idxs_part, order, axis=1)
        return topk_sims.astype(np.float32), idxs_sorted.astype(np.int64)

    dev = torch.device(device)
    Xdb_t = torch.from_numpy(Xdb).to(dev)
    result_sims = np.zeros((nq, k_eff), dtype=np.float32)
    result_idxs = np.zeros((nq, k_eff), dtype=np.int64)

    with torch.no_grad():
        for qstart in range(0, nq, chunk_q):
            qend = min(nq, qstart + chunk_q)
            Xq_chunk = torch.from_numpy(Xq[qstart:qend]).to(dev)

            topk_vals = None
            topk_idxs = None

            for dbstart in range(0, nd, chunk_db):
                dbend = min(nd, dbstart + chunk_db)
                Xdb_chunk = Xdb_t[dbstart:dbend]
                sims_chunk = torch.matmul(Xq_chunk, Xdb_chunk.t())
                k_cur = min(k_eff, sims_chunk.size(1))
                vals, idxs = torch.topk(sims_chunk, k_cur, dim=1)
                idxs = idxs + dbstart

                if topk_vals is None:
                    topk_vals = vals
                    topk_idxs = idxs
                else:
                    cat_vals = torch.cat([topk_vals, vals], dim=1)
                    cat_idxs = torch.cat([topk_idxs, idxs], dim=1)
                    k_choose = min(k_eff, cat_vals.size(1))
                    vals2, pos = torch.topk(cat_vals, k_choose, dim=1)
                    idxs2 = torch.gather(cat_idxs, 1, pos)
                    topk_vals = vals2
                    topk_idxs = idxs2

                del sims_chunk, vals, idxs
                torch.cuda.empty_cache()

            result_sims[qstart:qend] = topk_vals.cpu().numpy().astype(np.float32)
            result_idxs[qstart:qend] = topk_idxs.cpu().numpy().astype(np.int64)
            del Xq_chunk, topk_vals, topk_idxs
            torch.cuda.empty_cache()

    return result_sims, result_idxs


# ---------------------------
# self_dedup (unchanged behavior)
# ---------------------------
def self_dedup_records(
    records: List[Dict[str, Any]],
    threshold: float,
    topk: int = 2,
    device: str = "cuda",
    chunk_db: int = 20000,
    chunk_q: int = 256,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    N = len(records)
    if N == 0:
        return [], []

    X = np.vstack([r["emb"] for r in records]).astype(np.float32)
    X = np.ascontiguousarray(_ensure_normalized(X), dtype=np.float32)

    k_eff = min(max(2, int(topk)), N)
    sims, idxs = topk_sim_torch(X, X, topk=k_eff, device=device, chunk_db=chunk_db, chunk_q=chunk_q)

    keep_flags = np.ones((N,), dtype=bool)
    for i in range(N):
        if not keep_flags[i]:
            continue
        neighbors = idxs[i].tolist()
        sims_row = sims[i].tolist()
        for nb_pos, nb in enumerate(neighbors):
            if nb == i:
                continue
            nb_sim = sims_row[nb_pos]
            if nb_sim >= threshold:
                if nb < i:
                    keep_flags[i] = False
                else:
                    keep_flags[nb] = False
                break

    kept = [records[i] for i in range(N) if keep_flags[i]]
    removed = []
    for i in range(N):
        if not keep_flags[i]:
            neighbors = idxs[i].tolist()
            sims_row = sims[i].tolist()
            cause = None
            for nb_pos, nb in enumerate(neighbors):
                if nb == i:
                    continue
                if keep_flags[nb] and sims_row[nb_pos] >= threshold:
                    cause = {"best_sim": float(sims_row[nb_pos]), "best_idx": int(nb)}
                    break
            removed_rec = {
                "global_idx": int(records[i].get("global_idx")) if records[i].get("global_idx") is not None else None,
                "text": records[i].get("text", ""),
                "inputs": records[i].get("inputs"),
                "src_parquet": records[i].get("src_parquet"),
                "src_parquet_full": records[i].get("src_parquet_full"),
                "local_pos": records[i].get("local_pos"),
                "reason": cause,
            }
            for opt in ("uid", "model_name", "run_name", "run_date", "timestamp", "score", "label", "orig_input_json", "kmeans_label", "orig", "source", "idx"):
                if opt in records[i]:
                    removed_rec[opt] = records[i][opt]
            removed.append(removed_rec)
    return kept, removed


# ---------------------------
# pair mode (MODIFIED)
# ---------------------------
def filter_pair_emb_dirs_torch(
    train_emb_dir: str,
    test_emb_dir: str,
    threshold: float,
    topk: int = 1,
    device: str = "cuda",
    chunk_db: int = 20000,
    chunk_q: int = 256,
):
    X_train, train_texts, train_recs = build_array_from_emb_dir(train_emb_dir)
    X_test, test_texts, test_recs = build_array_from_emb_dir(test_emb_dir)

    if X_train.size == 0:
        return [], []
    if X_test.size == 0:
        kept = train_recs.copy()
        removed = []
        return kept, removed

    X_train = np.ascontiguousarray(_ensure_normalized(X_train), dtype=np.float32)
    X_test = np.ascontiguousarray(_ensure_normalized(X_test), dtype=np.float32)

    nd = X_test.shape[0]
    k_eff = min(int(topk), nd) if nd > 0 else 0
    sims, idxs = topk_sim_torch(X_train, X_test, topk=k_eff, device=device, chunk_db=chunk_db, chunk_q=chunk_q)

    best_arg = np.argmax(sims, axis=1)
    best_sims = sims[np.arange(sims.shape[0]), best_arg]
    best_idxs = idxs[np.arange(idxs.shape[0]), best_arg]

    # NOTE: 如果 train==test 且 threshold <= 1.0，那么 best_sim=1.0 会导致所有行被 removed（符合你说的“理论上全部筛除”）
    keep_mask = best_sims < threshold

    kept = [train_recs[i] for i, keep in enumerate(keep_mask) if keep]

    removed = []
    for i, keep in enumerate(keep_mask):
        if keep:
            continue

        best_sim = float(best_sims[i])
        best_idx = int(best_idxs[i])

        test_rec = test_recs[best_idx] if (0 <= best_idx < len(test_recs)) else {}
        test_prompt, test_orig = extract_prompt_or_orig(test_rec)

        rec = train_recs[i].copy()
        rec["matched_test_prompt"] = test_prompt
        rec["matched_test_orig"] = test_orig
        rec["matched_test_sim"] = best_sim
        rec["matched_test_index"] = best_idx

        removed.append(rec)

    return kept, removed


# ---------------------------
# records -> dataframe fallback (for json/npy sources)
# ---------------------------
def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = []
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    for r in records:
        row = {}
        for k in all_keys:
            v = r.get(k, None)
            if k == "emb" and v is not None:
                try:
                    row["embedding"] = (v.tolist() if isinstance(v, np.ndarray) else list(v))
                except Exception:
                    row["embedding"] = str(v)
            else:
                row[k] = v
        rows.append(row)
    return pd.DataFrame.from_records(rows)


# ---------------------------
# write parquet while preserving original columns when possible
# ---------------------------
def write_kept_removed_preserve_parquet(
    parquet_input_paths: List[str],
    kept_records: List[Dict[str, Any]],
    removed_records: List[Dict[str, Any]],
    out_kept_path: str,
    out_removed_path: Optional[str] = None,
):
    """
    - kept：严格保留原始 parquet 列，只抽行，不额外加列
    - removed：从原始 parquet 抽行 + 逐行附加 matched_test_* 列
    """
    basename_to_full = {os.path.basename(p): p for p in parquet_input_paths}

    def group_by_src(records: List[Dict[str, Any]]):
        by_src = {}
        fallback = []
        for r in records:
            src = r.get("src_parquet")
            src_full = r.get("src_parquet_full")
            local_pos = r.get("local_pos")
            if src is None and src_full is None:
                fallback.append(r)
                continue
            if src_full:
                key_full = os.path.abspath(src_full)
            else:
                key_full = basename_to_full.get(src)
            if key_full is None:
                fallback.append(r)
                continue
            s = by_src.get(key_full)
            if s is None:
                s = set()
                by_src[key_full] = s
            if local_pos is not None:
                s.add(local_pos)
            else:
                fallback.append(r)
        return by_src, fallback

    def group_by_src_with_extras(records: List[Dict[str, Any]]):
        by_src = {}
        fallback = []

        EXTRA_KEYS = ("matched_test_prompt", "matched_test_orig", "matched_test_sim", "matched_test_index")

        for r in records:
            src = r.get("src_parquet")
            src_full = r.get("src_parquet_full")
            local_pos = r.get("local_pos")

            if src is None and src_full is None:
                fallback.append(r)
                continue

            if src_full:
                key_full = os.path.abspath(src_full)
            else:
                key_full = basename_to_full.get(src)

            if key_full is None or local_pos is None:
                fallback.append(r)
                continue

            extras = {k: r.get(k, None) for k in EXTRA_KEYS if k in r}
            by_src.setdefault(key_full, []).append((local_pos, extras))

        return by_src, fallback

    kept_by_src, kept_fallback = group_by_src(kept_records)
    removed_by_src, removed_fallback = group_by_src_with_extras(removed_records)

    # ---- write kept ----
    kept_dfs = []
    for p in parquet_input_paths:
        p_abs = os.path.abspath(p)
        keep_set = kept_by_src.get(p_abs)
        if keep_set is None:
            continue
        try:
            df = pd.read_parquet(p_abs)
        except Exception as e:
            print(f"[WARN] failed to read parquet {p_abs} while writing kept: {e}", file=sys.stderr)
            continue
        if df.shape[0] == 0:
            continue

        mask = [False] * len(df)
        idx_index = df.index
        for pos in keep_set:
            try:
                loc = idx_index.get_indexer([pos])[0]
                if loc != -1:
                    mask[loc] = True
                    continue
            except Exception:
                pass
            try:
                ipos = int(pos)
                if 0 <= ipos < len(df):
                    mask[ipos] = True
            except Exception:
                pass

        df_keep = df[pd.Series(mask).values]
        if not df_keep.empty:
            kept_dfs.append(df_keep)

    if kept_fallback:
        df_fb = records_to_dataframe(kept_fallback)
        if not df_fb.empty:
            kept_dfs.append(df_fb)

    df_all_kept = pd.concat(kept_dfs, ignore_index=True, axis=0) if kept_dfs else pd.DataFrame()
    try:
        if not df_all_kept.empty:
            df_all_kept.to_parquet(out_kept_path, index=False)
        else:
            pd.DataFrame().to_parquet(out_kept_path, index=False)
        print(f"[INFO] parquet written to {out_kept_path}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] parquet write failed ({e}), trying csv fallback for kept", file=sys.stderr)
        try:
            df_all_kept.to_csv(out_kept_path + ".csv", index=False)
            print(f"[INFO] csv fallback written to {out_kept_path}.csv", file=sys.stderr)
        except Exception as e2:
            print(f"[ERROR] fallback write failed for kept: {e2}", file=sys.stderr)

    # ---- write removed ----
    if out_removed_path:
        rem_dfs = []
        for p in parquet_input_paths:
            p_abs = os.path.abspath(p)
            items = removed_by_src.get(p_abs)  # [(local_pos, extras), ...]
            if not items:
                continue

            try:
                df = pd.read_parquet(p_abs)
            except Exception as e:
                print(f"[WARN] failed to read parquet {p_abs} while writing removed: {e}", file=sys.stderr)
                continue
            if df.shape[0] == 0:
                continue

            idx_index = df.index
            locs = []
            extras_list = []

            for pos, extras in items:
                loc = -1
                try:
                    loc = idx_index.get_indexer([pos])[0]
                except Exception:
                    loc = -1
                if loc == -1:
                    try:
                        ipos = int(pos)
                        if 0 <= ipos < len(df):
                            loc = ipos
                    except Exception:
                        loc = -1

                if loc != -1:
                    locs.append(loc)
                    extras_list.append(extras)

            if not locs:
                continue

            df_rem = df.iloc[locs].copy()

            all_extra_keys = set()
            for ex in extras_list:
                all_extra_keys.update(ex.keys())

            for k in sorted(all_extra_keys):
                df_rem[k] = [ex.get(k, None) for ex in extras_list]

            rem_dfs.append(df_rem)

        if removed_fallback:
            df_fb = records_to_dataframe(removed_fallback)
            if not df_fb.empty:
                rem_dfs.append(df_fb)

        df_all_rem = pd.concat(rem_dfs, ignore_index=True, axis=0) if rem_dfs else pd.DataFrame()
        try:
            if not df_all_rem.empty:
                df_all_rem.to_parquet(out_removed_path, index=False)
            else:
                pd.DataFrame().to_parquet(out_removed_path, index=False)
            print(f"[INFO] removed parquet written to {out_removed_path}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] parquet write failed ({e}), trying csv fallback for removed", file=sys.stderr)
            try:
                df_all_rem.to_csv(out_removed_path + ".csv", index=False)
                print(f"[INFO] csv fallback written to {out_removed_path}.csv", file=sys.stderr)
            except Exception as e2:
                print(f"[ERROR] fallback write failed for removed: {e2}", file=sys.stderr)


# ---------------------------
# argparse & main
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Parquet-aware dedup (self_dedup/pair). Outputs kept/removed as parquet; does not change input."
    )
    ap.add_argument("--train", nargs="+")
    ap.add_argument("--test", nargs="+")
    ap.add_argument("--out", required=True, help="输出 kept parquet 文件路径（例如 kept.parquet）")
    ap.add_argument("--dump_removed", type=str, default=None, help="（可选）写 removed.parquet")
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--mode", type=str, choices=["self_dedup", "pair"], default="self_dedup")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--train_emb_dir", type=str, default=None)
    ap.add_argument("--test_emb_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--chunk_db", type=int, default=20000)
    ap.add_argument("--chunk_q", type=int, default=256)
    return ap.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    threshold = float(args.threshold)
    out_path = args.out
    dump_removed = args.dump_removed

    if mode == "self_dedup":
        emb_dir = args.train_emb_dir or (args.test_emb_dir if args.test_emb_dir else None)
        if emb_dir is None and args.train:
            candidate = args.train[0]
            if os.path.isdir(candidate) or (os.path.isfile(candidate) and candidate.lower().endswith(".parquet")):
                emb_dir = candidate
        if emb_dir is None:
            raise RuntimeError("self_dedup 模式需要指定 --train_emb_dir 或 --train 指向 emb 目录或 parquet 文件")

        print(f"[INFO] self_dedup on {emb_dir}", file=sys.stderr)
        records = load_emb_dir_to_records(emb_dir)
        if not records:
            print("[WARN] no records found", file=sys.stderr)
            pd.DataFrame().to_parquet(out_path, index=False)
            return

        kept, removed = self_dedup_records(
            records, threshold=threshold, topk=2, device=args.device, chunk_db=args.chunk_db, chunk_q=args.chunk_q
        )

        parquet_paths = resolve_parquet_paths(emb_dir)
        write_kept_removed_preserve_parquet(parquet_paths, kept, removed, out_path, out_removed_path=dump_removed)

        print(f"[INFO] self_dedup done. kept={len(kept)}, removed={len(removed)}", file=sys.stderr)

    else:
        train_dir = args.train_emb_dir or (
            args.train[0] if args.train and (os.path.isdir(args.train[0]) or args.train[0].lower().endswith(".parquet")) else None
        )
        test_dir = args.test_emb_dir or (
            args.test[0] if args.test and (os.path.isdir(args.test[0]) or args.test[0].lower().endswith(".parquet")) else None
        )
        if not train_dir or not test_dir:
            raise RuntimeError("pair 模式需要提供 --train_emb_dir 与 --test_emb_dir（或通过 --train/--test 指向 emb 目录或 parquet 文件）")

        print(f"[INFO] pair mode: train={train_dir}, test={test_dir}", file=sys.stderr)

        kept, removed = filter_pair_emb_dirs_torch(
            train_dir,
            test_dir,
            threshold=threshold,
            topk=args.topk,
            device=args.device,
            chunk_db=args.chunk_db,
            chunk_q=args.chunk_q,
        )

        parquet_paths = resolve_parquet_paths(train_dir)
        write_kept_removed_preserve_parquet(parquet_paths, kept, removed, out_path, out_removed_path=dump_removed)

        print(f"[INFO] pair done. kept={len(kept)}, removed={len(removed)}", file=sys.stderr)


if __name__ == "__main__":
    main()
