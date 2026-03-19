#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_and_append_kmeans_label_incremental.py

功能：
- 支持对大规模 embeddings（按 parquet 存储，可能达到百万级）做增量 PCA + 增量 MiniBatchKMeans
- 采样寻找 best_k（silhouette）
- 训练完成后逐文件按批次 predict 并原子写回 kmeans_label（int64）
- 可选把写回文件写到 output_dir（--write_labels_to_output_dir）
- 可选做 UMAP 可视化，但仅在采样数据上做（避免全量 UMAP 导致 OOM / 非常慢）

依赖：
  pyarrow, numpy, scikit-learn, umap-learn (可选), matplotlib (可选)
"""
from __future__ import annotations
import os
import argparse
import json
import tempfile
import traceback
import math
import random
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# optional
try:
    import umap.umap_ as umap
except Exception:
    umap = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# -----------------------
# 基础 I/O helper（保留原脚本的 append_label_to_parquet_atomic 行为）
# -----------------------
def read_embeddings_from_parquet(path):
    """
    返回 (n_rows, emb_dim), texts_list (or None), any_inputs_list (or None)
    保持 parquet 原始行序。
    该函数用于读取单文件全部 embeddings（仅在文件不太大时可用）。
    """
    tbl = pq.read_table(path)
    names = tbl.schema.names
    if "embedding" not in names:
        raise RuntimeError(f"parquet {path} 没有 'embedding' 列")
    emb_pylist = tbl.column("embedding").to_pylist()
    try:
        emb_np = np.vstack([np.asarray(x, dtype=np.float32) for x in emb_pylist]).astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"无法将 embedding 转为 numpy array: {e}")
    texts = tbl.column("text").to_pylist() if "text" in names else [None] * len(emb_pylist)
    inputs = None
    if "orig_input_json" in names:
        inputs = tbl.column("orig_input_json").to_pylist()
    elif "inputs" in names:
        inputs = tbl.column("inputs").to_pylist()
    else:
        inputs = [None] * len(emb_pylist)
    return emb_np, texts, inputs, tbl.num_rows

def append_label_to_parquet_atomic(path, labels_segment, write_back=True, dst_dir=None):
    """
    将 labels_segment 添加为末尾列 'kmeans_label'（int64）。
    如果已存在该列则先移除再添加，最后原子替换目标文件（dst）。
    返回写入后的目标路径。
    """
    if len(labels_segment) == 0:
        return path
    tbl = pq.read_table(path)
    if len(labels_segment) != tbl.num_rows:
        raise RuntimeError(f"labels length {len(labels_segment)} != rows {tbl.num_rows} for {path}")
    names = tbl.schema.names
    cols = []
    for n in names:
        if n != "kmeans_label":
            cols.append(tbl.column(n))
    label_arr = pa.array([int(x) for x in labels_segment], type=pa.int64())
    cols.append(label_arr)
    new_names = [n for n in names if n != "kmeans_label"] + ["kmeans_label"]
    new_tbl = pa.Table.from_arrays(cols, names=new_names)

    base_name = os.path.basename(path)
    if dst_dir:
        if write_back:
            dst = os.path.join(dst_dir, base_name)
        else:
            dst = os.path.join(dst_dir, base_name + ".with_labels.parquet")
    else:
        if write_back:
            dst = path
        else:
            dst = path + ".with_labels.parquet"

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet", prefix="tmp_parquet_", dir=os.path.dirname(dst) or ".")
    os.close(tmp_fd)
    pq.write_table(new_tbl, tmp_path, compression="zstd")
    os.replace(tmp_path, dst)
    return dst

# -----------------------
# Arrow -> NumPy（支持 FixedSizeList 快速路径）
# -----------------------
def _embedding_array_to_numpy(arr: pa.Array) -> np.ndarray:
    """
    将 RecordBatch 的 embedding 列（pa.Array）转为 np.ndarray [batch, dim] (float32)
    - FixedSizeList<float>: 走 values -> to_numpy -> reshape 的低拷贝路径
    - List<float>/LargeList<float>: 使用 offsets 从扁平值中抽取（若等长）否则退化到 pylist
    """
    t = arr.type

    # 1) FixedSizeList<float>
    if isinstance(t, pa.FixedSizeListType):
        dim = t.list_size
        flat = arr.values  # 注意：这是属性，不是函数！
        flat_np = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float32)
        if dim == 0:
            return np.empty((len(arr), 0), dtype=np.float32)
        if flat_np.size % dim != 0:
            raise RuntimeError(f"FixedSizeList 展平后长度 {flat_np.size} 不能被 dim={dim} 整除")
        return flat_np.reshape(-1, dim)

    # 2) List<float> / LargeList<float>
    if isinstance(t, (pa.ListType, pa.LargeListType)):
        # 若每行长度一致，可用 offsets 做零/低拷贝；否则退化 pylist
        offsets = arr.offsets  # Int32Array / Int64Array
        lens = offsets[1:].to_numpy() - offsets[:-1].to_numpy()
        unique_lens = np.unique(lens)
        values = arr.values  # 扁平化后的值数组
        vals_np = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32)
        if len(unique_lens) == 1:
            dim = int(unique_lens[0])
            if dim == 0:
                return np.empty((len(arr), 0), dtype=np.float32)
            if vals_np.size != len(arr) * dim:
                # 数据异常，退化 pylist
                pylist = arr.to_pylist()
                rows = [np.asarray(x, dtype=np.float32) for x in pylist if x is not None]
                return np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)
            return vals_np.reshape(-1, dim)
        else:
            # 行长度不一致 -> 退化 pylist（可能是脏数据或变长嵌入）
            pylist = arr.to_pylist()
            rows, first_len = [], None
            for x in pylist:
                if x is None:
                    continue
                a = np.asarray(x, dtype=np.float32)
                if first_len is None:
                    first_len = a.shape[0]
                elif a.shape[0] != first_len:
                    # 发现不等长，直接跳过该行（也可改成 pad/raise）
                    continue
                rows.append(a)
            return np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)

    # 3) 其他类型（不期望）：退化 pylist
    pylist = arr.to_pylist()
    rows = [np.asarray(x, dtype=np.float32) for x in pylist if x is not None]
    return np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)

# -----------------------
# 流式读取 embeddings（按文件、按批次）
# -----------------------
def iter_embeddings_batches(parquet_dir, batch_size=16384, shuffle_files=False, columns=["embedding"]):
    """
    逐文件、按 batch_size 返回 (file_path, np_array_batch)
    np_array_batch dtype=float32, shape=(b, dim)
    NOTE: 如果一个 parquet file 非常大，会分多个批次产出多次 (same file_path, different batches)
    """
    files = sorted([os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.lower().endswith(".parquet")])
    if shuffle_files:
        random.shuffle(files)
    for p in files:
        try:
            pf = pq.ParquetFile(p)
            for rb in pf.iter_batches(columns=columns, batch_size=batch_size):
                col0 = rb.column(0)
                arr = _embedding_array_to_numpy(col0)
                if arr.size == 0:
                    continue
                yield p, arr.astype(np.float32, copy=False)
        except Exception as e:
            print(f"[WARN] iter file {p} failed: {e}")
            continue

# -----------------------
# 采样用于 silhouette 搜索（避免整列读取；流式分批）
# -----------------------
def sample_embeddings_for_silhouette(
    parquet_dir,
    sample_size=20000,
    per_file_limit=1000,
    columns=["embedding"],
    rng_seed=42,
    sampler_batch_size=4096,   # 分批大小，按行数
):
    """
    从每个文件尽量均匀抽样（上限 per_file_limit），**流式分批**读取，避免整列 read 导致 List index overflow。
    返回 numpy [n<=sample_size, dim], float32
    """
    files = sorted([os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.lower().endswith(".parquet")])
    if not files:
        raise RuntimeError("目录内无 parquet 文件")

    rng = np.random.default_rng(rng_seed)
    collected = []
    total_target = sample_size
    target_per_file = max(1, min(per_file_limit, int(math.ceil(total_target / max(1, len(files))))))

    for p in files:
        try:
            pf = pq.ParquetFile(p)
            taken_this_file = 0

            for rb in pf.iter_batches(columns=columns, batch_size=sampler_batch_size):
                col = rb.column(0)  # embedding
                batch_np = _embedding_array_to_numpy(col)  # [b, dim]
                b = batch_np.shape[0]
                if b == 0:
                    continue

                remain = target_per_file - taken_this_file
                if remain <= 0:
                    break

                take = min(remain, b)
                if b <= take:
                    choice_idx = np.arange(b)
                else:
                    choice_idx = rng.choice(b, size=take, replace=False)

                collected.append(batch_np[choice_idx])
                taken_this_file += take

        except Exception as e:
            print(f"[WARN] streaming sample {p} failed: {e}")
            continue

    if not collected:
        raise RuntimeError("没有采样到任何 embeddings")

    Xs = np.vstack(collected).astype(np.float32)
    # 全局上限裁剪到 sample_size
    if Xs.shape[0] > sample_size:
        idx = rng.choice(Xs.shape[0], size=sample_size, replace=False)
        Xs = Xs[idx]

    return Xs

# -----------------------
# silhouette 搜索 best_k（在采样数据或降维后数据上运行）
# -----------------------
def search_best_k_by_silhouette(X_reduced, k_min, k_max, rng_seed, sample_size=None, sample_frac=None, min_sample=50):
    """
    在 X_reduced（numpy array）上穷举 k 并返回 best_k（silhouette 分数最大）
    """
    N = len(X_reduced)
    if N < 2:
        return 1
    if sample_frac is not None:
        chosen_n = int(max(min_sample, sample_frac * N))
    elif sample_size is not None:
        chosen_n = int(min(sample_size, N))
    else:
        chosen_n = min(max(min_sample, N), N)
    chosen_n = max(2, min(N, chosen_n))
    rng = np.random.default_rng(rng_seed)
    if chosen_n < N:
        idx = rng.choice(N, size=chosen_n, replace=False)
        X_sample = X_reduced[idx]
    else:
        X_sample = X_reduced
    best_k = None
    best_score = -999.0
    for k in range(k_min, k_max + 1):
        if k < 2 or k >= len(X_sample):
            continue
        try:
            km = MiniBatchKMeans(n_clusters=k, batch_size=1024, random_state=rng_seed)
            labs = km.fit_predict(X_sample)
            score = silhouette_score(X_sample, labs)
            print(f"[INFO] silhouette k={k} score={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            print(f"[WARN] silhouette for k={k} failed: {e}")
    if best_k is None:
        best_k = max(2, min(8, int(math.sqrt(len(X_reduced)))))
        print(f"[WARN] silhouette 无有效候选，兜底 K={best_k}")
    return best_k

# -----------------------
# 增量 PCA 训练
# -----------------------
def incremental_fit_pca(parquet_dir, pca_dim, batch_size=16384):
    """
    如果 pca_dim <= 0 则返回 None（不降维）。
    否则创建 IncrementalPCA 并对全量数据按批次 partial_fit。
    """
    if pca_dim is None or pca_dim <= 0:
        return None
    ipca = IncrementalPCA(n_components=pca_dim)
    seen = 0
    for _, arr in iter_embeddings_batches(parquet_dir, batch_size=batch_size):
        try:
            ipca.partial_fit(arr)
            seen += arr.shape[0]
        except Exception as e:
            print(f"[WARN] IPCA partial_fit batch failed: {e}")
            # 若某些 batch 太大，拆分小块
            nsub = max(1, arr.shape[0] // 2048)
            step = max(1, arr.shape[0] // nsub)
            for i in range(nsub):
                s = i * step
                eidx = (i + 1) * step if i < nsub - 1 else arr.shape[0]
                ipca.partial_fit(arr[s:eidx])
            seen += arr.shape[0]
    print(f"[INFO] IncrementalPCA fitted on ~{seen} samples")
    return ipca

# -----------------------
# 增量 MiniBatchKMeans 训练
# -----------------------
def incremental_fit_kmeans(parquet_dir, best_k, batch_size=8192, pca_transform=None, rng_seed=42):
    """
    使用 MiniBatchKMeans.partial_fit 逐批训练（流式）。返回训练好的模型。
    """
    if best_k <= 0:
        raise RuntimeError("best_k must > 0")
    mbk = MiniBatchKMeans(n_clusters=best_k, batch_size=4096, random_state=rng_seed)
    seen = 0
    for _, arr in iter_embeddings_batches(parquet_dir, batch_size=batch_size):
        X_batch = arr
        if pca_transform is not None:
            X_batch = pca_transform.transform(X_batch)
        try:
            mbk.partial_fit(X_batch)
        except Exception as e:
            # 若批次仍太大导致问题，拆分后 partial_fit
            nsub = max(1, X_batch.shape[0] // 2048)
            step = max(1, X_batch.shape[0] // nsub)
            for i in range(nsub):
                s = i * step
                eidx = (i + 1) * step if i < nsub - 1 else X_batch.shape[0]
                mbk.partial_fit(X_batch[s:eidx])
        seen += X_batch.shape[0]
    print(f"[INFO] MiniBatchKMeans partial_fitted on ~{seen} samples")
    return mbk

# -----------------------
# 逐文件 predict 并写回标签
# -----------------------
def predict_and_write_labels_by_file(parquet_dir, kmeans_model, batch_size=8192,
                                    pca_transform=None, write_back=True, dst_dir=None):
    """
    对每个文件按批次 predict label（内存只保留当前文件的 labels），然后调用 append_label_to_parquet_atomic 写回。
    """
    files = sorted([os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.lower().endswith(".parquet")])
    for p in files:
        try:
            pf = pq.ParquetFile(p)
            labels_list = []
            rows = 0
            for rb in pf.iter_batches(columns=["embedding"], batch_size=batch_size):
                col = rb.column(0)
                arr = _embedding_array_to_numpy(col)
                if arr.size == 0:
                    continue
                if pca_transform is not None:
                    arr = pca_transform.transform(arr)
                labs = kmeans_model.predict(arr)
                labels_list.append(labs)
                rows += arr.shape[0]
            if rows == 0:
                print(f"[INFO] skip empty file {p}")
                continue
            seg = np.concatenate(labels_list, axis=0).astype(np.int64)
            outp = append_label_to_parquet_atomic(p, seg, write_back=write_back, dst_dir=dst_dir)
            print(f"[INFO] wrote labels to {outp} (rows={rows})")
        except Exception as e:
            print(f"[ERROR] predict/write for {p} failed: {e}\n{traceback.format_exc()}")
            continue

# -----------------------
# 主流程
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", type=str, required=True, help="包含 parquet 的目录")
    parser.add_argument("--output_dir", type=str, default="./cluster_out")
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=15)
    parser.add_argument("--rng_seed", type=int, default=42)
    parser.add_argument("--pca_dim", type=int, default=50, help="PCA 后维度上限，<=0 表示不做 PCA")
    parser.add_argument("--sample_size_for_silhouette", type=int, default=20000,
                        help="用于 silhouette 搜索的采样总数（越大越稳，但更慢/占内存）")
    parser.add_argument("--per_file_sample_limit", type=int, default=1000,
                        help="每个文件采样上限（避免单文件主导样本）")
    parser.add_argument("--batch_size_ipca", type=int, default=16384, help="IncrementalPCA 批大小")
    parser.add_argument("--batch_size_kmeans", type=int, default=8192, help="kmeans partial_fit / predict 时的批大小")
    parser.add_argument("--sample_frac_for_search", type=float, default=None, help="可选：以 frac 采样用于 search")
    parser.add_argument("--min_sample_for_search", type=int, default=50)
    parser.add_argument("--do_umap", action="store_true")
    parser.add_argument("--no_write_back", action="store_true", help="如果设置，则不覆盖源 parquet，而写为 file.with_labels.parquet")
    parser.add_argument("--write_labels_to_output_dir", action="store_true", help="如果设置，则把写回文件写到 --output_dir （保留文件名或加 .with_labels.parquet）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    parquet_dir = args.parquet_dir

    # 1) 采样用于 silhouette 搜索（在原始维度或 PCA 后维度上）
    print("[STEP] Sampling embeddings for silhouette search ...")
    X_sample = sample_embeddings_for_silhouette(
        parquet_dir,
        sample_size=args.sample_size_for_silhouette,
        per_file_limit=args.per_file_sample_limit,
        columns=["embedding"],
        rng_seed=args.rng_seed,
        sampler_batch_size=4096,  # 如仍内存吃紧可调小
    )
    N_sample, dim = X_sample.shape
    print(f"[INFO] sampled X_sample shape = {X_sample.shape}")

    # 2) 如果需要 PCA，用 IncrementalPCA 对全量做 partial_fit（训练 transform）
    pca_transform = None
    if args.pca_dim and args.pca_dim > 0:
        print("[STEP] Fitting IncrementalPCA on full data (streaming) ...")
        try:
            pca_transform = incremental_fit_pca(parquet_dir, args.pca_dim, batch_size=args.batch_size_ipca)
            # 同时降维采样数据用于 silhouette 搜索（避免在原始维度上计算）
            try:
                X_sample_reduced = pca_transform.transform(X_sample)
            except Exception as e:
                print(f"[WARN] PCA transform on sample failed: {e}; using original X_sample")
                X_sample_reduced = X_sample
        except Exception as e:
            print(f"[WARN] IncrementalPCA failed: {e}; will use original X_sample for search")
            pca_transform = None
            X_sample_reduced = X_sample
    else:
        X_sample_reduced = X_sample

    # 3) silhouette 搜索 best_k（在采样或降维后的采样上）
    print("[STEP] Searching best_k by silhouette ...")
    try:
        best_k = search_best_k_by_silhouette(
            X_sample_reduced,
            args.k_min,
            args.k_max,
            args.rng_seed,
            sample_size=args.sample_size_for_silhouette,
            sample_frac=args.sample_frac_for_search,
            min_sample=args.min_sample_for_search
        )
    except Exception as e:
        print(f"[WARN] silhouette search failed: {e}; fallback best_k=8")
        best_k = 8
    print(f"[INFO] chosen best_k = {best_k}")

    # 4) 增量训练 MiniBatchKMeans（在可能的 PCA 空间中）
    print("[STEP] Incremental fitting MiniBatchKMeans on full data (streaming) ...")
    try:
        kmeans = incremental_fit_kmeans(
            parquet_dir,
            best_k,
            batch_size=args.batch_size_kmeans,
            pca_transform=pca_transform,
            rng_seed=args.rng_seed
        )
    except Exception as e:
        print(f"[ERROR] incremental kmeans failed: {e}")
        # 兜底：用单次 fit（在采样数据上）
        try:
            print("[INFO] Fallback: fit kmeans on sampled data")
            kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=args.rng_seed)
            kmeans.fit(X_sample_reduced)
        except Exception as e2:
            print(f"[ERROR] fallback kmeans on sample also failed: {e2}")
            kmeans = None

    # 5) 可选 UMAP（建议在采样上或降维采样上做）
    if args.do_umap and umap is not None and plt is not None:
        try:
            um = umap.UMAP(n_components=2, random_state=args.rng_seed)
            X_for_umap = X_sample_reduced if X_sample_reduced is not None else X_sample
            X2d = um.fit_transform(X_for_umap)
            # 若已训练 kmeans 且 X_for_umap 对应样本上能 predict
            labels_for_sample = None
            if kmeans is not None:
                try:
                    labels_for_sample = kmeans.predict(X_for_umap)
                except Exception:
                    pass
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(X2d[:, 0], X2d[:, 1], c=labels_for_sample if labels_for_sample is not None else None, s=6)
            ax.set_title(f"MiniBatchKMeans K={best_k}")
            if labels_for_sample is not None:
                plt.colorbar(sc, ax=ax)
            png = os.path.join(args.output_dir, "umap_2d.png")
            plt.savefig(png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] umap saved to {png}")
        except Exception as e:
            print(f"[WARN] umap failed: {e}")

    # 6) 逐文件 predict 并写回
    write_back = not args.no_write_back
    dst_dir = args.output_dir if args.write_labels_to_output_dir else None
    if kmeans is not None:
        print("[STEP] Predicting labels per file and writing back ...")
        predict_and_write_labels_by_file(
            parquet_dir,
            kmeans,
            batch_size=args.batch_size_kmeans,
            pca_transform=pca_transform,
            write_back=write_back,
            dst_dir=dst_dir
        )
    else:
        print("[WARN] kmeans 未能训练成功，跳过写回标签步骤")

    # 7) 保存 summary
    total_n = 0
    estimated_dim = int(dim) if 'dim' in locals() else -1
    for f in sorted([os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.lower().endswith(".parquet")]):
        try:
            pf = pq.ParquetFile(f)
            total_n += pf.metadata.num_rows
            if estimated_dim == -1:
                try:
                    # 仅取一个元素探测维度，避免整列读
                    for rb in pf.iter_batches(columns=["embedding"], batch_size=1):
                        col = rb.column(0)
                        a = _embedding_array_to_numpy(col)
                        if a.size > 0:
                            estimated_dim = a.shape[1]
                            break
                except Exception:
                    pass
        except Exception:
            pass

    res = {"best_k": int(best_k), "n_samples": int(total_n), "dim": int(estimated_dim)}
    res_path = os.path.join(args.output_dir, "result.json")
    with open(res_path, "w", encoding="utf-8") as wf:
        json.dump(res, wf, ensure_ascii=False, indent=2)
    print(f"[INFO] saved summary to {res_path}")

if __name__ == "__main__":
    main()
