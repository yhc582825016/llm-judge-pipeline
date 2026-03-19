#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parent


def _now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _echo_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _model_tag(model_name_or_path: str) -> str:
    raw = Path(str(model_name_or_path).rstrip("/")).name or str(model_name_or_path)
    tag = re.sub(r"[^0-9A-Za-z._-]+", "_", raw).strip("._-")
    return tag or "unknown_model"


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print(f"\n[RUN] {_echo_cmd(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def try_parse_json_obj(text: Any) -> Optional[Dict[str, Any]]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None

    s = text.strip()
    if not s:
        return None

    candidates = [s]
    candidates.extend(re.findall(r"\{.*?\}", s, flags=re.S))

    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            try:
                obj = json.loads(c.replace("'", '"'))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def extract_score(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)

    obj = try_parse_json_obj(x)
    if obj is not None:
        for key in ("score", "rating", "overall_score", "value"):
            if key in obj:
                v = parse_float(obj.get(key))
                if v is not None:
                    return v

    s = str(x) if x is not None else ""
    if not s.strip():
        return None

    patterns = [
        r"\[\[\s*(-?\d+(?:\.\d+)?)\s*\]\]",
        r"<score>\s*(-?\d+(?:\.\d+)?)\s*</score>",
        r'"(?:score|rating)"\s*:\s*"?(-?\d+(?:\.\d+)?)"?',
        r"(?:Rating|rating)\s*[:：]\s*(-?\d+(?:\.\d+)?)",
    ]
    for p in patterns:
        m = re.search(p, s, flags=re.I)
        if m:
            return float(m.group(1))

    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if len(nums) == 1:
        return float(nums[0])
    return None


def filter_by_score(
    input_path: Path,
    output_path: Path,
    score_col: str,
    min_score: Optional[float],
    max_score: Optional[float],
    dry_run: bool = False,
) -> Dict[str, int]:
    print(f"\n[INFO] 评分过滤: {input_path} -> {output_path}")
    if dry_run:
        return {"total": 0, "parsed": 0, "kept": 0}

    df = pd.read_parquet(input_path)
    if score_col not in df.columns:
        raise ValueError(f"评分列不存在: {score_col}，可选列: {list(df.columns)}")

    df = df.copy()
    df["judge_score"] = df[score_col].apply(extract_score)

    mask = df["judge_score"].notna()
    if min_score is not None:
        mask &= df["judge_score"] >= min_score
    if max_score is not None:
        mask &= df["judge_score"] <= max_score

    kept = df[mask].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(output_path, index=False)

    stats = {
        "total": int(len(df)),
        "parsed": int(df["judge_score"].notna().sum()),
        "kept": int(len(kept)),
    }
    print(f"[INFO] 评分过滤完成: total={stats['total']}, parsed={stats['parsed']}, kept={stats['kept']}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="文本数据蒸馏全流程总控脚本")

    # 基本输入输出
    p.add_argument("--input", required=True, help="原始数据，支持 json/jsonl/parquet")
    p.add_argument("--run-name", default=None, help="运行名称，不传则自动按时间生成")
    p.add_argument("--work-dir", default=str(ROOT / "runs"), help="流水线输出根目录")
    p.add_argument("--python-bin", default=sys.executable, help="用于调用子脚本的 Python")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="即使产物已存在也强制重跑每一步")

    # 阶段开关
    p.add_argument("--skip-embedding", action="store_true")
    p.add_argument("--skip-dedup", action="store_true")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--skip-filter", action="store_true")
    p.add_argument("--skip-inference", action="store_true")

    # Embedding
    p.add_argument("--embedding-model-path", required=True)
    p.add_argument("--embedding-batch-size", type=int, default=16)
    p.add_argument("--embedding-max-input-tokens", type=int, default=8192)
    p.add_argument("--embedding-tensor-parallel-size", type=int, default=1)
    p.add_argument("--embedding-source", default="")
    p.add_argument("--embedding-source-field", default="")
    p.add_argument("--embedding-concat-user-turns", action="store_true")

    # Dedup
    p.add_argument("--dedup-threshold", type=float, default=0.9)
    p.add_argument("--dedup-device", default="cuda")
    p.add_argument("--dedup-chunk-db", type=int, default=10000)
    p.add_argument("--dedup-chunk-q", type=int, default=128)

    # Judge
    p.add_argument("--judge-use-openai-client", dest="judge_use_openai_client", action="store_true", default=True)
    p.add_argument("--judge-no-openai-client", dest="judge_use_openai_client", action="store_false")
    p.add_argument("--judge-api-key", default=None)
    p.add_argument("--judge-base-url", default="http://localhost:8035/v1")
    p.add_argument("--judge-model", default="qwen3-32b")
    p.add_argument("--judge-concurrency", type=int, default=64)
    p.add_argument("--judge-timeout", type=float, default=30.0)
    p.add_argument("--judge-max-retries", type=int, default=5)
    p.add_argument("--judge-sleep-duration", type=float, default=5.0)
    p.add_argument("--judge-field", choices=["prompt", "response"], default="prompt")
    p.add_argument("--judge-prompt-key", default="rate_prompt")
    p.add_argument("--judge-prompt-map-file", default=str(ROOT / "prompt.py"))
    p.add_argument("--judge-extra-body", default=None)

    # Filter
    p.add_argument("--score-column", default="sglang_result")
    p.add_argument("--min-score", type=float, default=7.0)
    p.add_argument("--max-score", type=float, default=10.0)

    # Inference
    p.add_argument("--infer-use-openai-client", dest="infer_use_openai_client", action="store_true", default=True)
    p.add_argument("--infer-no-openai-client", dest="infer_use_openai_client", action="store_false")
    p.add_argument("--infer-api-key", default=None)
    p.add_argument("--infer-base-url", default="http://localhost:8035/v1")
    p.add_argument("--infer-model", default="qwen3-32b")
    p.add_argument("--infer-concurrency", type=int, default=64)
    p.add_argument("--infer-timeout", type=float, default=30.0)
    p.add_argument("--infer-max-retries", type=int, default=5)
    p.add_argument("--infer-sleep-duration", type=float, default=5.0)
    p.add_argument("--infer-field", choices=["prompt", "response"], default="prompt")
    p.add_argument("--infer-num-samples", type=int, default=1)
    p.add_argument("--infer-extra-body", default=None)
    p.add_argument("--infer-use-multi-turn", action="store_true")
    p.add_argument("--infer-max-turns", type=int, default=0)

    args = p.parse_args()

    run_name = args.run_name or _now_str()
    run_dir = Path(args.work_dir).resolve() / run_name
    embedding_model_tag = _model_tag(args.embedding_model_path)
    infer_model_tag = _model_tag(args.infer_model)

    emb_dir = ROOT / "emb_shards" / embedding_model_tag
    dedup_dir = ROOT / "decup_result" / embedding_model_tag
    judge_dir = run_dir / "judge"
    infer_dir = ROOT / "inference_res" / infer_model_tag

    kept_parquet = dedup_dir / f"{run_name}_kept.parquet"
    removed_parquet = dedup_dir / f"{run_name}_removed.parquet"
    judged_parquet = judge_dir / "judged.parquet"
    filtered_parquet = judge_dir / "filtered.parquet"
    generated_parquet = infer_dir / f"{run_name}_generated.parquet"

    for d in (run_dir, emb_dir, dedup_dir, judge_dir, infer_dir):
        d.mkdir(parents=True, exist_ok=True)

    embedding_py = ROOT / "embedding.py"
    dedup_py = ROOT / "embedding_decup.py"
    judge_py = ROOT / "llm_judge.py"
    infer_py = ROOT / "llm_inference.py"

    # 1) Embedding
    if not args.skip_embedding:
        emb_out = emb_dir / "shard_0.parquet"
        if emb_out.exists() and not args.force:
            print(f"[SKIP] embedding 已存在: {emb_out}")
        else:
            cmd = [
                args.python_bin,
                str(embedding_py),
                "--input",
                args.input,
                "--model_path",
                args.embedding_model_path,
                "--out_dir",
                str(emb_dir),
                "--embedding_batch_size",
                str(args.embedding_batch_size),
                "--max_input_tokens",
                str(args.embedding_max_input_tokens),
                "--tensor_parallel_size",
                str(args.embedding_tensor_parallel_size),
            ]
            if args.embedding_source:
                cmd += ["--source", args.embedding_source]
            if args.embedding_source_field:
                cmd += ["--source-field", args.embedding_source_field]
            if args.embedding_concat_user_turns:
                cmd += ["--concat-user-turns"]
            run_cmd(cmd, dry_run=args.dry_run)

    # 2) Dedup
    if not args.skip_dedup:
        if kept_parquet.exists() and not args.force:
            print(f"[SKIP] dedup 已存在: {kept_parquet}")
        else:
            cmd = [
                args.python_bin,
                str(dedup_py),
                "--mode",
                "self_dedup",
                "--train_emb_dir",
                str(emb_dir),
                "--out",
                str(kept_parquet),
                "--dump_removed",
                str(removed_parquet),
                "--threshold",
                str(args.dedup_threshold),
                "--device",
                args.dedup_device,
                "--chunk_db",
                str(args.dedup_chunk_db),
                "--chunk_q",
                str(args.dedup_chunk_q),
            ]
            run_cmd(cmd, dry_run=args.dry_run)

    # 3) Judge
    if not args.skip_judge:
        if judged_parquet.exists() and not args.force:
            print(f"[SKIP] judge 已存在: {judged_parquet}")
        else:
            cmd = [
                args.python_bin,
                str(judge_py),
                "-i",
                str(kept_parquet),
                "-o",
                str(judged_parquet),
                "--field",
                args.judge_field,
                "--base-url",
                args.judge_base_url,
                "--model",
                args.judge_model,
                "--concurrency",
                str(args.judge_concurrency),
                "--timeout",
                str(args.judge_timeout),
                "--max-retries",
                str(args.judge_max_retries),
                "--sleep-duration",
                str(args.judge_sleep_duration),
                "--prompt-key",
                args.judge_prompt_key,
                "--prompt-map-file",
                args.judge_prompt_map_file,
            ]
            if args.judge_use_openai_client:
                cmd += ["--use-openai-client"]
            if args.judge_api_key:
                cmd += ["--api-key", args.judge_api_key]
            if args.judge_extra_body:
                cmd += ["--extra-body", args.judge_extra_body]
            run_cmd(cmd, dry_run=args.dry_run)

    # 4) Filter
    filter_stats: Dict[str, int] = {"total": 0, "parsed": 0, "kept": 0}
    if not args.skip_filter:
        if filtered_parquet.exists() and not args.force:
            print(f"[SKIP] filter 已存在: {filtered_parquet}")
        else:
            filter_stats = filter_by_score(
                input_path=judged_parquet,
                output_path=filtered_parquet,
                score_col=args.score_column,
                min_score=args.min_score,
                max_score=args.max_score,
                dry_run=args.dry_run,
            )

    # 5) Inference
    if not args.skip_inference:
        infer_input = filtered_parquet if not args.skip_filter else judged_parquet
        if generated_parquet.exists() and not args.force:
            print(f"[SKIP] inference 已存在: {generated_parquet}")
        else:
            cmd = [
                args.python_bin,
                str(infer_py),
                "-i",
                str(infer_input),
                "-o",
                str(generated_parquet),
                "--field",
                args.infer_field,
                "--base-url",
                args.infer_base_url,
                "--model",
                args.infer_model,
                "--concurrency",
                str(args.infer_concurrency),
                "--timeout",
                str(args.infer_timeout),
                "--max-retries",
                str(args.infer_max_retries),
                "--sleep-duration",
                str(args.infer_sleep_duration),
                "--num-samples",
                str(args.infer_num_samples),
            ]
            if args.infer_use_openai_client:
                cmd += ["--use-openai-client"]
            if args.infer_api_key:
                cmd += ["--api-key", args.infer_api_key]
            if args.infer_extra_body:
                cmd += ["--extra-body", args.infer_extra_body]
            if args.infer_use_multi_turn:
                cmd += ["--use-multi-turn", "--max-turns", str(args.infer_max_turns)]
            run_cmd(cmd, dry_run=args.dry_run)

    manifest = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "input": str(Path(args.input).resolve()),
        "outputs": {
            "embedding_dir": str(emb_dir),
            "dedup_kept": str(kept_parquet),
            "dedup_removed": str(removed_parquet),
            "judged": str(judged_parquet),
            "filtered": str(filtered_parquet),
            "generated": str(generated_parquet),
        },
        "filter_stats": filter_stats,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = run_dir / "manifest.json"
    if args.dry_run:
        print(f"\n[DRY-RUN] manifest 将写入: {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[DONE] 流程完成，manifest: {manifest_path}")


if __name__ == "__main__":
    main()
