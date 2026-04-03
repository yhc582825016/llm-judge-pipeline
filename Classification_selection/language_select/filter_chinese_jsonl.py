#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a large jsonl file, judge each user turn in messages, "
            "and keep Chinese conversations with resume support."
        )
    )
    parser.add_argument("--input", required=True, help="Input jsonl path")
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument(
        "--state-path",
        default=None,
        help="Checkpoint state path. Defaults to <output>.resume.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved byte offset and append to output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output/state when not resuming",
    )
    parser.add_argument(
        "--method",
        choices=["regex", "fasttext", "lingua"],
        default="regex",
        help="Language detection backend",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.05,
        help="Minimum CJK ratio for regex mode",
    )
    parser.add_argument(
        "--min-cjk-count",
        type=int,
        default=4,
        help="Minimum number of Chinese characters for regex mode",
    )
    parser.add_argument(
        "--fasttext-model",
        default=None,
        help="Path to fastText lid model, such as lid.176.bin or lid.176.ftz",
    )
    parser.add_argument(
        "--fasttext-threshold",
        type=float,
        default=0.50,
        help="Minimum fastText probability for Chinese label",
    )
    parser.add_argument(
        "--lingua-min-confidence",
        type=float,
        default=0.50,
        help="Minimum Lingua confidence for Chinese",
    )
    parser.add_argument(
        "--min-text-length",
        type=positive_float,
        default=1.0,
        help="Skip turns shorter than this length after strip",
    )
    parser.add_argument(
        "--user-policy",
        choices=["all", "any"],
        default="all",
        help="Keep sample if all/any user turns satisfy the Chinese rule",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save checkpoint every N processed lines",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10000,
        help="Print progress every N processed lines",
    )
    return parser.parse_args()


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item.get("content"), str):
                parts.append(item["content"])
        return "\n".join(parts)
    return ""


def extract_user_turns(record: Dict[str, Any]) -> List[str]:
    messages = record.get("messages", [])
    user_turns: List[str] = []
    if not isinstance(messages, list):
        return user_turns
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        text = extract_text(msg.get("content", ""))
        if text.strip():
            user_turns.append(text)
    return user_turns


def chinese_stats(text: str) -> Tuple[int, int, float]:
    if not text:
        return 0, 0, 0.0
    cjk_count = len(CJK_RE.findall(text))
    text_len = len(text)
    ratio = cjk_count / text_len if text_len else 0.0
    return cjk_count, text_len, ratio


def is_chinese_turn_regex(text: str, min_ratio: float, min_cjk_count: int) -> bool:
    cjk_count, _, ratio = chinese_stats(text)
    return cjk_count >= min_cjk_count and ratio >= min_ratio


class RegexChineseDetector:
    def __init__(self, min_ratio: float, min_cjk_count: int) -> None:
        self.min_ratio = min_ratio
        self.min_cjk_count = min_cjk_count

    def is_chinese(self, text: str) -> bool:
        return is_chinese_turn_regex(
            text, min_ratio=self.min_ratio, min_cjk_count=self.min_cjk_count
        )


class FastTextChineseDetector:
    def __init__(self, model_path: str, threshold: float) -> None:
        try:
            import fasttext
        except ImportError as exc:
            raise ImportError(
                "fasttext is not installed. Please run `pip install fasttext`."
            ) from exc
        self.model = fasttext.load_model(model_path)
        self.threshold = threshold

    def is_chinese(self, text: str) -> bool:
        labels, probs = self.model.predict(text.replace("\n", " "), k=3)
        for label, prob in zip(labels, probs):
            normalized = label.replace("__label__", "").lower()
            if normalized.startswith("zh") and prob >= self.threshold:
                return True
        return False


class LinguaChineseDetector:
    def __init__(self, min_confidence: float) -> None:
        try:
            from lingua import Language, LanguageDetectorBuilder
        except ImportError as exc:
            raise ImportError(
                "lingua-language-detector is not installed. "
                "Please run `pip install lingua-language-detector`."
            ) from exc
        self.language = Language.CHINESE
        self.detector = LanguageDetectorBuilder.from_languages(
            Language.CHINESE,
            Language.ENGLISH,
            Language.JAPANESE,
            Language.KOREAN,
            Language.FRENCH,
            Language.GERMAN,
            Language.SPANISH,
            Language.RUSSIAN,
        ).build()
        self.min_confidence = min_confidence

    def is_chinese(self, text: str) -> bool:
        detected = self.detector.detect_language_of(text)
        if detected != self.language:
            return False
        confidence = self.detector.compute_language_confidence_values(text)
        for item in confidence:
            if item.language == self.language and item.value >= self.min_confidence:
                return True
        return False


def build_detector(args: argparse.Namespace) -> Any:
    if args.method == "regex":
        return RegexChineseDetector(
            min_ratio=args.min_ratio,
            min_cjk_count=args.min_cjk_count,
        )
    if args.method == "fasttext":
        if not args.fasttext_model:
            raise ValueError("--fasttext-model is required when --method fasttext")
        return FastTextChineseDetector(
            model_path=args.fasttext_model,
            threshold=args.fasttext_threshold,
        )
    return LinguaChineseDetector(min_confidence=args.lingua_min_confidence)


def build_progress_bar(total_bytes: int, initial_bytes: int) -> Any:
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(
        total=total_bytes,
        initial=initial_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Filtering",
        dynamic_ncols=True,
    )


def should_keep_record(
    user_turns: List[str], detector: Any, user_policy: str, min_text_length: float
) -> bool:
    if not user_turns:
        return False
    turn_flags = []
    for text in user_turns:
        if len(text.strip()) < min_text_length:
            turn_flags.append(False)
            continue
        turn_flags.append(detector.is_chinese(text))
    if user_policy == "all":
        return all(turn_flags)
    return any(turn_flags)


def default_state_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".resume.json")


def load_state(state_path: Path) -> Optional[Dict[str, Any]]:
    if not state_path.exists():
        return None
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    tmp_path = state_path.with_name(state_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, state_path)


def build_initial_state(args: argparse.Namespace, input_path: Path, output_path: Path) -> Dict[str, Any]:
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "offset_bytes": 0,
        "processed_lines": 0,
        "kept_lines": 0,
        "invalid_lines": 0,
        "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_size_bytes": input_path.stat().st_size,
        "method": args.method,
        "min_ratio": args.min_ratio,
        "min_cjk_count": args.min_cjk_count,
        "fasttext_model": args.fasttext_model,
        "fasttext_threshold": args.fasttext_threshold,
        "lingua_min_confidence": args.lingua_min_confidence,
        "min_text_length": args.min_text_length,
        "user_policy": args.user_policy,
    }


def validate_resume_state(
    state: Dict[str, Any], args: argparse.Namespace, input_path: Path, output_path: Path
) -> None:
    expected = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "method": args.method,
        "min_ratio": args.min_ratio,
        "min_cjk_count": args.min_cjk_count,
        "fasttext_model": args.fasttext_model,
        "fasttext_threshold": args.fasttext_threshold,
        "lingua_min_confidence": args.lingua_min_confidence,
        "min_text_length": args.min_text_length,
        "user_policy": args.user_policy,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"Resume state mismatch for {key}: state={state.get(key)!r}, arg={value!r}"
            )


def open_output(output_path: Path, resume: bool) -> Any:
    mode = "a" if resume else "w"
    return output_path.open(mode, encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    state_path = Path(args.state_path) if args.state_path else default_state_path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.resume:
        state = load_state(state_path)
        if state is None:
            raise FileNotFoundError(f"Resume requested but state file not found: {state_path}")
        if not output_path.exists():
            raise FileNotFoundError(f"Resume requested but output file not found: {output_path}")
        validate_resume_state(state, args, input_path, output_path)
    else:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite or --resume."
            )
        if state_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"State already exists: {state_path}. Use --overwrite or --resume."
            )
        state = build_initial_state(args, input_path, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    detector = build_detector(args)

    start_time = time.time()
    processed_lines = int(state["processed_lines"])
    kept_lines = int(state["kept_lines"])
    invalid_lines = int(state["invalid_lines"])
    offset_bytes = int(state["offset_bytes"])
    total_bytes = int(state["input_size_bytes"])
    progress_bar = build_progress_bar(total_bytes=total_bytes, initial_bytes=offset_bytes)

    try:
        with input_path.open("rb") as fin, open_output(output_path, resume=args.resume) as fout:
            fin.seek(offset_bytes)

            while True:
                raw_line = fin.readline()
                if not raw_line:
                    break

                next_offset = fin.tell()
                processed_lines += 1

                if progress_bar is not None:
                    progress_bar.update(next_offset - offset_bytes)
                    progress_bar.set_postfix(
                        processed=processed_lines,
                        kept=kept_lines,
                        invalid=invalid_lines,
                    )
                    offset_bytes = next_offset

                try:
                    line = raw_line.decode("utf-8")
                    record = json.loads(line)
                except Exception:
                    invalid_lines += 1
                    state.update(
                        {
                            "offset_bytes": next_offset,
                            "processed_lines": processed_lines,
                            "kept_lines": kept_lines,
                            "invalid_lines": invalid_lines,
                            "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                    if processed_lines % args.save_every == 0:
                        save_state(state_path, state)
                    continue

                user_turns = extract_user_turns(record)
                if should_keep_record(
                    user_turns=user_turns,
                    detector=detector,
                    user_policy=args.user_policy,
                    min_text_length=args.min_text_length,
                ):
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept_lines += 1

                state.update(
                    {
                        "offset_bytes": next_offset,
                        "processed_lines": processed_lines,
                        "kept_lines": kept_lines,
                        "invalid_lines": invalid_lines,
                        "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                if processed_lines % args.save_every == 0:
                    fout.flush()
                    save_state(state_path, state)

                if processed_lines % args.log_every == 0:
                    elapsed = time.time() - start_time
                    speed = processed_lines / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[progress] processed={processed_lines} kept={kept_lines} "
                        f"invalid={invalid_lines} offset={next_offset} speed={speed:.2f} lines/s",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    state.update(
        {
            "offset_bytes": input_path.stat().st_size,
            "processed_lines": processed_lines,
            "kept_lines": kept_lines,
            "invalid_lines": invalid_lines,
            "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed": True,
        }
    )
    save_state(state_path, state)

    elapsed = time.time() - start_time
    speed = processed_lines / elapsed if elapsed > 0 else 0.0
    print(
        f"[done] processed={processed_lines} kept={kept_lines} invalid={invalid_lines} "
        f"elapsed={elapsed:.2f}s speed={speed:.2f} lines/s",
        file=sys.stderr,
    )
    print(f"[done] output={output_path}", file=sys.stderr)
    print(f"[done] state={state_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
