import argparse
import ast
import json
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = "/mnt/code/yehangcheng/Intruct_augment/gen_data/agent_syn_data/synthetic_mock_success_only_2.jsonl"
DEFAULT_OUTPUT = "/mnt/code/yehangcheng/Intruct_augment/pipline/agent_rl_syn/syn_data/train_filtered_from_mock_2.parquet"


TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
}


def annotation_to_json_type(annotation: ast.AST | None) -> str:
    if annotation is None:
        return "string"

    if isinstance(annotation, ast.Name):
        return TYPE_MAP.get(annotation.id, "string")

    if isinstance(annotation, ast.Subscript):
        base = annotation.value
        if isinstance(base, ast.Name):
            base_name = base.id
            if base_name in {"list", "List"}:
                return "array"
            if base_name in {"dict", "Dict"}:
                return "object"
        return "string"

    if isinstance(annotation, ast.Attribute):
        return TYPE_MAP.get(annotation.attr, "string")

    return "string"


def extract_tool_schemas(module_code: str) -> list[dict]:
    tree = ast.parse(module_code)
    schemas = []

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_") or node.name == "run_demo":
            continue

        description = ast.get_docstring(node) or f"Auto-generated schema for {node.name}."
        properties = {}
        required = []

        positional_args = list(node.args.args)
        defaults = list(node.args.defaults)
        default_offset = len(positional_args) - len(defaults)

        for idx, arg in enumerate(positional_args):
            if arg.arg == "self":
                continue

            properties[arg.arg] = {
                "type": annotation_to_json_type(arg.annotation),
                "description": f"Parameter `{arg.arg}` for `{node.name}`.",
            }
            if idx < default_offset:
                required.append(arg.arg)

        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": node.name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )

    return schemas


def build_question(qa: dict) -> str:
    prompt = qa["PROMPT"].strip()
    output_format = qa.get("OUTPUT_FORMAT", "").strip()
    if not output_format:
        return prompt
    return (
        f"{prompt}\n\n"
        f"请将最终答案写成以下格式：\n"
        f"{output_format}\n"
        f"请严格保持格式中的符号一致，包括中英文逗号、括号和空格。"
    )


def extract_box_content(answer: str) -> str:
    answer = answer.strip()
    match = re.fullmatch(r"//box\{(.*)\}", answer, flags=re.DOTALL)
    if match:
        return match.group(1)
    return answer


def convert_record(record: dict, data_source: str, ability: str) -> dict:
    qa = record["qa"]
    tool_schemas = extract_tool_schemas(record["module_code"])
    extra_info_obj = {
        "env": record["module_code"],
        "func_schemas": json.dumps(tool_schemas, ensure_ascii=False),
        "index": str(uuid.uuid4()),
        "tool_schemas": tool_schemas,
    }

    return {
        "data_source": data_source,
        "question": build_question(qa),
        "ability": ability,
        "reward_model": {
            "ground_truth": np.array([extract_box_content(qa["ANSWER"])], dtype=object),
            "style": "rule",
        },
        "extra_info": json.dumps(extra_info_obj, ensure_ascii=False),
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_file(input_path: Path, output_path: Path, data_source: str, ability: str) -> pd.DataFrame:
    raw_records = load_jsonl(input_path)
    converted = [convert_record(record, data_source=data_source, ability=ability) for record in raw_records]
    df = pd.DataFrame(converted, columns=["data_source", "question", "ability", "reward_model", "extra_info"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert mock tool-call JSONL into recall-agent parquet format.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to source JSONL file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output parquet file.")
    parser.add_argument("--data-source", default="syntool_re_call", help="Value for data_source column.")
    parser.add_argument("--ability", default="re_call", help="Value for ability column.")
    args = parser.parse_args()

    df = convert_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        data_source=args.data_source,
        ability=args.ability,
    )
    print(f"Converted {len(df)} rows to {args.output}")
    print(df.head(2).to_dict(orient="records"))


if __name__ == "__main__":
    main()
