import json
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import time
import random
from prompt import *

prompt = '''
You need to solve the following problem. I will provide you with a reference answer to help you write your own response.\
     However, do not mention that you referred to the reference answer in your reply. \
        Please explain your reasoning process and final answer in detail, without excessive omissions or skipping steps. \
            There is no need to use tags such as "<think>", "</think>", "<answer>", or "</answer>" in your output.
'''

def preprocess(messages):
    # assert len(messages) == 2
    history = ''
    for mes in messages:
        if mes['role'] == 'assistant':
            history += '【AI assistant】：' + mes['content']
        elif mes['role'] == 'user':
            history += '【user】：' + mes['content']
    input_prompt = base_prompt + history
    messages = [{"role":"user","content":input_prompt}]
    return messages

def preprocess2(messages):
    # assert len(messages) == 2
    history = ''
    for mes in messages:
        if mes['from'] == 'gpt':
            history += '【AI assistant】：' + mes['value']
        elif mes['from'] == 'human':
            history += '【user】：' + mes['value']
    input_prompt = base_prompt + history
    messages = [{"role":"user","content":input_prompt}]
    return messages

def forward_local_api(messages, max_retries=5, sleep_duration=60, model='Qwen2.5-3B-KA-20w-v2-format-ep1', temperature=1):
    if "from" in messages[0] or "value" in messages[0]:
        messages = preprocess2(messages)
    elif "role" in messages[0] or "content" in messages[0]:
        messages = preprocess(messages)
    port = random.choices(['8026'])
    client = OpenAI(
        api_key="your_api_key_here",
        base_url=f"http://10.16.80.150:{port[0]}/v1",
    )
    # print('messages',messages)
    for attempt in range(1, max_retries + 1):
        try:
            # print(f"Attempt {attempt}: Sending messages to model...")
            # print(messages)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                stop=["<|eot_id|>", "<|im_end|>", "</s>", "<|endoftext|>", "</answer>"],
                extra_body={"repetition_penalty": 1.05}
                # "chat_template_kwargs": {"enable_thinking": args.enable_thinking}},
            )
            result = completion.choices[0].message.content.strip("\n")
            prompt_token = completion.usage.prompt_tokens
            completion_token = completion.usage.completion_tokens
            print('200ok, results:::', result)
            return result, prompt_token, completion_token

        except Exception as e:
            print(f"Error on attempt {attempt}: {e}")
            return '', '', ''


def load_jsonl_to_list(file_path, dedup_key=None):
    data = []
    seen = set()
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if dedup_key and dedup_key in obj:
                    key_val = json.dumps(obj[dedup_key], ensure_ascii=False)
                    if key_val in seen:
                        continue
                    seen.add(key_val)
                data.append(obj)
            except json.JSONDecodeError:
                continue
    return data

def load_jsonl_to_set(file_path, key="conversations"):
    result_set = set()
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return result_set
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if key in obj:
                    result_set.add(json.dumps(obj[key], ensure_ascii=False))
            except json.JSONDecodeError:
                continue
    return result_set

def save_jsonl(save_path, result_list):
    parent_dir = os.path.dirname(save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(save_path, 'a', encoding='utf-8') as f:
        for item in result_list:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--dtype", type=str, choices=["int8", "float16"], default="float16")
    parser.add_argument("--presence_penalty", type=float, default=-1)
    parser.add_argument("--frequency_penalty", type=float, default=-1)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_type", type=str, default="alpaca")
    parser.add_argument("--model_type", type=str, default="alpaca")
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--num_return_sequences", type=int, default=8)
    parser.add_argument("--num_threads", type=int, default=25)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--user_reference", type=bool, default=False)
    parser.add_argument("--drop_last", type=bool, default=False)
    # parser.add_argument("--enable_thinking", type=bool, default=False)
    return parser.parse_args()

class LLMInference:
    def __init__(self, args):
        self.args = args

        raw_dataset = load_jsonl_to_list(args.data_path, dedup_key="conversations")
        print('raw_dataset',len(raw_dataset))
        saved_convs = load_jsonl_to_set(args.save_path, key="conversations")
        print('saved_convs',len(saved_convs))
        filtered_data = [
            item for item in raw_dataset
            if json.dumps(item.get("conversations"), ensure_ascii=False) not in saved_convs
        ]

        if args.max_samples:
            filtered_data = filtered_data[:args.max_samples]

        self.dataset = filtered_data

    def process(self):
        ds = self.dataset
        max_workers = getattr(self.args, "num_threads", None) or os.cpu_count()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(forward_local_api, item['conversations'], 1, 60, self.args.model): item
                for item in ds
            }
            for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Inference"):
                item = future_to_item[future]
                result,prompt_tokens,completion_tokens = future.result()
                item['outputs'] = result
                item['prompt_tokens'] = prompt_tokens
                item['completion_tokens'] = completion_tokens

                # 💾 立即保存每条结果
                if item['prompt_tokens'] == '':
                    continue 
                save_jsonl(self.args.save_path, [item])

if __name__ == "__main__":
    args = get_args()
    engine = LLMInference(args)
    print(f"*** Starting inference: max_samples={args.max_samples}, pending samples={len(engine.dataset)}")
    engine.process()
