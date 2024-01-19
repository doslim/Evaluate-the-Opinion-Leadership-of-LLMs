
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer, LlamaTokenizer
import datasets
import json


def read_jsonl(model_checkpoint, path, max_seq_length, prompt_key,target_key,skip_overlength=False):
    if 'llama' in model_checkpoint.lower() or 'alpaca' in model_checkpoint.lower():
        tokenizer = LlamaTokenizer.from_pretrained(
            model_checkpoint, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_checkpoint, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length,prompt_key,target_key)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


def preprocess(tokenizer, config, example, max_seq_length, prompt_key, target_key):
    prompt = example[prompt_key]
    target = example[target_key]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(target, max_length=max_seq_length, truncation=True, add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

if __name__ == '__main__':
    model_checkpoint = '../LLM_weight/Baichuan2-13B-chat'
    input_file= 'wwqa.json'
    prompt_key = 'Question'
    target_key = 'Answer'
    save_name = 'wwqa_baichuan2-13B'
    max_seq_length = 2000
    skip_overlength = False

    input_file_path = f'./WWQA/{input_file}'
    save_path = f"./WWQA/tokenized_data/{save_name}"
    dataset = datasets.Dataset.from_generator(
            lambda: read_jsonl(model_checkpoint, input_file_path, max_seq_length, prompt_key, target_key, skip_overlength)
    )

    dataset.save_to_disk(save_path)
