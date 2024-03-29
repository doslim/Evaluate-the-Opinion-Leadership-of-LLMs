from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, BitsAndBytesConfig
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import DataCollatorForLanguageModeling
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dataclasses import dataclass, field
import datasets
import os
import random


@dataclass
class FinetuneArguments:
    model_version: str = field(default="Baichuan2-7B")
    tokenized_dataset: str = field(default=" ") # 
    train_size: int = field(default=900) 
    eval_size: int = field(default=100) 
    lora_rank: int = field(default=8)
    previous_lora_weights: str = field(default=None) 
    no_prompt_loss: int = field(default=0) 


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)
    

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs =  model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        self.model.save_pretrained(output_dir)


class ModifiedTrainer_GLM(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs =  model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def my_data_collator(features: list) -> dict:

    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"] # prompt length
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


writer = SummaryWriter()
finetune_args, training_args = HfArgumentParser(
    (FinetuneArguments, TrainingArguments)
).parse_args_into_dataclasses()

if finetune_args.model_version == 'Baichuan2-7B':
    model_checkpoint = '../LLM_weight/Baichuan2-7B-chat'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint, load_in_8bit=True, trust_remote_code=True, 
    device_map="auto" 
    )
    print(model.hf_device_map)
elif finetune_args.model_version == 'Baichuan2-13B':
    model_checkpoint = '../LLM_weight/Baichuan2-13B-chat-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint, load_in_8bit=True, trust_remote_code=True, 
    device_map="auto" 
    )
    print(model.hf_device_map)
elif finetune_args.model_version == 'Mistral-7B':
    model_checkpoint = '../LLM_weight/Mistral-7B-Instruct-v0.2'
    quantization_config = BitsAndBytesConfig(load_in_8bit = True)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map = 'auto', quantization_config = quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.pad_token = tokenizer.unk_token
elif finetune_args.model_version == 'chatglm3-6b':
    model_checkpoint = '../LLM_weight/chatglm3-6b'
    model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True)
    # model = model.quantize(8)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
elif finetune_args.model_version == 'internlm2-20b':
    model_checkpoint = '../LLM_weight/internlm2-chat-20b'
    quantization_config = BitsAndBytesConfig(load_in_8bit = True)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map = 'auto', quantization_config = quantization_config, trust_remote_code = True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token


if finetune_args.no_prompt_loss:
    print("*** If you see this message, ***")
    print("*** it means: Prompt is not calculated in loss. ***")
    data_collator = my_data_collator
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)


dataset = datasets.load_from_disk('WWQA/tokenized_data/' + finetune_args.tokenized_dataset)

dataset_size = len(dataset)
train_size = int(finetune_args.train_size)
eval_size = dataset_size - train_size
data_index = list(range(dataset_size))
random.shuffle(data_index)
train_dataset = dataset.select(data_index[:train_size]) 
eval_dataset = dataset.select(data_index[-eval_size:]) 
print(f"train: {len(train_dataset)}")
print(f"evaluation: {len(eval_dataset)}")


if 'chatglm3' in finetune_args.model_version or 'internlm' in finetune_args.model_version:
    model.gradient_checkpointing_enable() 
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = False 
else:
    model.gradient_checkpointing_enable() 
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.lm_head = CastOutputToFloat(model.lm_head)


if finetune_args.previous_lora_weights == None:
    if 'Baichuan' in finetune_args.model_version:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules = ["W_pack"] 
        )
    elif 'Mistral' in finetune_args.model_version:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
    elif 'chatglm3' in finetune_args.model_version:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1
        )
    elif 'internlm' in  finetune_args.model_version:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules = ['wqkv', 'wo', 'w1', 'w2', 'w3']
        )
    model = get_peft_model(model, peft_config)

else:
    model = PeftModel.from_pretrained(model, finetune_args.previous_lora_weights)
    # see: https://github.com/huggingface/peft/issues/184
    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True

model.save_pretrained(training_args.output_dir)
if 'chatglm3' in finetune_args.model_version:
    trainer = ModifiedTrainer_GLM(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    callbacks=[TensorBoardCallback(writer)],
    data_collator=data_collator,
)
else:
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )

trainer.train()
writer.close()
# save model
model.save_pretrained(training_args.output_dir)