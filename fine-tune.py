# Fine-tune the LLM

from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dataclasses import dataclass, field
import datasets
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

@dataclass
class FinetuneArguments:
    model_version: str = field(default="chat-7b")
    tokenized_dataset: str = field(default=" ") 
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


writer = SummaryWriter()
finetune_args, training_args = HfArgumentParser(
    (FinetuneArguments, TrainingArguments)
).parse_args_into_dataclasses()

if finetune_args.model_version == 'chat-7b':
    model_checkpoint = '../LLM_weight/Baichuan2-7B-chat'
else:
    model_checkpoint = '../LLM_weight/Baichuan2-13B-chat'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token

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


if finetune_args.no_prompt_loss:
    print("*** If you see this message, ***")
    print("*** it means: Prompt is not calculated in loss. ***")
    data_collator = my_data_collator
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)


dataset = datasets.load_from_disk('WWQA/tokenized_data/' + finetune_args.tokenized_dataset)
train_dataset = dataset.select(range(finetune_args.train_size))  
eval_dataset = dataset.select(list(range(len(dataset)))[-finetune_args.eval_size:])  
print(f"train: {len(train_dataset)}")
print(f"evaluation: {len(eval_dataset)}")

# init model
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint, load_in_8bit=False, trust_remote_code=True, 
    device_map="auto" 
)
print(model.hf_device_map)

model.gradient_checkpointing_enable() 
model.enable_input_require_grads()
model.lm_head = CastOutputToFloat(model.lm_head)


if finetune_args.previous_lora_weights == None:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules = ["W_pack"]  
    )
    
    model = get_peft_model(model, peft_config)
else:
    model = PeftModel.from_pretrained(model, finetune_args.previous_lora_weights)
    # see: https://github.com/huggingface/peft/issues/184
    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True

model.save_pretrained(training_args.output_dir)
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