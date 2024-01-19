from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from zhipuai import ZhipuAI
import random
import json
from sklearn.metrics import classification_report
import pandas as pd
import os
from peft import PeftModel


def get_prediction(model, tokenizer, prompt, temperature = 0.1, do_sample = True, local_model = True):
    message = []
    message.append({'role': 'user', 'content': prompt})
    if local_model:
        model.generation_config.temperature = temperature
        model.generation_config.do_sample = do_sample
        response_text = model.chat(tokenizer, message)
        if 'Yes' in response_text:
            return 1
        elif 'No' in response_text:
            return 0
        else:
            return random.choice([0,1])
    else:
        response =  model.chat.completions.create(
                        model = "glm-4",  # glm-3-turbo
                        messages = message,
                        do_sample = do_sample,
                        temperature = temperature
                    )
        response_text = response.choices[0].message.content
        if 'Yes' in response_text:
            return 1
        elif 'No' in response_text:
            return 0
        else:
            return random.choice([0,1])
        


def eval_on_wwqa(model_path, data_path, result_path, local_model = True, api_key = None, peft_path = None):
    if local_model:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, load_in_8bit = True, device_map = 'auto')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if peft_path is not None:
            model = PeftModel.from_pretrained(model, peft_path)

        label = []
        prediction = []
        with open(data_path, "r") as f:
            for line in tqdm(f.readlines()):
                data_sample = json.loads(line)
                prompt = data_sample['Question']
                answer = data_sample['Answer']
                if 'Yes' in answer:
                    label.append(1)
                else:
                    label.append(0)
                pred = get_prediction(model, tokenizer, prompt, do_sample = False)
                prediction.append(pred)

        results = pd.DataFrame(classification_report(label, prediction, digits=4, output_dict=True)).T
    
    else:
        model = ZhipuAI(api_key=api_key)
        tokenizer = None
        label = []
        prediction = []
        with open(data_path, "r") as f:
            for line in tqdm(f.readlines()):
                data_sample = json.loads(line)
                prompt = data_sample['Question']
                answer = data_sample['Answer']
                if 'Yes' in answer:
                    label.append(1)
                else:
                    label.append(0)
                pred = get_prediction(model, tokenizer, prompt, do_sample = False, local_model = False)
                prediction.append(pred)

        results = pd.DataFrame(classification_report(label, prediction, digits=4, output_dict=True)).T
    
    results.to_csv(os.path.join(result_path, 'wwqa_result.csv'))


if __name__ == '__main__':
    data_path = './WWQA/binary_qa.json'
    model_path = '../LLM_weight/Baichuan2-13B-chat/'
    result_path = './results/Baichuan2-13B-chat-FT'
    peft_path = './weights/wwqa_train_1453_baichuan2-13B'
    local_model = True

    # model_path = None
    # result_path = './results/GLM-4'
    api_key = ' '
    # local_model = False
    # peft_path = None


    eval_on_wwqa(model_path, data_path, result_path, local_model, api_key = api_key, peft_path = peft_path)