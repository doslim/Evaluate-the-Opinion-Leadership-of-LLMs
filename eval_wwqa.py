from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModel
from tqdm import tqdm
from zhipuai import ZhipuAI
from openai import OpenAI
import random
import json
from sklearn.metrics import classification_report
import pandas as pd
import os
from peft import PeftModel


def get_prediction(model, tokenizer, prompt, temperature = 0.1, do_sample = True, model_name = "Baichuan"):
    message = []
    message.append({'role': 'user', 'content': prompt})
    if model_name == 'Baichuan':
        model.generation_config.temperature = temperature
        model.generation_config.do_sample = do_sample
        response_text = model.chat(tokenizer, message, max_new_tokens = 20)
    elif model_name == 'Yi':
        message = []
        message.append({'role': 'user', 'content': prompt})
        chat_data = tokenizer.apply_chat_template(message, tokenize = False)
        output = model(chat_data, do_sample = do_sample, temperature = temperature, max_length = 4096)
        response_text = output[0]['generated_text'].replace(chat_data, '').strip()
    elif model_name == 'Mistral':
        message = []
        message.append({'role': 'user', 'content': prompt})
        tokenized_chat = tokenizer.apply_chat_template(message, tokenize=True, return_tensors="pt")
        tokenized_chat = tokenized_chat.to('cuda')
        model.generation_config.do_sample = do_sample
        model.generation_config.temperature = temperature
        model.generation_config.max_new_tokens = 20
        response = model.generate(input_ids = tokenized_chat)
        response_text = tokenizer.decode(response[0,tokenized_chat.shape[1]:])
    elif model_name == 'chatglm3':
        response_text, _ = model.chat(tokenizer, prompt, do_sample = do_sample, temperature = temperature)
    elif model_name == 'internlm':
        response_text, _ = model.chat(tokenizer, prompt, do_sample = do_sample, temperature = temperature)
    elif model_name == 'glm-4':
        response =  model.chat.completions.create(
                        model = "glm-4",  # glm-3-turbo
                        messages = message,
                        do_sample = do_sample,
                        temperature = temperature
                    )
        response_text = response.choices[0].message.content
    elif model_name == 'glm-3':
        response =  model.chat.completions.create(
                        model = "glm-3-turbo",  # glm-3-turbo
                        messages = message,
                        do_sample = do_sample,
                        temperature = temperature
                    )
        response_text = response.choices[0].message.content
    elif model_name == 'gpt-4':
        if not do_sample:
            temperature = 0
        completion = model.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=message,
            n = 1,
            temperature = temperature)

        response_text = completion.choices[0].message.content

    if 'Yes' in response_text:
        return 1
    elif 'No' in response_text:
        return 0
    else:
        return random.choice([0,1])
        


def eval_on_wwqa(model_path, data_path, result_path, local_model = True, api_key = None, peft_path = None):
    if local_model:
        if 'Baichuan' in model_path:
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, load_in_8bit = True, device_map = 'balanced_low_0')
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model_name = 'Baichuan'
            if peft_path is not None:
                model = PeftModel.from_pretrained(model, peft_path)
        elif 'Yi' in model_path:
            Yi_model = AutoModelForCausalLM.from_pretrained(model_path, device_map = 'auto')
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if peft_path is not None:
                Yi_model = PeftModel.from_pretrained(Yi_model, peft_path)
            model = pipeline("text-generation", model=Yi_model, tokenizer=tokenizer, device_map = 'auto')
            model_name = 'Yi'
        elif 'Mistral' in model_path:
            quantization_config = BitsAndBytesConfig(load_in_8bit = True)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map = 'auto', quantization_config = quantization_config)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_name = 'Mistral'
            if peft_path is not None:
                model = PeftModel.from_pretrained(model, peft_path) 
        elif 'chatglm' in model_path:
            model = AutoModel.from_pretrained(model_path, device_map = 'auto', trust_remote_code = True)
            if peft_path is not None:
                model = PeftModel.from_pretrained(model, peft_path)
            else:
                model = model.quantize(8)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
            model_name = 'chatglm3'
        elif 'internlm' in model_path:
            quantization_config = BitsAndBytesConfig(load_in_8bit = True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, quantization_config = quantization_config)
            if peft_path is not None:
                model = PeftModel.from_pretrained(model, peft_path) 
            model_name = 'internlm' 
            model = model.eval()


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
                pred = get_prediction(model, tokenizer, prompt, do_sample = False, model_name = model_name)
                prediction.append(pred)

        results = pd.DataFrame(classification_report(label, prediction, digits=4, output_dict=True)).T
    
    elif 'glm-4' in model_path:
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
                pred = get_prediction(model, tokenizer, prompt, do_sample = False, model_name = 'glm-4')
                prediction.append(pred)
    

        results = pd.DataFrame(classification_report(label, prediction, digits=4, output_dict=True)).T
    
    elif 'glm-3' in model_path:
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
                pred = get_prediction(model, tokenizer, prompt, do_sample = False, model_name = 'glm-3')
                prediction.append(pred)
    

        results = pd.DataFrame(classification_report(label, prediction, digits=4, output_dict=True)).T
    
    elif 'gpt-4' in model_path:
        model = OpenAI(api_key = api_key)
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
                pred = get_prediction(model, tokenizer, prompt, do_sample = False, model_name = 'gpt-4')
                prediction.append(pred)
    

        results = pd.DataFrame(classification_report(label, prediction, digits=4, output_dict=True)).T

    results.to_csv(os.path.join(result_path, 'wwqa_result.csv'))


if __name__ == '__main__':
    data_path = './WWQA/binary_qa.json'
    
    # Mistral-7B
    # model_path = '../LLM_weight/Mistral-7B-Instruct-v0.2/'
    # result_path = './results/Mistral-7B-FT'
    # peft_path = './weights/wwqa_train_1453_mistral-7B'
    # local_model = True

    # InternLM-20B
    model_path = '../LLM_weight/internlm2-chat-20b'
    result_path = './results/InternLM-20B'
    peft_path = './weights/wwqa_train_1453_internlm-20b'
    local_model = True

    # ChatGLM3-6b
    # model_path = '../LLM_weight/chatglm3-6b'
    # result_path = './results/chatglm3-6b-FT'
    # peft_path = './weights/wwqa_train_1453_chatglm3-6b'
    # local_model = True

    # model_path = 'glm-4'
    # result_path = './results/GLM-4'
    # api_key = ''
    # local_model = False
    # peft_path = None

    # model_path = 'glm-3'
    # result_path = './results/GLM-3'
    # api_key = ''
    # local_model = False
    # peft_path = None

    # model_path = 'gpt-4'
    # result_path = './results/GPT-4'
    api_key = ''
    # local_model = False
    # peft_path = None

    eval_on_wwqa(model_path, data_path, result_path, local_model, api_key = api_key, peft_path = peft_path)