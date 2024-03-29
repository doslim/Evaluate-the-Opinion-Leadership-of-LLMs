# Evaluate the Opinion Leadership of Large Language Models


## Requirements and Structures
All code is implemented by ```Python```. We evaluate the opinion leadership of different LLMs, which require different versions of ```transformers``` and other related packages.

For [Baichuan 2](https://github.com/baichuan-inc/Baichuan2) series models, the following packages are recommended.

- accelerate-0.24.1
- bitsandbytes-0.41.2.post2
- datasets-2.10.1
- peft-0.6.2
- sentencepiece-0.1.99
- torch-2.0.1
- transformers-4.33.0
- wandb-0.16.0

For [Yi](https://github.com/01-ai/Yi), [ChatGLM3](https://github.com/THUDM/ChatGLM3), [InternLM](https://github.com/InternLM/InternLM) and [Mistral](https://mistral.ai/news/announcing-mistral-7b/) models, the following packages are recommended.

- accelerate-0.27.2
- auto-gptq-0.6.0+cu118
- bitsandbytes-0.41.2.post2
- datasets-2.10.1
- peft-0.6.2
- torch-2.1.2+cu118
- transformers-4.38.2
- wandb-0.16.0
- xformers-0.0.23.post1

For GLM series models and GPT-4, the following packages are required.

- openai-1.14.3
- zhipuai-2.0.1

The structure of our project is as follows.
- ```data_preprocess.py```: to tokenize the WWQA dataset.
- ```eval_ol.py```: to evaluate the opinion leadership of LLMs.
- ```eval_wwqa.py```: to evaluate LLMs on the binary QA dataset.
- ```fine-tune.py```: to fine-tune LLMs using tokenized WWQA dataset.
- ```human_eval.py```: entrance to the human evaluation experiment.
- ```logger.py```: the logger.
- ```main.py```: entrance to evaluate the opinion leadership of LLMs through simulations.
- ```moderator.py```: the class for the moderator, which can control the game.
- ```prompt.py```: predefined prompt templates.
- ```role.py```: the class for different roles in the game.
- ```utils.py```: define tools to parse outputs of LLMs.
- ```fine-tune.sh```: the script to fine-tune LLMs.
- ```/WWQA```: the WWQA dataset.
- ```/logs```: directory to save the game logs.
- ```/weights```: the directory to save model weights during fine-tuning.
- ```/results```: the directory to save evaluation results.


## Usage
All commands should be run from the root directory of our project. Ensure that your environment matches the LLMs.

*1. Evaluate LLMs on the binary QA dataset* 

To begin with, insert the game rules in front of each question in the binary QA dataset and save the processed files in the ```WWQA``` folder. Make sure to configure the appropriate LLM weight path, dataset path, and result storage path in the ```eval_wwqa.py```, and then run the following command.
```
python eval_wwqa.py
```

*2. Fine-tune LLMs on the WWQA dataset*

First, tokenize the dataset using ```data_preprocess.py```. The processed files will be saved by default in the ```tokenized_data``` directory under the ```WWQA``` folder.  
```
python data_preprocess.py
```
Then run the following command to fine-tune LLMs. You can switch the fine-tuned model by modifying the script. We currently support fine-tuning on ChatGLM3-6B, Mistral-7B, Baichuan2-13B, and InternLM-20B.
```
sh fine-tune.sh
```

*3. Simulations*

After configuring the model weight path, game log storage path, result storage path, and setting the random seed in ```main.py```, run the following command. We currently support ChatGLM3-6B, Mistral-7B, Baichuan2-13B, InternLM-20B, Yi-34B, GLM-3, GLM-4, and GPT-4.
```
python main.py
```

*4. Human Evaluation*

Run the following command and the game log and evaluation result will be saved in ```logs``` and ```results``` by default.
```
python human_eval.py
```
