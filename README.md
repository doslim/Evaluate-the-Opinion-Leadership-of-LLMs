# Evaluate the Opinion Leadership of LLMs in the Werewolf Game

**Helmsman of the Masses? Evaluate the Opinion Leadership of Large Language Models in the Werewolf Game**

[![Paper](https://img.shields.io/badge/Paper-Arvix%20Link-green)](https://arxiv.org/abs/2404.01602)

<div align="center">
    <img src="demo.gif"  width="50%">
</div>

## News
- [x] [2024.08.26] The camera-ready version of our [paper](https://openreview.net/forum?id=xMt9kCv5YR) is released.
- [x] [2024.07.10] 🎉 Happy to announce that our paper is accepted to First Conference on Language Modeling (COLM) 2024! 
- [x] [2024.06.04] We update several new game settings. Refer to [Game Setting](#game-setting) for more details. 
  - Update ```eval_ol.py```, ```main.py```, ```moderator.py```, ```prompt.py```, ```role.py```.
- [x] [2024.04.02] We release our [paper](https://arxiv.org/abs/2404.01602) and this repo, including code and the WWQA dataset.

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

For [Yi](https://github.com/01-ai/Yi), [ChatGLM3](https://github.com/THUDM/ChatGLM3), [InternLM2](https://github.com/InternLM/InternLM) and [Mistral](https://mistral.ai/news/announcing-mistral-7b/) models, the following packages are recommended.

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

## Game Setting

1. **Homogeneous evaluation**: All players are the same LLM-based agents. We can specify the role of the Sheriff in the ```assign_roles()``` method. 
2. **Heterogeneous evaluation**: The Sheriff is implemented by the selected (tested) LLM-based agent while other players are the same LLM-based agents (default to be GLM-3). We can specify the role of the Sheriff in the ```assign_roles()``` method. 
3. **Human evaluation**: One player is a human while other players are the same LLM-based agents. The Sheriff MUST BE a LLM-based agent.
4. **Human baseline**: One player is a human while other players are the same LLM-based agents. The Sheriff is the human player.
5. **Homogeneous evaluation variant 1**: All players are the same LLM-based agents. It contains the election phase.
6. **Heterogeneous evaluation variant 1**:  All players are initialized by the same LLM-based agents (default to be GLM-3), and when the election phase is over, the sheriff is replaced with the LLM to be tested.
7. **Heterogeneous evaluation variant 2**:  One player is implemented by the selected (tested) LLM-based agent while other players are the same LLM-based agents (default to be GLM-3). It contains the election process, and if the LLM to be tested is not selected as the Sheriff, the simulation ends.

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
Then run the following command to fine-tune LLMs. You can switch the fine-tuned model by modifying the script. We currently support fine-tuning on ChatGLM3-6B, Mistral-7B, Baichuan2-13B, and InternLM2-20B.
```
sh fine-tune.sh
```

*3. Simulations*

After configuring the model weight path, game log storage path, result storage path, and setting the random seed in ```main.py```, run the following command. We currently support ChatGLM3-6B, Mistral-7B, Baichuan2-13B, InternLM2-20B, Yi-34B, GLM-3, GLM-4, and GPT-4.
```
python main.py
```

*4. Human Evaluation*

Run the following command and the game log and evaluation result will be saved in ```logs``` and ```results``` by default.
```
python human_eval.py
```

*5. Human Baseline*

Run the following command and the game log and evaluation result will be saved in ```logs``` and ```results``` by default.
```
python human_baseline.py
```

## Citation

If you find this project useful in your research, please consider citing:

```
@inproceedings{
    du2024helmsman,
    title={Helmsman of the Masses? Evaluate the Opinion Leadership of Large Language Models in the Werewolf Game},
    author={Silin Du and Xiaowei Zhang},
    booktitle={First Conference on Language Modeling},
    year={2024},
    url={https://openreview.net/forum?id=xMt9kCv5YR}
}
```
