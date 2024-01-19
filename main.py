import json
import pickle as pkl
import os
import numpy as np
from eval_ol import play_game_eval, get_metric
from logger import Logging


def eval_opinion_leadership(max_round: int, model_path: str, log_path: str, result_path: str, random_seeds: list, peft_path = None):
    logger = Logging().log(log_path)
    eval_results = {}
    str_random_seeds = [str(i) for i in random_seeds]
    str_random_seeds = '_'.join(str_random_seeds)

    ratio_list = []
    rel_improvement_list = []
    decision_change_list = []
    for random_seed in random_seeds:
        date = play_game_eval(max_round, model_path, log_path, result_path, random_seed, peft_path)
        result_file = os.path.join(result_path, f'{date}', f'results_{random_seed}.pkl')
        with open(result_file, 'rb') as f:
            data = pkl.load(f)
        logger.info(data)
        ratio, rel_improvement, decision_change = get_metric(data)

        ratio_list.append(ratio)
        rel_improvement_list.append(rel_improvement)
        decision_change_list.append(decision_change)

        eval_results[random_seed] = {}
        eval_results[random_seed]['ratio'] = ratio
        eval_results[random_seed]['rel_improvement'] = rel_improvement
        eval_results[random_seed]['decision_change'] = decision_change
        logger.info(eval_results)

        
        with open(os.path.join(result_path, f'eval_results_{str_random_seeds}.json'), "w", encoding = 'utf8') as fp:
            json.dump(eval_results, fp, indent = 4)


    eval_results['ratio'] = np.mean(ratio_list)
    eval_results['rel_improvement'] = np.mean(rel_improvement_list)
    eval_results['decision_change'] = np.mean(decision_change_list)
    with open(os.path.join(result_path, f'eval_results_{str_random_seeds}.json'), "w", encoding = 'utf8') as fp:
        json.dump(eval_results, fp, indent = 4)



if __name__ == '__main__':
    max_round = 6
    model_path = "../LLM_weight/Baichuan2-13B-chat"
    log_path = './logs/Baichuan2-13B'
    result_path = "./results/Baichuan2-13B-chat"
    # peft_path = "./weights/wwqa_train_1453_baichuan2-13B"
    peft_path = None
    # model_path = "chatGLM"
    # log_path = './logs/GLM-3'
    # result_path = "./results/GLM-3"
    # random_seeds = [429, 514, 827, 712, 2021, 2019, 1997, 1999, 2024, 711]
    random_seeds = [2019, 1997, 1999, 2024, 711]
    eval_opinion_leadership(max_round, model_path, log_path, result_path, random_seeds, peft_path)