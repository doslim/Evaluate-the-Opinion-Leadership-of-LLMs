import json
import pickle as pkl
import os
import numpy as np
from eval_ol import play_game_eval, get_metric, get_metric_homo
from logger import Logging


def eval_opinion_leadership(max_round: int, model_path: str, log_path: str, result_path: str, random_seeds: list, peft_path = None, game_setting = 2, sheriff_role = None, early_stop = True):
    logger = Logging().log(log_path)
    eval_results = {}
    str_random_seeds = [str(i) for i in random_seeds]
    str_random_seeds = '_'.join(str_random_seeds)
    if sheriff_role is not None:
        str_random_seeds += f'_{sheriff_role}'
    ratio_list = []
    rel_improvement_list = []
    decision_change_list = []
    decision_change_star_list = []
    ratio_dict = {}

    if isinstance(sheriff_role, list):
        for sheriff_identity in sheriff_role:
            logger.info(f"Current Sheriff Role: {sheriff_identity}")
            for random_seed in random_seeds:
                
                dict_key = sheriff_identity + str(random_seed)
                eval_results[dict_key] = {}
                date = play_game_eval(max_round, model_path, log_path, result_path, random_seed, peft_path, game_setting, sheriff_identity, early_stop)
                result_file = os.path.join(result_path, f'{date}', f'results_{random_seed}.pkl')
                try:
                    with open(result_file, 'rb') as f:
                        data = pkl.load(f)
                except Exception as e:
                    logger.exception("An error occurred: {}".format(e))
                    continue

                logger.info(data)
                if game_setting == 2 or game_setting == 6 or game_setting == 7:
                    ratio, rel_improvement, decision_change, decision_change_star = get_metric(data)

                elif game_setting == 1 or game_setting == 5:
                    ratio, rel_improvement, decision_change, decision_change_star, ratio_for_all = get_metric_homo(data)
                    for r in ratio_for_all:
                        if r in ratio_dict:
                            ratio_dict[r].append(ratio_for_all[r])
                        else:
                            ratio_dict[r] = []
                            ratio_dict[r].append(ratio_for_all[r])

                    eval_results[dict_key]['ratio_for_all'] = ratio_for_all

                ratio_list.append(ratio)
                rel_improvement_list.append(rel_improvement)
                decision_change_list.append(decision_change)
                decision_change_star_list.append(decision_change_star)

            
                eval_results[dict_key]['ratio'] = ratio
                eval_results[dict_key]['rel_improvement'] = rel_improvement
                eval_results[dict_key]['decision_change'] = decision_change
                eval_results[dict_key]['decision_change_star'] = decision_change_star
                logger.info(eval_results)
                
                with open(os.path.join(result_path, f'eval_results_{str_random_seeds}.json'), "w", encoding = 'utf8') as fp:
                    json.dump(eval_results, fp, indent = 4)
    else:
        for random_seed in random_seeds:

            eval_results[random_seed] = {}
            date = play_game_eval(max_round, model_path, log_path, result_path, random_seed, peft_path, game_setting, sheriff_role, early_stop)
            result_file = os.path.join(result_path, f'{date}', f'results_{random_seed}.pkl')
            try:
                with open(result_file, 'rb') as f:
                    data = pkl.load(f)
            except Exception as e:
                logger.exception("An error occurred: {}".format(e))
                continue

            logger.info(data)
            if game_setting == 2  or game_setting == 6 or game_setting == 7:
                ratio, rel_improvement, decision_change, decision_change_star = get_metric(data)

            elif game_setting == 1 or game_setting == 5:
                ratio, rel_improvement, decision_change, decision_change_star, ratio_for_all = get_metric_homo(data)
                for r in ratio_for_all:
                    if r in ratio_dict:
                        ratio_dict[r].append(ratio_for_all[r])
                    else:
                        ratio_dict[r] = []
                        ratio_dict[r].append(ratio_for_all[r])

                eval_results[random_seed]['ratio_for_all'] = ratio_for_all

            ratio_list.append(ratio)
            rel_improvement_list.append(rel_improvement)
            decision_change_list.append(decision_change)
            decision_change_star_list.append(decision_change_star)

            
            eval_results[random_seed]['ratio'] = ratio
            eval_results[random_seed]['rel_improvement'] = rel_improvement
            eval_results[random_seed]['decision_change'] = decision_change
            eval_results[random_seed]['decision_change_star'] = decision_change_star
            logger.info(eval_results)
                
            with open(os.path.join(result_path, f'eval_results_{str_random_seeds}.json'), "w", encoding = 'utf8') as fp:
                json.dump(eval_results, fp, indent = 4)

    for r in ratio_dict:
        temp = np.array(ratio_dict[r])
        ratio_dict[r] = np.mean(temp[temp>0])
    eval_results['ratio_dict'] = ratio_dict
    eval_results['ratio'] = np.mean(ratio_list)
    eval_results['rel_improvement'] = np.mean(rel_improvement_list)
    eval_results['decision_change'] = np.mean(decision_change_list)
    eval_results['decision_change_star'] = np.mean(decision_change_star_list)
    with open(os.path.join(result_path, f'eval_results_{str_random_seeds}.json'), "w", encoding = 'utf8') as fp:
        json.dump(eval_results, fp, indent = 4)



if __name__ == '__main__':
    max_round = 6

    # Mistral-7B
    # model_path = "../LLM_weight/Mistral-7B-Instruct-v0.2"
    # log_path = './logs/Mistral-7B-FT'
    # result_path = "./results/Mistral-7B-FT"
    # peft_path = "./weights/wwqa_train_1453_mistral-7B"
    # peft_path = None 

    # Baichuan2-13B
    # model_path = "../LLM_weight/Baichuan2-13B-chat-v2"
    # log_path = './logs/Baichuan2-13B-v2'
    # result_path = "./results/Baichuan2-13B-chat-v2"
    # sheriff_role = "Werewolf"
    # peft_path = "./weights/wwqa_train_1453_baichuan2-13B-v2"
    # peft_path = None 

    # chatglm3-6b
    model_path = "../LLM_weight/chatglm3-6b"
    log_path = './logs/chatglm3-6b'
    result_path = "./results/chatglm3-6b"
    sheriff_role = None
    peft_path = None 
    # peft_path = "./weights/wwqa_train_1453_chatglm3-6b"

    # Yi-34B
    # model_path = "../LLM_weight/Yi-34B-Chat-8bits"
    # log_path = './logs/Yi-34B'
    # result_path = "./results/Yi-34B"
    # peft_path = "./weights/wwqa_train_1453_chatglm3-6b"
    # peft_path = None 

    # InternLM-20B
    # model_path = "../LLM_weight/internlm2-chat-20b"
    # log_path = './logs/InternLM-20B'
    # result_path = "./results/InternLM-20B"
    # sheriff_role = 'Villager'
    # peft_path = None 
    # peft_path = "./weights/wwqa_train_1453_internlm-20b"

    # model_path = "glm-3"
    # log_path = './logs/GLM-3'
    # result_path = "./results/GLM-3"
    # peft_path = None 
    # sheriff_role = ['Werewolf', 'Villager', 'Seer', 'Guard']
    
    # model_path = "glm-4"
    # log_path = './logs/GLM-4'
    # result_path = "./results/GLM-4"
    # peft_path = None 
    # sheriff_role = ['Werewolf', 'Villager', 'Seer', 'Guard']

    # model_path = "gpt-4"
    # log_path = './logs/GPT-4'
    # result_path = "./results/GPT-4"
    # peft_path = None 


    random_seeds = []
    game_setting = 6
    early_stop = True
    eval_opinion_leadership(max_round, model_path, log_path, result_path, random_seeds, peft_path, game_setting, sheriff_role, early_stop)