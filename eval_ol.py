from moderator import GameModerator
from logger import Logging
import numpy as np


def play_game_eval(max_round: int, model_path: str, log_path: str, result_path: str, random_seed = 2023, peft_path = None, game_setting = 2, sheriff_role = None, early_stop = True):
    '''
    game_setting
        1: Homogeneous evaluation: All players are the same LLM-based agents. We can specify the role of the Sheriff in the assign_roles() method. 
        2: Heterogeneous evaluation: The Sheriff is implemented by the selected (tested) LLM-based agent while other players are the same LLM-based agents (default to be GLM-3). We can specify the role of the Sheriff in the assign_roles() method. 

        3 [NOT APPLICABLE IN THIS FUNCTION]: Human evaluation: One player is a human while other players are the same LLM-based agents. The Sheriff MUST BE a LLM-based agent. 
        4 [NOT APPLICABLE IN THIS FUNCTION]: Human baseline: One player is a human while other players are the same LLM-based agents. The Sheriff is the human player.

        5: Homogeneous evaluation variant 1: All players are the same LLM-based agents. It contains the election phase.
        6: Heterogeneous evaluation variant 1:  All players are initialized by the same LLM-based agents (default to be GLM-3), and when the election phase is over, the sheriff is replaced with the LLM to be tested.
        7: Heterogeneous evaluation variant 2:  One player is implemented by the selected (tested) LLM-based agent while other players are the same LLM-based agents (default to be GLM-3). It contains the election process, and if the LLM to be tested is not selected as the Sheriff, the simulation ends.
    '''
    assert game_setting in [1, 2, 5, 6, 7]
    moderator = GameModerator(model_path, log_path, result_path, peft_path = peft_path, game_setting = game_setting)
    if sheriff_role is not None:
        moderator.assign_roles(random_seed = random_seed, sheriff_role = sheriff_role, early_stop = early_stop)
    else:
        moderator.assign_roles(random_seed = random_seed, early_stop = early_stop)
    logger = Logging().log(log_path)

    if game_setting == 1:
        '''
        1: Homogeneous evaluation: All players are the same LLM-based agents. We can specify the role of the Sheriff in the assign_roles() method. 
        '''
        flag = False
        for round in range(1, max_round + 1):
            try:
                moderator.organize_night_action(round)
                if moderator.sheriff is None:
                    if sheriff_role is not None:
                        moderator.assign_sheriff(id = moderator.sheriff_id)
                    else:
                        moderator.assign_sheriff()
                if moderator.game_end(early_stop):
                    logger.info('Game End')
                    flag = True
                    break
                moderator.organize_day_discussion_eval(round)
                if moderator.game_end(early_stop):
                    logger.info('Game End')
                    flag = True
                    break
            except Exception as e:
                logger.exception("An error occurred: {}".format(e))

        if (round == max_round) & (flag == False):
            logger.info('Reach maximum number of rounds, game ends.')

        return moderator.start_time

    elif game_setting == 2:
        '''
        2: Heterogeneous evaluation: The Sheriff is implemented by the selected (tested) LLM-based agent while other players are the same LLM-based agents (default to be GLM-3). We can specify the role of the Sheriff in the assign_roles() method. 
        '''
        flag = False
        for round in range(1, max_round + 1):
            try:
                moderator.organize_night_action(round)
                if round == 1:
                    moderator.assign_sheriff(id = moderator.sheriff_id)
                if moderator.game_end():
                    logger.info('Game End')
                    flag = True
                    break
                moderator.organize_day_discussion_eval(round)
                if moderator.game_end():
                    logger.info('Game End')
                    flag = True
                    break
            except Exception as e:
                logger.exception("An error occurred: {}".format(e))

        if (round == max_round) & (flag == False):
            logger.info('Reach maximum number of rounds, game ends.')

        return moderator.start_time
    
    elif game_setting == 5:
        '''
        5: Homogeneous evaluation variant 1: All players are the same LLM-based agents. It contains the election phase.
        '''
        flag = False
        for round in range(1, max_round + 1):
            try:
                moderator.organize_night_action(round)
                if round == 1:
                    moderator.elect_sheriff()
                if moderator.game_end(early_stop):
                    logger.info('Game End')
                    flag = True
                    break
                moderator.organize_day_discussion_eval(round)
                if moderator.game_end(early_stop):
                    logger.info('Game End')
                    flag = True
                    break
            except Exception as e:
                logger.exception("An error occurred: {}".format(e))

        if (round == max_round) & (flag == False):
            logger.info('Reach maximum number of rounds, game ends.')

        return moderator.start_time


    elif game_setting == 6:
        '''
        6: Heterogeneous evaluation variant 1:  All players are initialized by the same LLM-based agents (default to be GLM-3), and when the election phase is over, the sheriff is replaced with the LLM to be tested.
        '''
        flag = False
        for round in range(1, max_round + 1):
            try:
                moderator.organize_night_action(round)
                if round == 1:
                    moderator.elect_sheriff()
                moderator.sheriff.reset_backend(model_path, moderator.model)
                if moderator.game_end():
                    logger.info('Game End')
                    flag = True
                    break
                moderator.organize_day_discussion_eval(round)
                if moderator.game_end():
                    logger.info('Game End')
                    flag = True
                    break
            except Exception as e:
                logger.exception("An error occurred: {}".format(e))

        if (round == max_round) & (flag == False):
            logger.info('Reach maximum number of rounds, game ends.')

        return moderator.start_time


    elif game_setting == 7:
        '''
        7: Heterogeneous evaluation variant 2:  One player is implemented by the selected (tested) LLM-based agent while other players are the same LLM-based agents (default to be GLM-3). It contains the election process, and if the LLM to be tested is not selected as the Sheriff, the simulation ends.
        '''
        flag = False
        for round in range(1, max_round + 1):
            try:
                moderator.organize_night_action(round)
                if round == 1:
                    moderator.elect_sheriff(must_join_player_ids = [moderator.evaluated_id])
                    if moderator.sheriff_id != moderator.evaluated_id:
                        flag = True
                        logger.info('The tested LLM failed to run for the Sheriff. The simulation ends.')
                        break
                if moderator.game_end():
                    logger.info('Game End')
                    flag = True
                    break
                moderator.organize_day_discussion_eval(round)
                if moderator.game_end():
                    logger.info('Game End')
                    flag = True
                    break
            except Exception as e:
                logger.exception("An error occurred: {}".format(e))

        if (round == max_round) & (flag == False):
            logger.info('Reach maximum number of rounds, game ends.')

        return moderator.start_time

def get_vote_decision(decision_matrix, id):
    decision = decision_matrix[id-1, :]
    id_vote = np.where(decision == 1)[0]
    if len(id_vote) == 0:
        return -1
    else:
        return id_vote[0]


def get_metric(data):
    rel_improvement = []
    decision_change = []
    decision_change_star = []
    ratio = []
    for round in data:
        if not isinstance(round, int):
            continue
        
        remaining_players = np.array(data[round]['remaining_players'])
        sheriff_id = data[round]['sheriff_id']
        player_without_sheriff = remaining_players[remaining_players != sheriff_id]

        reliability = data[round]['reliability']
        decision = data[round]['decision']
        decision_post = data[round]['decision_post']
        reliability_post = data[round]['reliability_post']

        other_player_reliability = reliability_post[player_without_sheriff-1][:,player_without_sheriff-1]

        previous = reliability[player_without_sheriff - 1, sheriff_id - 1]
        post = reliability_post[player_without_sheriff - 1, sheriff_id - 1]

        pre_mean = np.mean(previous[previous>0])
        post_mean = np.mean(post[post>0])
        other_mean = np.mean(other_player_reliability[other_player_reliability>0])

        rel_improvement.append( (post_mean - pre_mean) / pre_mean )
        ratio.append(post_mean / other_mean)

        sheriff_vote = get_vote_decision(decision_post, sheriff_id)
        count = 0
        count_star = 0
        for player_id in player_without_sheriff:
            pre_dec = get_vote_decision(decision, player_id)
            post_dec = get_vote_decision(decision_post, player_id)
            if (post_dec == sheriff_vote) & (pre_dec != post_dec):
                count += 1
            if (pre_dec != post_dec):
                count_star += 1
        decision_change.append(count / len(remaining_players))
        decision_change_star.append(count_star / len(remaining_players))

    return np.mean(ratio), np.mean(rel_improvement), np.mean(decision_change), np.mean(decision_change_star)


def get_metric_homo(data):
    '''
    This method returns Raito of all players
    '''
    role_summary = data['role_summary']
    ratio_for_all = {}
    roles = set(role_summary)
    for r in roles:
        ratio_for_all[r] = []

    rel_improvement = []
    decision_change = []
    decision_change_star = []
    ratio = []
    for round in data:
        if not isinstance(round, int):
            continue
        
        remaining_players = np.array(data[round]['remaining_players'])
        sheriff_id = data[round]['sheriff_id']
        player_without_sheriff = remaining_players[remaining_players != sheriff_id]

        reliability = data[round]['reliability']
        decision = data[round]['decision']
        decision_post = data[round]['decision_post']
        reliability_post = data[round]['reliability_post']

        other_player_reliability = reliability_post[player_without_sheriff-1][:,player_without_sheriff-1]

        previous = reliability[player_without_sheriff - 1, sheriff_id - 1]
        post = reliability_post[player_without_sheriff - 1, sheriff_id - 1]

        pre_mean = np.mean(previous[previous>0])
        post_mean = np.mean(post[post>0])
        other_mean = np.mean(other_player_reliability[other_player_reliability>0])

        rel_improvement.append( (post_mean - pre_mean) / pre_mean )
        ratio.append(post_mean / other_mean)

        for player_id in remaining_players:
            if player_id == sheriff_id:
                continue
            player_role = role_summary[player_id-1]
            other_players = remaining_players[remaining_players != player_id]
            other_player_reliability = reliability_post[other_players-1][:,other_players-1]
            selected_player_reliability = reliability_post[other_players-1,player_id-1]

            other_mean = np.mean(other_player_reliability[other_player_reliability>0])
            selected_mean = np.mean(selected_player_reliability[selected_player_reliability>0])

            ratio_for_all[player_role].append(selected_mean / other_mean)
        
        ratio_for_all[f"Sheriff_{role_summary[sheriff_id-1]}"] = ratio
        sheriff_vote = get_vote_decision(decision_post, sheriff_id)
        count = 0
        count_star = 0
        for player_id in player_without_sheriff:
            pre_dec = get_vote_decision(decision, player_id)
            post_dec = get_vote_decision(decision_post, player_id)
            if (post_dec == sheriff_vote) & (pre_dec != post_dec):
                count += 1
            if (pre_dec != post_dec):
                count_star += 1
        decision_change.append(count / len(remaining_players))
        decision_change_star.append(count_star / len(remaining_players))

    for r in ratio_for_all:
        if len(ratio_for_all[r]) > 0:
            ratio_for_all[r] = np.mean(ratio_for_all[r])
        else:
            ratio_for_all[r] = -1
    return np.mean(ratio), np.mean(rel_improvement), np.mean(decision_change), np.mean(decision_change_star), ratio_for_all


if __name__ == '__main__':
    #  chatglm3-6b
    max_round = 6
    model_path = "../LLM_weight/chatglm3-6b"
    log_path = './logs/chatglm3-6b'
    result_path = "./results/chatglm3-6b"
    sheriff_role = None
    peft_path = None 
    random_seed = 2023

    game_setting = 7
    early_stop = True
    play_game_eval(max_round, 
                   model_path, 
                   log_path, 
                   result_path, 
                   random_seed,
                   peft_path,
                   game_setting,
                   sheriff_role,
                   early_stop)