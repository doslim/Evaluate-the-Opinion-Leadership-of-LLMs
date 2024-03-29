from moderator import GameModerator
from logger import Logging
import numpy as np


def play_game_eval(max_round: int, model_path: str, log_path: str, result_path: str, random_seed = 2023, peft_path = None, game_setting = 2, sheriff_role = None):
    '''
    game_setting
        1: Homogeneous evaluation: All players are the same LLM-based agents. We do not specify the role of the Sheriff
        2: Heterogeneous evaluation: The sheriff is implemented by the selected LLM-based agents while other players are the same LLM-based agents (default to be LlaMA-7B). We can specify the role of the Sheriff in the assign_roles() method. 
        3: Human evaluation: One player is a human while other players are the same LLM-based agents. The Sheriff MUST BE a LLM-based agent.
    '''
    
    moderator = GameModerator(model_path, log_path, result_path, peft_path = peft_path, game_setting = game_setting)
    if sheriff_role is not None:
        moderator.assign_roles(random_seed = random_seed, sheriff_role = sheriff_role)
    else:
        moderator.assign_roles(random_seed = random_seed)
    logger = Logging().log(log_path)

    if game_setting == 1:
        flag = False
        for round in range(1, max_round + 1):
            try:
                moderator.organize_night_action(round)
                if moderator.sheriff is None:
                    moderator.assign_sheriff()
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
            logger.info('Reach maximum number of roumds, game ends.')

        return moderator.start_time

    elif game_setting == 2:
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
            logger.info('Reach maximum number of roumds, game ends.')

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

        other_player_reliability = reliability_post[remaining_players-1][:,player_without_sheriff-1]

        previous = reliability[player_without_sheriff - 1, sheriff_id - 1]
        post = reliability_post[player_without_sheriff - 1, sheriff_id - 1]

        pre_mean = np.mean(previous[previous>0])
        post_mean = np.mean(post[post>0])
        other_mean = np.mean(other_player_reliability[other_player_reliability>0])

        rel_improvement.append( (post_mean - pre_mean) / pre_mean )
        ratio.append(post_mean / other_mean)

        sheriff_vote = get_vote_decision(decision_post, sheriff_id)
        count = 0
        for player_id in player_without_sheriff:
            pre_dec = get_vote_decision(decision, player_id)
            post_dec = get_vote_decision(decision_post, player_id)
            if (post_dec == sheriff_vote) & (pre_dec != post_dec):
                count += 1
        
        decision_change.append(count / len(remaining_players))


    return np.mean(ratio), np.mean(rel_improvement), np.mean(decision_change)


if __name__ == '__main__':
    model_path = "../LLM_weight/Baichuan2-13B-chat"
    log_path = './logs'
    result_path = "./results/Baichuan2-13B-chat"
    play_game_eval(6, model_path, log_path, result_path, 1999)