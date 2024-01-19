from moderator import GameModerator
from logger import Logging

model_path = "../LLM_weight/Baichuan2-7B-chat"
log_path = './logs'

def play_game(max_round: int, model_path: str, log_path: str, random_seed = 2023):

    moderator = GameModerator(model_path, log_path)
    moderator.assign_roles(random_seed = random_seed)
    logger = Logging().log(log_path)

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
            moderator.organize_day_discussion(round)
            if moderator.game_end():
                logger.info('Game End')
                flag = True
                break
        except Exception as e:
            logger.exception("An error occurred: {}".format(e))


    if (round == max_round) & (flag == False):
        logger.info('Reach maximum number of roumds, game ends.')

play_game(6, model_path, log_path, 827)