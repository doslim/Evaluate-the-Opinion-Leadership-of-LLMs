from moderator import GameModerator
from prompt import GAME_RULE
from logger import Logging
from colorama import init, Fore, Style
init(autoreset=True)
import json
import os
import numpy as np
import random
import pickle as pkl
from copy import deepcopy
from eval_ol import get_vote_decision, get_metric
from scipy.stats import spearmanr
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.history import FileHistory


def prompt_continuation(width, line_number, wrap_count):
    """
    The continuation: display line numbers and '->' before soft wraps.

    Notice that we can return any kind of formatted text from here.

    The prompt continuation doesn't have to be the same width as the prompt
    which is displayed before the first line, but in this example we choose to
    align them. The `width` input that we receive here represents the width of
    the prompt.
    """
    if wrap_count > 0:
        return " " * (width - 3) + "-> "
    else:
        text = ("- %i - " % (line_number + 1)).rjust(width)
        return HTML("<strong>%s</strong>") % text

width, _ = os.get_terminal_size()
role_avatar = {"player_1": "🤖1️⃣: ", 
               "player_2": "🤖2️⃣: ", 
               "player_3": "🤖3️⃣: ", 
               "player_4": "🤖4️⃣: ", 
               "player_5": "🤖5️⃣: ", 
               "player_6": "🤖6️⃣: ", 
               "player_7": "🤖7️⃣: ", 
               "human_player": "🧑‍💻: ",
               "moderator": "📢: "}

def display_user_input(text):
    formatted_text = "{:>{width}}".format(role_avatar['human_player'] + text, width=width)
    print(Fore.BLUE + formatted_text)

def display_moderator_output(text):
    print(Fore.RED + role_avatar['moderator'] + text + '\n')

def display_llm_player_output(text, player_id):
    print(Fore.GREEN + role_avatar[f'player_{player_id}'] + f' player_{player_id}: ' + text + '\n')


prompt_style = Style.from_dict({
            'prompt': 'red',
            'input': ''})
history = FileHistory('./logs/GLM-3-human/' + 'history.txt')
placeholder = HTML('<b>You are allowed to input multiple lines. Press [ESC] followed by [Enter] to quit.  </b>')
def get_multiple_line(text, default=""):
    text = role_avatar['moderator'] + text + '\n'
    print(Fore.YELLOW + text)
    output = prompt(
        "", multiline=True, prompt_continuation=prompt_continuation, style=prompt_style, history=history, placeholder=placeholder, default=default
    )
    display_user_input(output)
    return output



response = get_multiple_line('Please input the backbone model. Select one from [\'glm-3\', \'glm-4\']')
if response == 'glm-3':
    model_path = 'glm-3'
    log_path = './logs/GLM-3-human-baseline/'
    result_path = './results/GLM-3-human-baseline'
    print(Fore.RED + 'You will play with GLM-3.')
elif response == 'glm-4':
    model_path = 'glm-4'
    log_path = './logs/GLM-4-human-baseline/'
    result_path = './results/GLM-4-human-baseline'
    print(Fore.RED + 'You will play with GLM-4.')
else:
    model_path = 'glm-3'
    log_path = './logs/GLM-3-human-baseline/'
    result_path = './results/GLM-3-human-baseline'
    print(Fore.RED + 'You will play with GLM-3.')

peft_path = None
max_round = 6

response = get_multiple_line('Please input an integer as the random seed to initialize the game.')
if response.isdigit():
    random_seed = int(response)
else:
    random_seed = random.choice([10, 51, 82, 7, 21, 2019, 1997, 514, 429, 711])
moderator = GameModerator(model_path, log_path, result_path, peft_path = peft_path, game_setting = 4)
moderator.assign_roles(random_seed)

print(Fore.YELLOW + GAME_RULE + "\n")
print(Fore.RED + role_avatar['moderator'] + "You are player_{}, the {}. You're playing with 6 other players. \n".format(moderator.human_player_id, moderator.human_player.role) + "Now the night 1 round begins.")

start_time = moderator.start_time
flag = False

for round in range(1, max_round + 1):

    # organize nighe actions
    moderator.game_log[round] = {}
    moderator.temp_game_log = []
    players_selected_by_werewolf = []
    player_selected_by_guard = -1
    player_selected_by_seer = -1
    agent_werewolf_decision = False

    display_moderator_output(f'Now the night {round} round begins.')
    for player_id in moderator.remaining_players:
        current_player = moderator.players[player_id - 1]
        if current_player.role == "Werewolf":
            available_players = moderator.get_remaining_players()
            if current_player.player_id == moderator.human_player_id:
                if not agent_werewolf_decision:
                    # Let LLM-based Werewolf makes a decision first
                    teammate_id = moderator.get_teammate_id(player_id)
                    if teammate_id in moderator.remaining_players:
                        # teammate is alive
                        # teammate_id make decisions
                        selected_id = moderator.players[teammate_id - 1].take_night_action(round, player_id, available_players)
                        if selected_id == -1:
                            continue
                        players_selected_by_werewolf.append(selected_id)
                        msg = f'In night {round} round, your teammate (player_{teammate_id}) chose to kill player_{selected_id}.'
                        moderator.players[player_id - 1].fact.append(msg)
                        display_moderator_output(msg)
                        moderator.temp_game_log.append(f'In night {round} round, player_{teammate_id} chose to kill player_{selected_id}.')
                        moderator.players[teammate_id-1].save_log(start_time)
                        agent_werewolf_decision = True

                available_players = moderator.get_remaining_players()
                if round > 1:
                    human_instruction = moderator.human_player.deductive_reasoning(available_players, f'night {round}')
                    response = get_multiple_line(human_instruction)
                    if moderator.human_player.parse_reasoning(response):
                        display_moderator_output('Reliability update successful')
                    else:
                        display_moderator_output('Invalid format!')
                human_instruction = moderator.human_player.take_night_action(round, available_players)
                response = get_multiple_line(human_instruction)
                selected_id = moderator.human_player.parse_night_action(response, round, available_players)
                if selected_id:
                    display_moderator_output('Your action has been recorded.')
                    players_selected_by_werewolf.append(selected_id)
                    moderator.human_player.save_log(start_time)
                    teammate_id = moderator.get_teammate_id(player_id)
                    moderator.players[teammate_id - 1].fact.append(f'In night {round} round, your teammate (player_{player_id}) chose to kill player_{selected_id}.')
                    moderator.temp_game_log.append(f'In night {round} round, player_{player_id} chose to kill player_{selected_id}.')
                else:
                    display_moderator_output('Invalid format! Your action has not been recorded.')
                #     response = get_multiple_line(human_instruction + 'Please strictly follow the format instructions.')
            else:
                if agent_werewolf_decision:
                    continue
                teammate_id = moderator.get_teammate_id(player_id)
                selected_id = moderator.players[player_id - 1].take_night_action(round, teammate_id, available_players)
                if selected_id == -1:
                    continue
                players_selected_by_werewolf.append(selected_id)
                msg = f'In night {round} round, your teammate (player_{player_id}) chose to kill player_{selected_id}.'
                moderator.players[teammate_id - 1].fact.append(msg)
                if teammate_id == moderator.human_player_id:
                    display_moderator_output(msg)
                    agent_werewolf_decision = True
                moderator.temp_game_log.append(f'In night {round} round, player_{player_id} chose to kill player_{selected_id}.')
                moderator.players[player_id-1].save_log(start_time)
                

        if current_player.role == 'Guard':
            available_players = moderator.get_remaining_players()
            if current_player.player_id == moderator.human_player_id:
                if round > 1:
                    human_instruction = moderator.human_player.deductive_reasoning(available_players, f'night {round}')
                    response = get_multiple_line(human_instruction)
                    if moderator.human_player.parse_reasoning(response):
                        display_moderator_output('Reliability update successful')
                    else:
                        display_moderator_output('Invalid format!')
                human_instruction = moderator.human_player.take_night_action(round, available_players)
                response = get_multiple_line(human_instruction)
                player_selected_by_guard = moderator.human_player.parse_night_action(response, round, available_players)
                if player_selected_by_guard:
                    display_moderator_output('Your action has been recorded.')
                    moderator.human_player.save_log(start_time)
                    moderator.temp_game_log.append(f'In night {round} round, player_{player_id} chose to protect player_{player_selected_by_guard}.')
                else:
                    display_moderator_output('Invalid format.')
            else:
                player_selected_by_guard = moderator.guard.take_night_action(round, available_players)
                if player_selected_by_guard == -1:
                    continue
                moderator.temp_game_log.append(f'In night {round} round, player_{player_id} chose to protect player_{player_selected_by_guard}.')
                moderator.players[player_id-1].save_log(start_time)


        if current_player.role == 'Seer':
            available_players = moderator.get_remaining_players()
            if current_player.player_id == moderator.human_player_id:
                if round > 1:
                    human_instruction = moderator.human_player.deductive_reasoning(available_players, f'night {round}')
                    response = get_multiple_line(human_instruction)
                    if moderator.human_player.parse_reasoning(response):
                        display_moderator_output('Reliability update successful')
                    else:
                        display_moderator_output('Invalid format!')
                human_instruction = moderator.human_player.take_night_action(round, available_players)
                response = get_multiple_line(human_instruction)
                player_selected_by_seer = moderator.human_player.parse_night_action(response, round, available_players)
                if player_selected_by_seer:
                    display_moderator_output('Your action has been recorded.')
                    role_selected_by_seer = moderator.players[player_selected_by_seer-1].role
                    msg = f'In night {round}, you chose to see player_{player_selected_by_seer} and player_{player_selected_by_seer} is a {role_selected_by_seer}.'
                    display_moderator_output(msg)
                    moderator.temp_game_log.append(f'In night {round} round, player_{player_id} chose to see player_{player_selected_by_seer} and player_{player_selected_by_seer} is a {role_selected_by_seer}.')
                    moderator.seer.fact.append(msg)
                    moderator.human_player.save_log(start_time)
                else:
                    display_moderator_output('Invalid format! Your action has not been recorded.')
            else:
                player_selected_by_seer = moderator.seer.take_night_action(round, available_players)
                if player_selected_by_seer == -1:
                    # self.temp_game_log.append(f'In night {round} round, player_{player_id} doesn\'t select a player to see.')
                    # self.seer.fact.append(f'In night {round}, you don\'t select a player to see.')
                    # self.players[player_id-1].save_log(self.start_time)
                    continue

                role_selected_by_seer = moderator.players[player_selected_by_seer-1].role
                moderator.temp_game_log.append(f'In night {round} round, player_{player_id} chose to see player_{player_selected_by_seer} and player_{player_selected_by_seer} is a {role_selected_by_seer}.')
                moderator.seer.fact.append(f'In night {round}, you chose to see player_{player_selected_by_seer} and player_{player_selected_by_seer} is a {role_selected_by_seer}.')
                moderator.players[player_id-1].save_log(start_time)
    
    if len(set(players_selected_by_werewolf)) == 1:
        killed_player = players_selected_by_werewolf[0]
        if killed_player != player_selected_by_guard:
            moderator.temp_game_log.append(f'In night {round}, player_{killed_player} was killed by Werewolves.')
            msg = f'In night {round} round, player_{killed_player} was killed.'
            moderator.announcement(msg)
            display_moderator_output(msg)
            moderator.remaining_players.remove(killed_player)

            if round > 1 :
                if (moderator.sheriff is not None) and (killed_player == moderator.sheriff.player_id):
                    display_moderator_output("You have been killed. Game over.")
                    flag = True
                    break
        else:
            moderator.temp_game_log.append(f'In night {round}, player_{killed_player} was selected by Werewolves and Guard')
            msg = f'In night {round} round, no player was killed.'
            moderator.temp_game_log.append(msg)
            moderator.announcement(msg)
            display_moderator_output(msg)
    else:
        msg = f'In night {round} round, no player was killed.'
        moderator.temp_game_log.append(msg)
        moderator.announcement(msg)
        display_moderator_output(msg)

    moderator.game_log[round]['night'] = moderator.temp_game_log
    moderator.save_log()


    # determine whether the game is over
    if moderator.game_end() == 2:
        display_moderator_output("You have been killed. Game over.")
        flag = True
        break
    elif moderator.game_end():
        display_moderator_output(moderator.game_log['result'])
        flag = True
        break


    # We skip the election phase and set the human player as the Sheriff
    if round == 1:
        moderator.assign_sheriff(id = moderator.human_player.player_id)

    # organize day discussion
    display_moderator_output(f'Now the day {round} round begins.')
    moderator.temp_game_log = []

    # ask the human player to determine the statement order
    available_players = moderator.get_remaining_players()
    human_instruction = moderator.human_player.deductive_reasoning(available_players, f'day {round}')
    response = get_multiple_line(human_instruction)
    if moderator.human_player.parse_reasoning(response):
        display_moderator_output('Reliability update successful.')
    else:
        display_moderator_output('Invalid format!')
    human_instruction = moderator.human_player.determine_statement_order(round, available_players)
    response = get_multiple_line(human_instruction)
    statement_order, order_flag, first_player_id = moderator.human_player.parse_statement_order(response, round, available_players)
    moderator.statement_order = statement_order
    if not order_flag:
        msg = f'Invaild decision on the order of statement. Automatically set player_{first_player_id} to make a statement first in day {round}'
    else:
        msg = f'You (the Sheriff) chose player_{first_player_id} to make a statement first in day {round}'
    display_moderator_output(msg)
    moderator.temp_game_log.append(msg)
    moderator.decision_and_reliability[round] = {}
    moderator.decision_and_reliability[round]['sheriff_id'] = moderator.sheriff_id
    moderator.decision_and_reliability[round]['remaining_players'] = deepcopy(moderator.remaining_players)
    moderator.save_decision_and_reliability()
    moderator.save_log()
    moderator.game_log[round]['statement_order'] = moderator.statement_order
    moderator.announcement(msg, moderator.sheriff.player_id)

    for player_id in statement_order:
        if player_id == moderator.sheriff_id:
            continue
        if player_id != moderator.human_player.player_id:
            available_players = moderator.get_remaining_players()
            statement = moderator.players[player_id-1].day_discussion(round, available_players)
            moderator.players[player_id - 1].save_log(start_time)
            if statement is None:
                transferred_statement = f'In day {round} round, player_{player_id} said nothing.'
                display_llm_player_output(transferred_statement, player_id)
                moderator.temp_game_log.append(transferred_statement)
                moderator.public_information(transferred_statement, player_id)
                moderator.game_log[round]['day'] = moderator.temp_game_log
                moderator.save_log()
                continue
            else:
                transferred_statement = f'In day {round} round, player_{player_id} said: "{statement}".'
                moderator.temp_game_log.append(transferred_statement)
                display_llm_player_output(f'{statement}', player_id)
                moderator.public_information(transferred_statement, player_id)
                moderator.game_log[round]['day'] = moderator.temp_game_log
                moderator.save_log()
        
    decision_matrix = np.zeros((moderator.num_players, moderator.num_players))
    reliability_matrix = np.zeros((moderator.num_players, moderator.num_players))
    # collect the decision of the all LLM-based players
    for player_id in moderator.statement_order:
        if player_id == moderator.sheriff_id:
            continue
        available_players = moderator.get_remaining_players()
        selected_id = moderator.players[player_id-1].pseudo_vote(round, available_players)
        if selected_id != -1:
            decision_matrix[player_id-1][selected_id-1] = 1
            
        reliability = []
        for id in moderator.all_players:
            if id == player_id:
                reliability.append(0)
            elif f'player_{id}' in moderator.players[player_id-1].reliability:
                reliability.append(moderator.players[player_id-1].reliability[f'player_{id}'])
            else:
                # data missing
                reliability.append(-1)
        reliability_matrix[player_id-1] = reliability

    moderator.decision_and_reliability[round]['decision'] = decision_matrix
    moderator.decision_and_reliability[round]['reliability'] = reliability_matrix
    moderator.save_decision_and_reliability()


    # The Sheriff makes a statement
    available_players = moderator.get_remaining_players()
    human_instruction = moderator.human_player.deductive_reasoning(available_players, f'day {round}')
    response = get_multiple_line(human_instruction)
    if moderator.human_player.parse_reasoning(response):
        display_moderator_output('Reliability update successful.')
    else:
        display_moderator_output('Invalid format!')
    human_instruction = moderator.human_player.day_discussion(round, available_players)
    response = get_multiple_line(human_instruction)
    statement = moderator.human_player.parse_statement(response, round)
    if statement:
        display_moderator_output('Your statement has been recorded.')
        transferred_statement = f'In day {round} round, player_{player_id} said: "{statement}".'
        moderator.temp_game_log.append(transferred_statement)
        moderator.public_information(transferred_statement, player_id)
        moderator.game_log[round]['day'] = moderator.temp_game_log
        moderator.save_log()
    else:
        display_moderator_output('Invalid format!')
    moderator.players[player_id-1].save_log(start_time)


    # Now begin to vote and collect the reliability and decision matrix again   
    decision_matrix_post = np.zeros((moderator.num_players, moderator.num_players))
    reliability_matrix_post = np.zeros((moderator.num_players, moderator.num_players))

    voting_results = {}
    voting_record = {}
    for player_id in statement_order:
        if player_id != moderator.human_player.player_id:
            available_players = moderator.get_remaining_players()
            selected_id = moderator.players[player_id-1].vote(round, available_players)
            moderator.players[player_id - 1].save_log(start_time)
            voting_record[player_id] = selected_id
            if selected_id != -1:
                if selected_id not in voting_results:
                    voting_results[selected_id] = []
                voting_results[selected_id].append(f'player_{player_id}')
        else:
            available_players = moderator.get_remaining_players()
            human_instruction = moderator.human_player.deductive_reasoning(available_players, f'day {round}')
            response = get_multiple_line(human_instruction)
            if moderator.human_player.parse_reasoning(response):
                display_moderator_output('Reliability update successful.')
            else:
                display_moderator_output('Invalid format!')
            human_instruction = moderator.human_player.vote(round, available_players)
            response = get_multiple_line(human_instruction)
            selected_id = moderator.human_player.parse_voting(response, round, available_players)
            if selected_id:
                display_moderator_output('Your action has been recorded.')
                voting_record[player_id] = selected_id
                if selected_id != -1:
                    if selected_id not in voting_results:
                        voting_results[selected_id] = []
                    voting_results[selected_id].append(f'player_{player_id}')
            else:
                display_moderator_output('Invalid format!')
            moderator.players[player_id-1].save_log(start_time)

        reliability = []
        # note that the HumanPlayer uses a different structure to store the reliability.
        for id in moderator.all_players:
            if id == player_id:
                reliability.append(0)
            elif f'player_{id}' in moderator.players[player_id-1].reliability:
                if player_id == moderator.human_player.player_id:
                    reliability.append(moderator.players[player_id-1].reliability[f'player_{id}']['reliability'])
                else:
                    reliability.append(moderator.players[player_id-1].reliability[f'player_{id}'])
            else:
                # data missing
                reliability.append(-1)
        reliability_matrix_post[player_id-1] = reliability
    
    # announce the voting results
    for player_id in statement_order:
        selected_id = voting_record[player_id]
        if selected_id == -1:
            msg = f'In day {round} round, player_{player_id} did not vote.'
            moderator.temp_game_log.append(msg)
            moderator.announcement(msg, player_id)
            display_moderator_output(msg)
        else:
            decision_matrix_post[player_id-1][selected_id-1] = 1
            msg = f'In day {round} round, player_{player_id} voted to eliminate player_{selected_id}.'
            moderator.temp_game_log.append(msg)
            display_moderator_output(msg)
            moderator.announcement(msg, player_id)
            moderator.game_log[round]['day'] = moderator.temp_game_log
            moderator.save_log()

    
    moderator.decision_and_reliability[round]['decision_post'] = decision_matrix_post
    moderator.decision_and_reliability[round]['reliability_post'] = reliability_matrix_post
    moderator.save_decision_and_reliability()

    moderator.temp_game_log.append(voting_results)
    moderator.game_log[round]['day'] = moderator.temp_game_log
    moderator.save_log()

    temp_id = -1
    max_votes = -1
    flag = False
    for player_id in voting_results:
        if len(voting_results[player_id]) > max_votes:
            temp_id = player_id
            max_votes = len(voting_results[player_id])
            flag = True
        elif len(voting_results[player_id]) == max_votes:
            flag = False
    if flag:
        msg = f'In day {round} round, player_{temp_id} had the most votes and was eliminated.'
        display_moderator_output(msg)
        moderator.remaining_players.remove(temp_id)
        if (moderator.sheriff is not None) and (temp_id == moderator.sheriff.player_id):
            display_moderator_output("You have been killed. Game over.")
            flag = True
            break
    else:
        msg = f'In day {round} round, more than two players got the same votes and no player was eliminated.'
        display_moderator_output(msg)

    moderator.temp_game_log.append(msg)
    moderator.announcement(msg)
    moderator.game_log[round]['day'] = moderator.temp_game_log
    moderator.game_log[round]['remaining_player'] = moderator.get_remaining_players()
    moderator.save_log()
    

    # determine whether the game is over
    if moderator.game_end() == 2:
        display_moderator_output("You have been killed. Game over.")
        flag = True
        break
    elif moderator.game_end():
        display_moderator_output(moderator.game_log['result'])
        flag = True
        break


if (round == max_round) & (flag == False):
    display_moderator_output('Reach maximum number of roumds, game over.')


result_file = os.path.join(result_path, f'{start_time}', f'results_{random_seed}.pkl')
try:
    with open(result_file, 'rb') as f:
        data = pkl.load(f)
except Exception as e:
    print('Cannot open the result file')
ratio, rel_improvement, decision_change, decision_change_star = get_metric(data)
print(ratio, rel_improvement, decision_change, decision_change_star)

