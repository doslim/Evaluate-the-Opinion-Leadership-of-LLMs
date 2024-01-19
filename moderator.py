# The class for moderator
# Author: Silin Du
# Date: 2024-1
# Version: 2.0


import time
from logger import Logging
from transformers import AutoModelForCausalLM
import random
from role import Villager, Werewolf, Guard, Seer 
import os
import json
import numpy as np
import pickle as pkl
from peft import PeftModel

class GameModerator:
    def __init__(self, model_path: str, log_path: str, result_path = None, num_players = 7, peft_path = None) -> None:
        self.num_players = num_players
        self.log_path = log_path
        self.model_path = model_path
        self.result_path = result_path
        self.game_log = {} 
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        if model_path in ['chatGLM', 'gpt-3.5', 'gpt-4']:
            self.model = None
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, load_in_8bit = True, device_map = 'auto')
        self.logger = Logging().log(log_path, level='DEBUG')

        self.decision_and_reliability = {}

        self.peft_path = peft_path
        if peft_path is not None:
            self.model = PeftModel.from_pretrained(self.model, peft_path)


    def assign_roles(self, random_seed = None, villager_path = None, wolf_path = None, guard_path = None, witch_path = None, seer_path = None) -> None:

        # Initialization
        if villager_path is None:
            self.villager_1 = Villager(self.model_path, self.log_path, True, self.model)
            self.villager_2 = Villager(self.model_path, self.log_path, True, self.model)
            self.villager_3 = Villager(self.model_path, self.log_path, True, self.model)
        else:
            self.villager_1 = Villager(villager_path, self.log_path)
            self.villager_2 = Villager(villager_path, self.log_path)
            self.villager_3 = Villager(villager_path, self.log_path)


        if wolf_path is None:
            self.werewolf_1 = Werewolf(self.model_path, self.log_path, True, self.model)
            self.werewolf_2 = Werewolf(self.model_path, self.log_path, True, self.model)
        else:
            self.werewolf_1 = Werewolf(wolf_path, self.log_path)
            self.werewolf_2 = Werewolf(wolf_path, self.log_path)


        if guard_path is None:
            self.guard = Guard(self.model_path, self.log_path, True, self.model)
        else:
            self.guard = Guard(guard_path, self.log_path)

        # if witch_path is None:
        #     self.witch = Witch(self.model_path, self.log_path)
        # else:
        #     self.witch = Witch(witch_path, self.log_path)

        if seer_path is None:
            self.seer = Seer(self.model_path, self.log_path, True, self.model)
        else:
            self.seer = Seer(seer_path, self.log_path)

        self.players = [self.werewolf_1, self.werewolf_2, self.villager_1, self.villager_2, self.villager_3, self.guard, self.seer]
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(self.players)
        self.random_seed = random_seed
        self.game_log['random_seed'] = self.random_seed

        self.player_1 = self.players[0]
        self.player_1.set_player_id(1)
        self.player_2 = self.players[1]
        self.player_2.set_player_id(2)
        self.player_3 = self.players[2]
        self.player_3.set_player_id(3)
        self.player_4 = self.players[3]
        self.player_4.set_player_id(4)
        self.player_5 = self.players[4]
        self.player_5.set_player_id(5)
        self.player_6 = self.players[5]
        self.player_6.set_player_id(6)
        self.player_7 = self.players[6]
        self.player_7.set_player_id(7)

        self.sheriff = None

        role_summary = { \
              "player_1": self.player_1.role,
              "player_2": self.player_2.role,
              "player_3": self.player_3.role,
              "player_4": self.player_4.role,
              "player_5": self.player_5.role,
              "player_6": self.player_6.role,
              "player_7": self.player_7.role,
        }

        self.logger.info(role_summary)
        self.decision_and_reliability['role_summary'] = [i.role for i in self.players]
        self.game_log['role_summary'] = role_summary
        self.save_log()
        self.all_players = [1, 2, 3, 4, 5, 6, 7]
        self.remaining_players = [1, 2, 3, 4, 5, 6, 7]
        self.werewolf_id = {self.werewolf_1.player_id, self.werewolf_2.player_id}
        self.werewolf_1.fact.append(f'player_{self.werewolf_2.player_id} is your teammate and is a Werewolf.' )
        self.werewolf_2.fact.append(f'player_{self.werewolf_1.player_id} is your teammate and is a Werewolf.' )


    def assign_sheriff(self):
        if self.random_seed is not None:
            random.seed(self.random_seed)
        self.sheriff_id = random.choice(self.remaining_players)
        self.sheriff = self.players[self.sheriff_id - 1]
        self.sheriff.fact.append('You are selected as the Sheriff, which can determine the order of statement, summarize the discussion and provide advice for voting at last.')
        msg = f"After discussion and a vote, player_{self.sheriff_id} was selected as the Sheriff, which can determine the order of statement, summarize the discussion and provide advice for voting at last."
        self.announcement(msg, self.sheriff_id)
        self.logger.info(msg)


    def convey_sheriff(self, time_str: str):
        if self.random_seed is not None:
            random.seed(self.random_seed)
        previous_sheriff_id = self.sheriff.player_id
        self.sheriff_id = random.choice(self.remaining_players)
        self.sheriff = self.players[self.sheriff_id - 1]

        if 'day' in time_str:
            msg = f'The Sheriff, player_{previous_sheriff_id}, was eliminated in {time_str}. It selects you as the next Sheriff, which can determine the order of statement, summarize the discussion and provide advice for voting at last.'
        else:
            msg = f'The Sheriff, player_{previous_sheriff_id}, was killed in {time_str}. It selects you as the next Sheriff, which can determine the order of statement, summarize the discussion and provide advice for voting at last.'
        self.sheriff.fact.append(msg)
        msg = f"player_{previous_sheriff_id} selected player_{self.sheriff_id} as the next Sheriff, which can determine the order of statement, summarize the discussion and provide advice for voting at last."
        self.announcement(msg)
        self.logger.info(msg)


    def organize_night_action(self, round) -> None:
        self.game_log[round] = {}
        self.temp_game_log = []
        players_selected_by_werewolf = []
        player_selected_by_guard = -1
        player_selected_by_seer = -1

        for player_id in self.remaining_players:
            if self.players[player_id - 1].role == "Werewolf":
                available_players = self.get_remaining_players()
                teammate_id = self.get_teammate_id(player_id)
                selected_id = self.players[player_id - 1].take_night_action(round, teammate_id, available_players)
                if selected_id == -1:
                    continue
                players_selected_by_werewolf.append(selected_id)

                self.players[teammate_id - 1].fact.append(f'In night {round} round, your teammate (player_{player_id}) chose to kill player_{selected_id}.')
                self.temp_game_log.append(f'In night {round} round, player_{player_id} chose to kill player_{selected_id}.')
                self.players[player_id-1].save_log(self.start_time)


        for player_id in self.remaining_players:
            if self.players[player_id - 1].role == "Guard":
                available_players = self.get_remaining_players()
                player_selected_by_guard = self.guard.take_night_action(round, available_players)
                if player_selected_by_guard == -1:
                    continue
                self.temp_game_log.append(f'In night {round} round, player_{player_id} chose to protect player_{player_selected_by_guard}.')
                self.players[player_id-1].save_log(self.start_time)


        for player_id in self.remaining_players:
            if self.players[player_id - 1].role == "Seer":
                available_players = self.get_remaining_players()
                player_selected_by_seer = self.seer.take_night_action(round, available_players)
                if player_selected_by_seer == -1:
                    # self.temp_game_log.append(f'In night {round} round, player_{player_id} doesn\'t select a player to see.')
                    # self.seer.fact.append(f'In night {round}, you don\'t select a player to see.')
                    # self.players[player_id-1].save_log(self.start_time)
                    continue

                role_selected_by_seer = self.players[player_selected_by_seer-1].role
                self.temp_game_log.append(f'In night {round} round, player_{player_id} chose to see player_{player_selected_by_seer} and player_{player_selected_by_seer} is a {role_selected_by_seer}.')
                self.seer.fact.append(f'In night {round}, you chose to see player_{player_selected_by_seer} and player_{player_selected_by_seer} is a {role_selected_by_seer}.')
                self.players[player_id-1].save_log(self.start_time)


        if len(set(players_selected_by_werewolf)) == 1:
            killed_player = players_selected_by_werewolf[0]
            if killed_player != player_selected_by_guard:
                self.temp_game_log.append(f'In night {round}, player_{killed_player} was killed by Werewolves.')
                self.announcement(f'In night {round} round, player_{killed_player} was killed.')
                self.remaining_players.remove(killed_player)

                if (self.sheriff is not None) and (killed_player == self.sheriff.player_id):
                    time_str = f'night {round}'
                    self.convey_sheriff(time_str)
            else:
                self.temp_game_log.append(f'In night {round}, player_{killed_player} was selected by Werewolves and Guard')
                self.temp_game_log.append(f'In night {round} round, no player was killed.')
                self.announcement(f'In night {round} round, no player was killed.')

        self.game_log[round]['night'] = self.temp_game_log
        # self.game_log[round]['remaining_player'] = self.get_remaining_players()
        self.save_log()


    def organize_day_discussion(self, round: int) -> None:
        self.temp_game_log = []

        if self.sheriff is None:
            self.statement_order = self.remaining_players
        else:
            self.statement_order = self.sheriff.determine_statement_order(round, self.get_remaining_players())

        msg = f'The Sheriff (player_{self.sheriff.player_id}) chose player_{self.statement_order[0]} to make a statement first in day {round}'
        self.game_log[round]['statement_order'] = self.statement_order
        self.temp_game_log.append(msg)
        self.announcement(msg, self.sheriff.player_id)
        for player_id in self.statement_order:
            available_players = self.get_remaining_players()
            statement = self.players[player_id-1].day_discussion(round, available_players)
            self.players[player_id - 1].save_log(self.start_time)
            if statement is None:
                transferred_statement = f'In day {round} round, player_{player_id} said nothing.'
                self.temp_game_log.append(transferred_statement)
                self.public_information(transferred_statement, player_id)
                self.game_log[round]['day'] = self.temp_game_log
                self.save_log()
                continue

            transferred_statement = f'In day {round} round, player_{player_id} said: "{statement}".'
            self.temp_game_log.append(transferred_statement)
            self.public_information(transferred_statement, player_id)
            self.game_log[round]['day'] = self.temp_game_log
            self.save_log()

        voting_results = {}
        for player_id in self.statement_order:
            available_players = self.get_remaining_players()
            selected_id = self.players[player_id-1].vote(round, available_players)
            self.players[player_id - 1].save_log(self.start_time)
            if selected_id == -1:
                msg = f'In day {round} round, player_{player_id} did not vote.'
                self.temp_game_log.append(msg)
                self.announcement(msg, player_id)
                continue

            msg = f'In day {round} round, player_{player_id} voted to eliminate player_{selected_id}.'
            self.temp_game_log.append(msg)
            self.announcement(msg, player_id)
            self.game_log[round]['day'] = self.temp_game_log
            self.save_log()

            if selected_id not in voting_results:
                voting_results[selected_id] = []
            voting_results[selected_id].append(f'player_{player_id}')
        
        self.temp_game_log.append(voting_results)
        self.game_log[round]['day'] = self.temp_game_log
        self.save_log()

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
            self.remaining_players.remove(temp_id)
            if (self.sheriff is not None) and (temp_id == self.sheriff.player_id):
                time_str = f'day {round}'
                self.convey_sheriff(time_str)
        else:
            msg = f'In day {round} round, more than two players got the same votes and no player was eliminated.'

        self.temp_game_log.append(msg)
        self.announcement(msg)
        self.game_log[round]['day'] = self.temp_game_log
        self.game_log[round]['remaining_player'] = self.get_remaining_players()
        self.save_log()


    def organize_day_discussion_eval(self, round: int) -> None:
        '''
        In this method, we will evaluate the opinion leadership during daily discussion.
        '''
        self.temp_game_log = []

        if self.sheriff is None:
            return None
            # self.statement_order = self.remaining_players
        else:
            self.statement_order = self.sheriff.determine_statement_order(round, self.get_remaining_players())

        self.decision_and_reliability[round] = {}
        self.decision_and_reliability[round]['sheriff_id'] = self.sheriff_id
        self.decision_and_reliability[round]['remaining_players'] = self.remaining_players
        self.save_decision_and_reliability()
        self.save_log()
        
        msg = f'The Sheriff (player_{self.sheriff.player_id}) chose player_{self.statement_order[0]} to make a statement first in day {round}'
        self.game_log[round]['statement_order'] = self.statement_order
        self.temp_game_log.append(msg)
        self.announcement(msg, self.sheriff.player_id)

        # All players except the Sheriff make a statement
        for player_id in self.statement_order:
            if player_id == self.sheriff_id:
                continue
            available_players = self.get_remaining_players()
            statement = self.players[player_id-1].day_discussion(round, available_players)
            self.players[player_id - 1].save_log(self.start_time)
            if statement is None:
                transferred_statement = f'In day {round} round, player_{player_id} said nothing.'
                self.temp_game_log.append(transferred_statement)
                self.public_information(transferred_statement, player_id)
                self.game_log[round]['day'] = self.temp_game_log
                self.save_log()
                continue

            transferred_statement = f'In day {round} round, player_{player_id} said: "{statement}".'
            self.temp_game_log.append(transferred_statement)
            self.public_information(transferred_statement, player_id)
            self.game_log[round]['day'] = self.temp_game_log
            self.save_log()


        # Now we collect the reliability matrix and decision matrix from available players
        decision_matrix = np.zeros((self.num_players, self.num_players))
        reliability_matrix = np.zeros((self.num_players, self.num_players))
        for player_id in self.statement_order:
            if player_id == self.sheriff_id:
                continue
            available_players = self.get_remaining_players()
            selected_id = self.players[player_id-1].pseudo_vote(round, available_players)
            if selected_id != -1:
                decision_matrix[player_id-1][selected_id-1] = 1
            
            reliability = []
            for id in self.all_players:
                if id == player_id:
                    reliability.append(0)
                elif f'player_{id}' in self.players[player_id-1].reliability:
                    reliability.append(self.players[player_id-1].reliability[f'player_{id}'])
                else:
                    # data missing
                    reliability.append(-1)
            reliability_matrix[player_id-1] = reliability

        self.decision_and_reliability[round]['decision'] = decision_matrix
        self.decision_and_reliability[round]['reliability'] = reliability_matrix
        self.save_decision_and_reliability()


        # The Sheriff makes a statement
        available_players = self.get_remaining_players()
        statement = self.sheriff.day_discussion(round, available_players)
        self.sheriff.save_log(self.start_time)
        if statement is None:
            transferred_statement = f'In day {round} round, player_{player_id} said nothing.'
            self.temp_game_log.append(transferred_statement)
            self.public_information(transferred_statement, player_id)
            self.game_log[round]['day'] = self.temp_game_log
            self.save_log()


        transferred_statement = f'In day {round} round, player_{player_id} said: "{statement}".'
        self.temp_game_log.append(transferred_statement)
        self.public_information(transferred_statement, self.sheriff.player_id)
        self.game_log[round]['day'] = self.temp_game_log
        self.save_log()


        # Now begin to vote and collect the reliability and decision matrix again   
        decision_matrix_post = np.zeros((self.num_players, self.num_players))
        reliability_matrix_post = np.zeros((self.num_players, self.num_players))

        voting_results = {}
        for player_id in self.statement_order:
            available_players = self.get_remaining_players()
            selected_id = self.players[player_id-1].vote(round, available_players)
            self.players[player_id - 1].save_log(self.start_time)
            if selected_id == -1:
                msg = f'In day {round} round, player_{player_id} did not vote.'
                self.temp_game_log.append(msg)
                self.announcement(msg, player_id)
            else:
                decision_matrix_post[player_id-1][selected_id-1] = 1
                msg = f'In day {round} round, player_{player_id} voted to eliminate player_{selected_id}.'
                self.temp_game_log.append(msg)
                self.announcement(msg, player_id)
                self.game_log[round]['day'] = self.temp_game_log
                self.save_log()

                if selected_id not in voting_results:
                    voting_results[selected_id] = []
                voting_results[selected_id].append(f'player_{player_id}')


            reliability = []
            for id in self.all_players:
                if id == player_id:
                    reliability.append(0)
                elif f'player_{id}' in self.players[player_id-1].reliability:
                    reliability.append(self.players[player_id-1].reliability[f'player_{id}'])
                else:
                    # data missing
                    reliability.append(-1)
            reliability_matrix_post[player_id-1] = reliability
        
        self.decision_and_reliability[round]['decision_post'] = decision_matrix_post
        self.decision_and_reliability[round]['reliability_post'] = reliability_matrix_post
        self.save_decision_and_reliability()

        self.temp_game_log.append(voting_results)
        self.game_log[round]['day'] = self.temp_game_log
        self.save_log()

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
            self.remaining_players.remove(temp_id)
            if (self.sheriff is not None) and (temp_id == self.sheriff.player_id):
                time_str = f'day {round}'
                self.convey_sheriff(time_str)
        else:
            msg = f'In day {round} round, more than two players got the same votes and no player was eliminated.'

        self.temp_game_log.append(msg)
        self.announcement(msg)
        self.game_log[round]['day'] = self.temp_game_log
        self.game_log[round]['remaining_player'] = self.get_remaining_players()
        self.save_log()


    def announcement(self, msg: str, main_player_id = None) -> None:
        if main_player_id is None:
            # The message is for all players
            for player_id in self.remaining_players:
                self.players[player_id-1].recieve_fact(msg)
        else:
            for player_id in self.remaining_players:
                if player_id == main_player_id:
                    continue
                self.players[player_id-1].recieve_fact(msg)


    def public_information(self, msg: str, speaker_id: int):
        for player_id in self.remaining_players:
            self.players[player_id-1].receive_public_info(msg, f'player_{speaker_id}')


    # def secret_information(self, msg: str, player_id: int) -> None:
    #     self.players[player_id].recieve_message(msg)
    #     pass


    def get_remaining_players(self) -> list:
        return [f'player_{i}' for i in self.remaining_players]


    def get_teammate_id(self, werewolf_id: int) -> int:
        return self.werewolf_id.difference(set([werewolf_id])).pop()


    def save_log(self):
        dir_path = os.path.join(self.log_path, self.start_time)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if self.random_seed is None:
            with open(os.path.join(dir_path, 'game_log.json'), "w", encoding = 'utf8') as fp:
                json.dump(self.game_log, fp, indent = 4)
        else:
            with open(os.path.join(dir_path, f'game_log_{self.random_seed}.json'), "w", encoding = 'utf8') as fp:
                json.dump(self.game_log, fp, indent = 4)


    def save_decision_and_reliability(self):
        dir_path = os.path.join(self.result_path, self.start_time)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if self.random_seed is None:
            with open(os.path.join(dir_path, 'results.pkl'), 'wb') as f:
                pkl.dump(self.decision_and_reliability, f)
        else:
            with open(os.path.join(dir_path, f'results_{self.random_seed}.pkl'), 'wb') as f:
                pkl.dump(self.decision_and_reliability, f)


    def game_end(self):
        num_villager = 0
        num_werewolf = 0
        for player_id in self.remaining_players:
            temp_role = self.players[player_id-1].role
            if temp_role == 'Werewolf':
                num_werewolf += 1
            else:
                num_villager += 1
        
        if num_werewolf >= num_villager:
            self.logger.info('Game End, the Werewolves win.')
            self.game_log['result'] = 'Game End, the Werewolves win.'
            self.save_log()
            return True
        elif num_werewolf == 0:
            self.logger.info('Game End, the Villagers win.')
            self.game_log['result'] = 'Game End, the Villagers win.'
            self.save_log()
            return True

        else:
            return False