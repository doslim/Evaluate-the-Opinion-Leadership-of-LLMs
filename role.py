from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import numpy as np
import random
import json
from json.decoder import JSONDecodeError
import torch
import os
import time
from copy import deepcopy
from logger import Logging
from zhipuai import ZhipuAI
from openai import OpenAI
from utils import extract_json_strings

from prompt import GAME_RULE_TEMPLATE, WEREWOLF_ACTION_TEMPLATE, SEER_ACTION_TEMPLATE, GUARD_ACTION_TEMPLATE, \
STATEMENT_TEMPLATE, WEREWOLF_VOTING_TMEPLATE, VILLAGER_VOTING_TMEPLATE, REASONING_TEMPLATE, STATEMENT_ORDER_TEMPLATE, \
SHERRIF_STATEMENT_TEMPLATE, SEER_ACTION_HUMAN_TEMPLATE, WEREWOLF_ACTION_HUMAN_TEMPLATE, GUARD_ACTION_HUMAN_TEMPLATE, \
STATEMENT_HUMAN_TEMPLATE, WEREWOLF_VOTING_HUMAN_TMEPLATE, VILLAGER_VOTING_HUMAN_TMEPLATE, REASONING_HUMAN_TEMPLATE, \
VILLAGER_VOTING_HUMAN_PSEUDO_TMEPLATE, WEREWOLF_VOTING_HUMAN_PSEUDO_TMEPLATE, ELECTION_STATEMENT_TEMPLATE, \
ELECTION_VOTE_TEMPLATE, ELECTION_VOTE_HUMAN_TEMPLATE, STATEMENT_ORDER_HUMAN_TEMPLATE


# The base class for different roles
class BaseAgent:
    def __init__(self, model_path: str, log_path: str, model_outside = False, model = None, logger_level = None) -> None:
        self.model_path = model_path
        self.log_path = log_path
        self.history =  []
        self.fact = []
        self.potential_truth = [] # a list of dictionary
        self.potential_deception = [] # a list of dictionary
        self.public_info = [] # a list of dictionary  [{'player_1': In day 1 round, player_1 says: ...}]
        self.reliability = {}
        
        self.temp_fact = []
        self.temp_truth = []
        self.temp_deceptions = []
        self.reasoning_history = []
        self.action_history = []
        self.statement_history = []
        self.public_info_history = []

        if logger_level is None:
            self.logger = Logging().log(log_path, level='DEBUG')
        else:
            self.logger = Logging().log(log_path, level=logger_level)
        if model_path in ['glm-3', 'glm-4', 'gpt-4', 'gpt-3.5']:
            self.local_model = False
            self.model_name = self.model_path
            if model_path in ['glm-3']:
                self.model = ZhipuAI(api_key='')
            if model_path in ['gpt-4', 'gpt-3.5']:
                self.model = OpenAI(api_key='', timeout=60)
            if model_path in ['glm-4']:
                self.model = ZhipuAI(api_key='')
        else:
            self.local_model = True
            if model_outside:
                # do not need to load the model
                self.model = model
            else:
                # need to load the model
                self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, load_in_8bit = True, device_map = 'auto')

            if 'chatglm' in self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)   
                self.model_name = 'chatglm3-6b'
            
            if 'Baichuan' in self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.generation_config = GenerationConfig.from_pretrained(model_path)     
                self.model_name = 'Baichuan2-13b'
            
            if 'Mistral' in self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model_name = 'Mistral-7B-Instruct'

            if 'Yi' in self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model_name = 'Yi-34B'

            if 'internlm' in self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model_name = 'InternLM-20B'


        self.role = ''
        self.player_id = 0


    def reset_backend(self, model_path: str, model):
        self.model_path = model_path
        if model_path in ['glm-3', 'glm-4', 'gpt-4', 'gpt-3.5']:
            self.local_model = False
            self.model_name = self.model_path
            if model_path in ['glm-3']:
                self.model = ZhipuAI(api_key='')
            if model_path in ['gpt-4', 'gpt-3.5']:
                self.model = OpenAI(api_key='', timeout=60)
            if model_path in ['glm-4']:
                self.model = ZhipuAI(api_key='')
        else:
            self.local_model = True
            self.model = model

            if 'chatglm' in self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)   
                self.model_name = 'chatglm3-6b'
            
            if 'Baichuan' in self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.generation_config = GenerationConfig.from_pretrained(model_path)     
                self.model_name = 'Baichuan2-13b'
            
            if 'Mistral' in self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model_name = 'Mistral-7B-Instruct'

            if 'Yi' in self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model_name = 'Yi-34B'

            if 'internlm' in self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model_name = 'InternLM-20B'


    def set_player_id(self, id: int):
        self.player_id = id
        if f'You are player_{id}' not in self.fact:
            self.fact.append(f'You are player_{id}.')
        if f'You are a {self.role}' not in self.fact:
            self.fact.append(f'You are a {self.role}.')


    def receive_public_info(self, msg: str, player: str) -> None:
        self.public_info.append({player: msg})


    def recieve_fact(self, msg: str) -> None:
        if msg not in self.fact:
            self.fact.append(msg)

    # def generate(self, prompt: str, temperature = 0.1, do_sample = True) -> str:        
    #     prompt_tensor = self.tokenizer(prompt, return_tensors='pt', truncation= True, max_length= 4096)['input_ids']
    #     # self.model.to(self.device)
    #     # prompt_tensor = prompt_tensor.to(self.device)
    #     prompt_tensor = prompt_tensor.cuda()
    #     response = self.model.generate(prompt_tensor, temperature = temperature, do_sample = do_sample)
    #     response_text = self.tokenizer.batch_decode(response)[0]

    #     del prompt_tensor
    #     # self.model.to('cpu')
    #     response_text = response_text.replace(prompt, "")
    #     response_text = response_text.replace('</s>', "")
    #     response_text = response_text.replace('```', "")


        # return response_text


    def generate(self, prompt: str, temperature = 0.1, do_sample = True) -> str:
        '''
        Generate response
        '''
        message = []
        message.append({'role': 'user', 'content': prompt})

        if self.local_model and 'chatglm' in self.model_path:
            response_text, _ = self.model.chat(self.tokenizer, prompt, do_sample = do_sample, temperature = temperature)

            return response_text

        elif self.local_model and 'Baichuan' in self.model_path:
            self.generation_config.temperature = temperature
            self.generation_config.do_sample = do_sample

            # for _ in range(3):
            #     response_text = self.model.chat(self.tokenizer, message, stream=False, generation_config = self.generation_config)
            #     response_text = response_text.replace('```', "") 
            #     begin_idx = response_text.find('{')
            #     try:
            #         parsing_result = json.loads(response_text[begin_idx:])
            #         break
            #     except Exception as e:
            #         self.logger.error('Invalid output format')
            #         self.logger.exception('An error occurs: {}'.format(e))
            #         self.logger.info(response_text)
            #         message.append({'role':'assistant', 'content':response_text})
            #         format_instruction = 'Please response in JSON format.'
            #         message.append({'role': 'user', 'content': format_instruction})

            response_text = self.model.chat(self.tokenizer, message, stream=False, generation_config = self.generation_config)
            return response_text
        
        elif self.local_model and 'Mistral' in self.model_path:
            # transformers >= 4.34
            
            tokenized_chat = self.tokenizer.apply_chat_template(message, tokenize=True, return_tensors="pt")
            tokenized_chat = tokenized_chat.to('cuda')
            self.model.generation_config.do_sample = do_sample
            self.model.generation_config.temperature = temperature
            self.model.generation_config.max_length = 8192
            response = self.model.generate(input_ids = tokenized_chat)
            response_text = self.tokenizer.decode(response[0,tokenized_chat.shape[1]:])

            return response_text

        elif self.local_model and 'Yi' in self.model_path:
 
            chat_data = self.tokenizer.apply_chat_template(message, tokenize = False)
            output = self.model(chat_data, do_sample = do_sample, temperature = temperature, max_new_tokens=1024)
            response_text = output[0]['generated_text'].replace(chat_data, '').strip()

            return response_text

        elif self.local_model and 'internlm' in self.model_path:
            response_text, _ = self.model.chat(self.tokenizer, prompt, do_sample = do_sample, temperature = temperature)

            return response_text
        elif self.model_name == 'glm-3':
            try:
                response =  self.model.chat.completions.create(
                    model = "glm-3-turbo",  # glm-4
                    messages = message,
                    do_sample = do_sample,
                    temperature = temperature
                )
                response_text = response.choices[0].message.content
                response_text = response_text.replace('```', "") 
            except Exception as e:
                self.logger.exception(f"An exception occurs: {e}")
                self.logger.error('Generation Error')
                # self.logger.info(response)
                return ""
                
                # try:
                #     begin_idx = response_text.find('{')
                #     json_strings = extract_json_strings(response)
                #     parsing_result = json.loads(json_strings[0])
                #     break
                # except JSONDecodeError:
                #     self.logger.error('Invalid output format')
                #     self.logger.info(response_text)
                #     message.append({'role':'assistant', 'content': response_text})
                #     format_instruction = 'Please response in JSON format.'
                #     message.append({'role': 'user', 'content': format_instruction})
            return response_text
        
        elif self.model_name == 'glm-4':
            try:
                response =  self.model.chat.completions.create(
                    model = "glm-4",  
                    messages = message,
                    do_sample = do_sample,
                    temperature = temperature
                )
                response_text = response.choices[0].message.content
                response_text = response_text.replace('```', "") 
            except Exception as e:
                self.logger.exception(f"An exception occurs: {e}")
                self.logger.error('Generation Error')
                self.logger.info(response)
                return ""

            return response_text
        elif self.model_name == 'gpt-4':
            if not do_sample:
                temperature = 0
            try:
                response =  self.model.chat.completions.create(
                    model = "gpt-4-turbo-preview",  
                    messages = message,
                    temperature = temperature,
                    n = 1
                )
                response_text = response.choices[0].message.content
                response_text = response_text.replace('```', "") 
            except Exception as e:
                self.logger.exception(f"An exception occurs: {e}")
                self.logger.error('Generation Error')
                self.logger.info(response)
                return ""

            return response_text

    def day_discussion(self, round: int, remaining_player, sherrif = False) -> str:
        rule = GAME_RULE_TEMPLATE.format(self.player_id, self.role)
        if sherrif:
            statement = SHERRIF_STATEMENT_TEMPLATE.format(round, self.player_id, self.role)
        else:
            statement = STATEMENT_TEMPLATE.format(round, self.player_id, self.role)
        time_str = f'day {round}'
        reasoning_result = self.deductive_reasoning(remaining_player, time_str, clear = True, remove = False)
        self.reasoning_history.append({f'day {round}': reasoning_result})
        context = self.organize_context(remaining_player)

        prompt = rule + '\n' + context + '\n' + statement

        response_text = self.generate(prompt)
        begin_idx = response_text.find('{')
        try:
            json_strings = extract_json_strings(response_text[begin_idx:])
            statement_result = json.loads(json_strings[0])
            words = statement_result['statement']
            self.fact.append(f'In day {round}, you said: {words}')
            self.statement_history.append({f'day {round}': words})

            return words
        except Exception as e:
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.info(response_text)
            self.fact.append(f'In day {round}, you say nothing')
            self.statement_history.append({f'day {round}': 'Nothing'})
            return None
    

    def election_statement(self, remaining_player) -> str:
        rule = GAME_RULE_TEMPLATE.format(self.player_id, self.role)
        context = self.organize_context(remaining_player)
        statement = ELECTION_STATEMENT_TEMPLATE.format(self.player_id, self.role)
        prompt = rule + '\n' + context + '\n' + statement

        response_text = self.generate(prompt)
        begin_idx = response_text.find('{')
        try:
            json_strings = extract_json_strings(response_text[begin_idx:])
            statement_result = json.loads(json_strings[0])
            words = statement_result['statement']
            self.fact.append(f'During the election phase of day 1, you said: {words}')
            self.statement_history.append({'election phase': words})

            return words
        except Exception as e:
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.info(response_text)
            self.fact.append('During the election phase of day 1, you said nothing')
            self.statement_history.append({'election phase': 'Nothing'})
            return None


    def election_vote(self, sheriff_candidate, remaining_player) -> int:
        rule = GAME_RULE_TEMPLATE.format(self.player_id, self.role)
        available_player_id = [int(i.split('_')[1]) for i in sheriff_candidate]
        action = ['vote for' + i for i in sheriff_candidate]
        action = ', '.join(action)
        sheriff_candidate_str = ', '.join(sheriff_candidate)
        voting = ELECTION_VOTE_TEMPLATE.format(sheriff_candidate_str, self.player_id, self.role, action)
        time_str = 'day 1'
        self.deductive_reasoning(sheriff_candidate, time_str, clear = True, remove = True)
        context = self.organize_context(remaining_player)
        prompt = rule + '\n' + context + '\n' + voting

        response_text = self.generate(prompt, do_sample=False)
        begin_idx = response_text.find('{')
        try:
            json_strings = extract_json_strings(response_text[begin_idx:])
            voting_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.error('Invalid output of voting')
            self.logger.info(response_text)

            self.fact.append('You did not vote during the election phase of day 1')
            self.action_history.append({'election phase': 'You did not vote'})
            return -1

        try:
            if voting_result['action'].split('_')[1].isdigit():
                selected_player = int(voting_result['action'].split('_')[1])
                if selected_player in available_player_id:
                    self.fact.append(f'You voted for player_{selected_player} during the election phase of day 1')
                    self.action_history.append({'election phase': f'You voted for player_{selected_player}'})
                else:
                    self.logger.error('Invalid voting actions or do not vote')
                    self.logger.info(voting_result)
                    selected_player = -1
                    self.fact.append('You did not vote during the election phase of day 1')
                    self.action_history.append({'election phase': 'You did not vote'})
            else:
                self.logger.error('Invalid voting actions or do not vote')
                self.logger.info(voting_result)
                selected_player = -1
                self.fact.append('You did not vote during the election phase of day 1')
                self.action_history.append({'election phase': 'You did not vote'})
        except Exception as e:
            self.logger.error('Invalid voting actions or do not vote')
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.info(voting_result)
            selected_player = -1
            self.fact.append('You did not vote during the election phase of day 1')
            self.action_history.append({'election phase': 'You did not vote'})

        return selected_player

    def vote(self, round: int, remaining_player) -> int:
        rule = GAME_RULE_TEMPLATE.format(self.player_id, self.role)
        available_player_id = [int(i.split('_')[1]) for i in remaining_player]
        action = ['vote to eliminate ' + i for i in remaining_player]
        action = ', '.join(action)
        if self.role in ['Werewolf']:
            voting = WEREWOLF_VOTING_TMEPLATE.format(round, self.player_id, self.role, action)
        else:
            voting = VILLAGER_VOTING_TMEPLATE.format(round, self.player_id, self.role, action)

        time_str = f'day {round}'
        self.deductive_reasoning(remaining_player, time_str, True, remove = True)
        context = self.organize_context(remaining_player)
        prompt = rule + '\n' + context + '\n' + voting

        response_text = self.generate(prompt, do_sample=False)
        begin_idx = response_text.find('{')
        try:
            json_strings = extract_json_strings(response_text[begin_idx:])
            voting_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.error('Invalid output of voting')
            self.logger.info(response_text)

            self.fact.append(f'You did not vote in day {round}')
            self.action_history.append({f'day {round}': 'You did not vote'})
            return -1

        try:
            if voting_result['action'].split('_')[1].isdigit():
                selected_player = int(voting_result['action'].split('_')[1])
                if selected_player in available_player_id:
                    self.fact.append(f'You voted to eliminate player_{selected_player} in day {round}')
                    self.action_history.append({f'day {round}': f'You voted to eliminate player_{selected_player}'})
                else:
                    self.logger.error('Invalid voting actions or do not vote')
                    self.logger.info(voting_result)
                    selected_player = -1
                    self.fact.append(f'You did not vote in day {round}')
                    self.action_history.append({f'day {round}': 'You did not vote'})
            else:
                self.logger.error('Invalid voting actions or do not vote')
                self.logger.info(voting_result)
                selected_player = -1
                self.fact.append(f'You did not vote in day {round}')
                self.action_history.append({f'day {round}': 'You did not vote'})
        except Exception as e:
            self.logger.error('Invalid voting actions or do not vote')
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.info(voting_result)
            selected_player = -1
            self.fact.append(f'You did not vote in day {round}')
            self.action_history.append({f'day {round}': f'You did not vote'})

        return selected_player


    def determine_statement_order(self, round: int, remaining_player):
        rule = GAME_RULE_TEMPLATE.format(self.player_id, self.role)
        remaining_player_ids = [int(i.split('_')[1]) for i in remaining_player]
        your_idx = remaining_player_ids.index(self.player_id)
        if your_idx == 0:
            left_idx = -1
            right_idx = 1
        elif your_idx == len(remaining_player_ids) - 1:
            right_idx = 0
            left_idx = your_idx - 1
        else:
            left_idx = your_idx - 1
            right_idx = your_idx + 1
        
        left_player = remaining_player_ids[left_idx]
        right_player = remaining_player_ids[right_idx]
        action_set = [f'player_{left_player}', f'player_{right_player}']
        time_str = f'day {round}'
        self.deductive_reasoning(remaining_player, time_str, clear = True, remove = False)
        statement_order = STATEMENT_ORDER_TEMPLATE.format(round, self.player_id, self.role, action_set)
        context = self.organize_context(remaining_player)
        prompt = rule + '\n' + context + '\n' + statement_order

        response_text = self.generate(prompt, do_sample=False)
        begin_idx = response_text.find('{')
        try:
            json_strings = extract_json_strings(response_text[begin_idx:])
            order_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.error('Invalid output of deciding the statement order.')
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.info(response_text)

            self.fact.append(f'You chose player_{left_player} to make a statement first in day {round}')
            
            return self.take_order(remaining_player_ids, left_player, left = True)

        try:
            if order_result['action'].split('_')[1].isdigit():
                selected_player = int(order_result['action'].split('_')[1])
                if selected_player == left_player:
                    self.fact.append(f'You chose player_{left_player} to make a statement first in day {round}')
                    self.action_history.append({f'day {round}': f'You chose player_{left_player} to make a statement first'})
                    return self.take_order(remaining_player_ids, left_player, left = True)
                elif selected_player == right_player:
                    self.fact.append(f'You chose player_{right_player} to make a statement first in day {round}')
                    self.action_history.append({f'day {round}': f'You chose player_{right_player} to make a statement first'})
                    return self.take_order(remaining_player_ids, right_player, left = False)
                else:
                    self.logger.error('Invalid decision on the order of statement.')
                    self.logger.info(order_result)

                    self.fact.append(f'You chose player_{left_player} to make a statement first in day {round}')
                    self.action_history.append({f'day {round}': f'You chose player_{left_player} to make a statement first'})
                    return self.take_order(remaining_player_ids, left_player, left = True)
            else:
                self.logger.error('Invalid decision on the order of statement.')
                self.logger.info(order_result)
                self.fact.append(f'You chose player_{left_player} to make a statement first in day {round}')
                self.action_history.append({f'day {round}': f'You chose player_{left_player} to make a statement first'})
                return self.take_order(remaining_player_ids, left_player, left = True)
        except Exception as e:
            self.logger.error('Invalid decision on the order of statement.')
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.info(order_result)
            self.fact.append(f'You chose player_{left_player} to make a statement first in day {round}')
            self.action_history.append({f'day {round}': f'You chose player_{left_player} to make a statement first'})
            return self.take_order(remaining_player_ids, left_player, left = True)


    def pseudo_vote(self, round: int, remaining_player) -> int:
        '''
        The agent is asked to 'vote' before the Sherrif makes a statement

        Note:
        In this method, we DO NOT CHANGE self.fact, self.potential_truth, self.potential_deceptions.

        The voting result will not be added into self.action_history and self.fact.
        '''

        rule = GAME_RULE_TEMPLATE.format(self.player_id, self.role)
        available_player_id = [int(i.split('_')[1]) for i in remaining_player]
        action = ['vote to eliminate ' + i for i in remaining_player]
        action = ', '.join(action)
        if self.role in ['Werewolf']:
            voting = WEREWOLF_VOTING_TMEPLATE.format(round, self.player_id, self.role, action)
        else:
            voting = VILLAGER_VOTING_TMEPLATE.format(round, self.player_id, self.role, action)

        time_str = f'day {round}'
        self.deductive_reasoning(remaining_player, time_str, clear = True, remove = False, pseudo = True)
        context = self.organize_context(remaining_player, pseudo = True)
        prompt = rule + '\n' + context + '\n' + voting

        response_text = self.generate(prompt, do_sample=False)
        begin_idx = response_text.find('{')
        try:
            json_strings = extract_json_strings(response_text[begin_idx:])
            voting_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.error('Invalid output of voting')
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.info(response_text)
            return -1

        try:
            if voting_result['action'].split('_')[1].isdigit():
                selected_player = int(voting_result['action'].split('_')[1])
                if selected_player not in available_player_id:
                    self.logger.error('Invalid voting actions or do not vote')
                    self.logger.info(voting_result)
                    selected_player = -1
            else:
                self.logger.error('Invalid voting actions or do not vote')
                self.logger.info(voting_result)
                selected_player = -1
        except Exception as e:
            self.logger.error('Invalid voting actions or do not vote')
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.info(voting_result)
            selected_player = -1

        return selected_player



    def deductive_reasoning(self, remaining_player, time_str, clear = True, remove = False, pseudo = False):
        '''
        Analyse the reliability of each other player
        
        - pseudo: whether to modify the fact, potential truth and potential deceptions
        ''' 
        rule = GAME_RULE_TEMPLATE.format(self.player_id, self.role)
        context = self.organize_context(remaining_player, pseudo)
        avaliable_player = deepcopy(remaining_player)
        if f'player_{self.player_id}' in avaliable_player:
            avaliable_player.remove(f'player_{self.player_id}')
        # player_except_myself = ', '.join(remaining_player)
        # reasoning = REASONING_TEMPLATE.format(time_str, self.player_id, self.role, player_except_myself)

        useful_information = set()
        results_all = {}
        for i in avaliable_player:
            reasoning = REASONING_TEMPLATE.format(time_str, self.player_id, self.role, i, i)
            prompt = rule + '\n' + context + '\n' + reasoning
            # print(prompt)
            response_text = self.generate(prompt, do_sample=False)
            # print(response_text)
            begin_idx = response_text.find('{')
            try:
                json_strings = extract_json_strings(response_text[begin_idx:])
                reasoning_result = json.loads(json_strings[0])
                results_all[i] = reasoning_result
            except Exception as e:
                self.logger.error('Invalid output of reliability reasoning.')
                self.logger.exception("An error occurred: {}".format(e))
                self.logger.info(response_text)
                results_all[i] = response_text
                continue

            # print(reasoning_result)
            # update reliability
            for player in reasoning_result:
                try:
                    player_role = reasoning_result[player]['role']
                    if player_role == 'Werewolf' and self.role != 'Werewolf':
                        self.reliability[player] = 11 - int(reasoning_result[player]['confidence'])
                    else:
                        self.reliability[player] = int(reasoning_result[player]['confidence'])
                    
                    useful_information = useful_information.union(set(reasoning_result[player]['evidence']))
                except Exception as e:
                    self.logger.error(f'Failed to update reliability of player_{player}')
                    self.logger.exception("An error occurred: {}".format(e))
                    self.logger.info(reasoning_result)

        if remove:
            self.remove_useless_context(useful_information)
        if pseudo:
            self.pseudo_add_new_context()
        else:
            self.add_new_context(clear)


        return results_all
    

    def organize_context(self, remaining_player, pseudo = False) -> str:
        '''
        If pseudo == True, we use temporary fact, potential truth and potential deception.
        '''

        if pseudo: 
            str_remaining_palyer = ', '.join(remaining_player)
            a = f'All information you can leverage is listed below. \n Remaining Players: {str_remaining_palyer}. \n'
            b = 'The following information is true.\n'
            for i in range(1, len(self.temp_fact) + 1):
                temp = f'[{i}] ' + self.temp_fact[i-1] + '\n'
                b = b + temp 
            
            c = 'The following information might be true.\n'
            if len(self.temp_truth) == 0:
                c += 'None \n'
            i = len(self.temp_fact)
            for j in range(1, len(self.temp_truth) + 1):
                temp = f'[{i+j}]' + list(self.temp_truth[j-1].values())[0] + '\n'
                c = c + temp

            d = 'The following information might be false.\n'
            j = len(self.temp_truth)
            if len(self.temp_deceptions) == 0:
                d += 'None \n'
            for k in range(1, len(self.temp_deceptions) + 1):
                temp = f'[{i+j+k}]' + list(self.temp_deceptions[k-1].values())[0] + '\n'
                d = d + temp
            
            e = 'The following public information still needs further clarification. \n'
            if len(self.public_info) == 0:
                e += 'None \n'
            for k in range(1, len(self.public_info) + 1):
                temp =  list(self.public_info[k-1].values())[0] + '\n'
                e = e + temp

            return a + b + c + d + e
        

        str_remaining_palyer = ', '.join(remaining_player)
        a = f'All information you can leverage is listed below. \n Remaining Players: {str_remaining_palyer}. \n'
        b = 'The following information is true.\n'
        for i in range(1, len(self.fact) + 1):
            temp = f'[{i}] ' + self.fact[i-1] + '\n'
            b = b + temp 
        
        c = 'The following information might be true.\n'
        if len(self.potential_truth) == 0:
            c += 'None \n'
        i = len(self.fact)
        for j in range(1, len(self.potential_truth) + 1):
            temp = f'[{i+j}]' + list(self.potential_truth[j-1].values())[0] + '\n'
            c = c + temp

        d = 'The following information might be deceptive.\n'
        j = len(self.potential_truth)
        if len(self.potential_deception) == 0:
            d += 'None \n'
        for k in range(1, len(self.potential_deception) + 1):
            temp = f'[{i+j+k}]' + list(self.potential_deception[k-1].values())[0] + '\n'
            d = d + temp

        e = 'The following public information still needs further clarification. \n'
        if len(self.public_info) == 0:
            e += 'None \n'
        for k in range(1, len(self.public_info) + 1):
            temp =  list(self.public_info[k-1].values())[0] + '\n'
            e = e + temp
        return a + b + c + d + e
    

    def remove_useless_context(self, useful_information) -> None:
        num_fact = len(self.fact)
        num_potential_truth = len(self.potential_truth)
        num_potential_deception = len(self.potential_deception)
        num_context = num_fact + num_potential_truth + num_potential_deception
        useless_information = set(list(range(1, num_context + 1))).difference(set(useful_information))
        useless_information = np.array(list(useless_information))

        potential_truth_useless = useless_information[(useless_information > num_fact) & (useless_information <= num_fact + num_potential_truth)] - num_fact - 1
        use_less_element = [self.potential_truth[i] for i in potential_truth_useless]
        for i in use_less_element:
            self.potential_truth.remove(i)
        
        potential_deception_useless = useless_information[(useless_information > num_fact + num_potential_truth)] - num_fact - num_potential_truth - 1
        use_less_element = [self.potential_deception[i] for i in potential_deception_useless]
        for i in use_less_element:
            self.potential_deception.remove(i)
        # for i in potential_deception_useless:
        #     self.potential_deception.pop(i-1)


    def add_new_context(self, clear) -> None:
        '''
        Add public info into the context according to the reliability
        '''
        for info in self.public_info:
            player = list(info.keys())[0]
            if player not in self.reliability:
                self.potential_deception.append({player: info[player]})
                for j in self.potential_truth:
                    if player in j:
                        self.potential_deception.append(j)
                        self.potential_truth.remove(j)
                continue

            if self.reliability[player] >= 6:
                self.potential_truth.append({player: info[player]})
                for j in self.potential_deception:
                    if player in j:
                        self.potential_truth.append(j)
                        self.potential_deception.remove(j)
            else:
                self.potential_deception.append({player: info[player]})
                for j in self.potential_truth:
                    if player in j:
                        self.potential_deception.append(j)
                        self.potential_truth.remove(j)
        if clear:
            self.public_info_history.extend(self.public_info)
            self.public_info = []


    def pseudo_add_new_context(self) -> None:
        '''
        This method is first DEEP copy the fact, potential truch, and potential deceptions and then add new public information.
        '''
        self.temp_fact = deepcopy(self.fact)
        self.temp_truth = deepcopy(self.potential_truth)
        self.temp_deceptions = deepcopy(self.potential_deception)

        for info in self.public_info:
            player = list(info.keys())[0]
            if player not in self.reliability:
                self.temp_deceptions.append({player: info[player]})
                for j in self.temp_truth:
                    if player in j:
                        self.temp_deceptions.append(j)
                        self.temp_truth.remove(j)
                continue

            if self.reliability[player] >= 6:
                self.temp_truth.append({player: info[player]})
                for j in self.temp_deceptions:
                    if player in j:
                        self.temp_truth.append(j)
                        self.temp_deceptions.remove(j)
            else:
                self.temp_deceptions.append({player: info[player]})
                for j in self.temp_truth:
                    if player in j:
                        self.temp_deceptions.append(j)
                        self.temp_truth.remove(j)


    def take_order(self, remaining_players_ids, first_player_id, left = True) -> list:
        statement_order = []
        first_player_idx = remaining_players_ids.index(first_player_id)
        for i in range(len(remaining_players_ids)):
            if left:
                statement_order.append(remaining_players_ids[first_player_idx - i])
            else:
                temp = first_player_idx + i
                if temp > len(remaining_players_ids) - 1:
                    temp = temp - len(remaining_players_ids)
                statement_order.append(remaining_players_ids[temp])
        
        return statement_order


    def save_log(self, experiment_start_time = None):
        player_log = {'model_path': self.model_path,
                      'model_name': self.model_name,
                      'player_id': self.player_id,
                      'role': self.role,
                      'fact': self.fact,
                      'potential_truth': self.potential_truth,
                      'potential_deception': self.potential_deception,
                      'reliability': self.reliability,
                      'action_history': self.action_history, 
                      'statement_history': self.statement_history,
                      'reasoning_history': self.reasoning_history,
                      'public_info_history': self.public_info_history}
        
        if experiment_start_time is not None:
            dir_path = os.path.join(self.log_path, experiment_start_time)
        else:
            dir_path = self.log_path
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        
        with open(os.path.join(dir_path, f'{self.player_id}_{self.role}.json'), "w", encoding = 'utf8') as fp:
            json.dump(player_log, fp, indent = 4)



class Villager(BaseAgent):
    def __init__(self, model_path: str, log_path: str, model_outside = False, model = None, logger_level = None) -> None:
        super().__init__(model_path, log_path, model_outside, model, logger_level)
        self.role = 'Villager'


class Werewolf(BaseAgent):
    def __init__(self, model_path: str, log_path: str, model_outside = False, model = None, logger_level = None) -> None:
        super().__init__(model_path, log_path, model_outside, model, logger_level)
        self.role = 'Werewolf'

    def take_night_action(self, round: int, teammate_id: int, remaining_player) -> int:
        # action = ', '.join([f'kill {i}' for i in remaining_player])
        available_player_id = [int(i.split('_')[1]) for i in remaining_player]
        action = [f'kill {i}' for i in remaining_player]
        were_wolf_action = WEREWOLF_ACTION_TEMPLATE.format(round, self.player_id, self.role, action, action)
        rule = GAME_RULE_TEMPLATE.format(self.player_id, self.role)
        time_str = f'night {round}'
        if round > 1:
            super().deductive_reasoning(remaining_player, time_str, clear = True)
        context = super().organize_context(remaining_player)

        prompt = rule + '\n' + context + '\n' + were_wolf_action
        # print(prompt)
        response_text = super().generate(prompt, do_sample = False)
        begin_idx = response_text.find('{')
        try:
            json_strings = extract_json_strings(response_text[begin_idx:])
            action_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.error('Invalid output of taking night action.')
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.info(response_text)

            selected_player_id = random.choice(available_player_id)
            self.fact.append(f'In night {round} round, you chose to kill player_{selected_player_id}.')
            self.action_history.append({f'night {round}': f'You chose to kill player_{selected_player_id}.'})
            return selected_player_id

        
        try:
            if action_result['action'].split('_')[1].isdigit():
                selected_player_id = int(action_result['action'].split('_')[1])
                if selected_player_id in available_player_id:
                    self.fact.append(f'In night {round} round, you chose to kill player_{selected_player_id}.')
                    self.action_history.append({f'night {round}': f'You chose to kill player_{selected_player_id}.'})
                    return selected_player_id
            else:
                self.logger.error("Cannot parse the action")
                self.logger.info(action_result)
        except Exception as e:
            self.logger.error("Cannot parse the action")
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.info(action_result)
            
        selected_player_id = random.choice(available_player_id)
        self.fact.append(f'In night {round} round, you chose to kill player_{selected_player_id}.')
        self.action_history.append({f'night {round}': f'You chose to kill player_{selected_player_id}.'})
        return selected_player_id


class Guard(BaseAgent):
    def __init__(self, model_path: str, log_path: str, model_outside = False, model = None, logger_level = None) -> None:
        super().__init__(model_path, log_path, model_outside, model, logger_level)
        self.role = 'Guard'

    def take_night_action(self, round: int, remaining_player) -> int:
        # action = ', '.join([f'protect {i}' for i in remaining_player])
        available_player_id = [int(i.split('_')[1]) for i in remaining_player]
        action = [f'protect {i}' for i in remaining_player]
        guard_action = GUARD_ACTION_TEMPLATE.format(round, self.player_id, self.role, action, action)
        rule = GAME_RULE_TEMPLATE.format(self.player_id, self.role)
        time_str = f'night {round}'
        if round > 1:
            super().deductive_reasoning(remaining_player, time_str, clear = True)
        context = super().organize_context(remaining_player)

        prompt = rule + '\n' + context + '\n' + guard_action
        # print(prompt)
        response_text = super().generate(prompt, do_sample = False)
        begin_idx = response_text.find('{')
        try:
            json_strings = extract_json_strings(response_text[begin_idx:])
            action_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.error('Invalid output of taking night action.')
            self.logger.info(response_text)
            self.logger.exception('An error occurs: {}'.format(e))

            selected_player_id = random.choice(available_player_id)
            self.fact.append(f'In night {round} round, you chose to protect player_{selected_player_id}.')
            self.action_history.append({f'night {round}': f'You chose to proctect player_{selected_player_id}.'})
            return selected_player_id
        
        try:
            if action_result['action'].split('_')[1].isdigit():
                selected_player_id = int(action_result['action'].split('_')[1])
                if selected_player_id in available_player_id:
                    self.fact.append(f'In night {round} round, you chose to protect player_{selected_player_id}.')
                    self.action_history.append({f'night {round}': f'You chose to proctect player_{selected_player_id}.'})
                    return selected_player_id
            else:
                self.logger.error("Cannot parse the action")
                self.logger.info(action_result)

        except Exception as e:
            self.logger.error("Cannot parse the action")
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.info(action_result)

        selected_player_id = random.choice(available_player_id)
        self.fact.append(f'In night {round} round, you chose to protect player_{selected_player_id}.')
        self.action_history.append({f'night {round}': f'You chose to proctect player_{selected_player_id}.'})
        return selected_player_id


class Seer(BaseAgent):
    def __init__(self, model_path: str, log_path: str, model_outside = False, model = None, logger_level = None):
        super().__init__(model_path, log_path, model_outside, model, logger_level)
        self.role = 'Seer'


    def take_night_action(self, round: int, remaining_player) -> int:
        # action = ', '.join([f'see {i}' for i in remaining_player])
        action = [f'see {i}' for i in remaining_player]
        available_player_id = [int(i.split('_')[1]) for i in remaining_player]
        seer_action = SEER_ACTION_TEMPLATE.format(round, self.player_id, self.role, action, action)
        rule = GAME_RULE_TEMPLATE.format(self.player_id, self.role)
        time_str = f'night {round}'
        if round > 1:
            super().deductive_reasoning(remaining_player, time_str, True)
        context = super().organize_context(remaining_player)

        prompt = rule + '\n' + context + '\n' + seer_action
        # print(prompt)
        response_text = super().generate(prompt, do_sample = False)
        begin_idx = response_text.find('{')
        try:
            json_strings = extract_json_strings(response_text[begin_idx:])
            action_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.error('Invalid output of taking night action.')
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.info(response_text)

            selected_player_id = random.choice(available_player_id)
            self.action_history.append({f'night {round}': f'You chose to see player_{selected_player_id}.'})
            return selected_player_id

        try:
            if action_result['action'].split('_')[1].isdigit():
                selected_player_id = int(action_result['action'].split('_')[1])
                if selected_player_id in available_player_id:
                    self.action_history.append({f'night {round}': f'You chose to see player_{selected_player_id}.'})
                    return selected_player_id
            else:
                self.logger.error("Cannot parse the action")
                self.logger.info(action_result)

        except Exception as e:
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.error("Cannot parse the action")
            self.logger.info(action_result)

        selected_player_id = random.choice(available_player_id)
        self.action_history.append({f'night {round}': f'You chose to see player_{selected_player_id}.'})
        return selected_player_id


class HumanPlayer:
    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self.history =  []
        self.fact = []
        self.potential_truth = [] # a list of dictionary
        self.potential_deception = [] # a list of dictionary
        self.public_info = [] # a list of dictionary  [{'player_1': In day 1 round, player_1 says: ...}]
        self.reliability = {}
        
        self.temp_fact = []
        self.temp_truth = []
        self.temp_deceptions = []
        self.reasoning_history = []
        self.action_history = []
        self.statement_history = []
        self.public_info_history = []

        self.logger = Logging().log(log_path, level='CRITICAL')
        self.role = ''
        self.player_id = 0


    def set_player_id(self, id: int):
        self.player_id = id
        if f'You are player_{id}' not in self.fact:
            self.fact.append(f'You are player_{id}.')
        if f'You are a {self.role}' not in self.fact:
            self.fact.append(f'You are a {self.role}.')


    def receive_public_info(self, msg: str, player: str) -> None:
        self.public_info.append({player: msg})


    def recieve_fact(self, msg: str) -> None:
        if msg not in self.fact:
            self.fact.append(msg)


    def take_order(self, remaining_players_ids, first_player_id, left = True) -> list:
        statement_order = []
        first_player_idx = remaining_players_ids.index(first_player_id)
        for i in range(len(remaining_players_ids)):
            if left:
                statement_order.append(remaining_players_ids[first_player_idx - i])
            else:
                temp = first_player_idx + i
                if temp > len(remaining_players_ids) - 1:
                    temp = temp - len(remaining_players_ids)
                statement_order.append(remaining_players_ids[temp])
        
        return statement_order

    # We require human player to take actions by providing instructions
    # Therefore, the following methods organize [instructions] for the human player. Then the instructions will presented on the screen.
    # We define additional methods to parse the action results of the human player. 
    def take_night_action(self, round: int, remaining_player) -> str:
        if self.role == 'Villager':
            return 'You are the Villager, so you don\'t need to act at night.'
        elif self.role == "Seer":
            action = [f'see {i}' for i in remaining_player]
            seer_action = SEER_ACTION_HUMAN_TEMPLATE.format(round, self.player_id, self.role, action)

            context = self.organize_context(remaining_player)
            return context + '\n' + seer_action
        elif self.role == "Guard":
            action = [f'protect {i}' for i in remaining_player]
            guard_action = GUARD_ACTION_HUMAN_TEMPLATE.format(round, self.player_id, self.role, action)

            context = self.organize_context(remaining_player)
            return context + '\n' + guard_action
        elif self.role == 'Werewolf':
            action = [f'kill {i}' for i in remaining_player]
            were_wolf_action = WEREWOLF_ACTION_HUMAN_TEMPLATE.format(round, self.player_id, self.role, action)

            context = self.organize_context(remaining_player)
            return context + '\n' + were_wolf_action

    
    def determine_statement_order(self, round: int, remaining_player):
        context = self.organize_context(remaining_player)
        remaining_player_ids = [int(i.split('_')[1]) for i in remaining_player]
        your_idx = remaining_player_ids.index(self.player_id)
        if your_idx == 0:
            left_idx = -1
            right_idx = 1
        elif your_idx == len(remaining_player_ids) - 1:
            right_idx = 0
            left_idx = your_idx - 1
        else:
            left_idx = your_idx - 1
            right_idx = your_idx + 1
        
        left_player = remaining_player_ids[left_idx]
        right_player = remaining_player_ids[right_idx]
        action_set = [f'player_{left_player}', f'player_{right_player}']
        statement_order = STATEMENT_ORDER_HUMAN_TEMPLATE.format(round, self.player_id, self.role, action_set)
        context = self.organize_context(remaining_player)

        return context + '\n' + statement_order


    def day_discussion(self, round: int, remaining_player) -> str:
        statement = STATEMENT_HUMAN_TEMPLATE.format(round, self.player_id, self.role)
        context = self.organize_context(remaining_player)

        return context + '\n' + statement


    def vote(self, round: int, remaining_player) -> str:
        action = ['vote to eliminate ' + i for i in remaining_player]
        action = ', '.join(action)
        if self.role == 'Werewolf':
            voting = WEREWOLF_VOTING_HUMAN_TMEPLATE.format(round, self.player_id, self.role, action)
        else:
            voting = VILLAGER_VOTING_HUMAN_TMEPLATE.format(round, self.player_id, self.role, action)

        context = self.organize_context(remaining_player)
        return context + '\n' + voting


    def election_vote(self, sheriff_candidate, remaining_player):
        action = ['vote for ' + i for i in sheriff_candidate]
        action = ', '.join(action)
        sheriff_candidate_str = ', '.join(sheriff_candidate)
        context = self.organize_context(remaining_player)
        voting = ELECTION_VOTE_HUMAN_TEMPLATE.format(sheriff_candidate_str, self.player_id, self.role, action)

        return context + '\n' + voting


    def pseudo_vote(self, round: int, remaining_player) -> str:
        action = ['vote to eliminate ' + i for i in remaining_player]
        action = ', '.join(action)
        if self.role == 'Werewolf':
            voting = WEREWOLF_VOTING_HUMAN_PSEUDO_TMEPLATE.format(round, self.player_id, self.role, action)
        else:
            voting = VILLAGER_VOTING_HUMAN_PSEUDO_TMEPLATE.format(round, self.player_id, self.role, action)

        context = self.organize_context(remaining_player)
        return context + '\n' + voting


    def deductive_reasoning(self, remaining_player, time_str):
        context = self.organize_context(remaining_player)
        reasoning = REASONING_HUMAN_TEMPLATE.format(time_str, self.player_id, self.role)
        if len(self.reliability) > 0:
            current_reliability = f'''\n Your current reasoning results about other players are as follows. \n {self.reliability} '''
        else:
            current_reliability = ''

        return context + reasoning + current_reliability


    def parse_night_action(self, response: str, round: int, remaining_player):
        available_player_id = [int(i.split('_')[1]) for i in remaining_player]
        try:
            json_strings = extract_json_strings(response)
            action_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.error('Invalid output of taking night action.')
            self.logger.info(response)
            self.logger.exception('An error occurs: {}'.format(e))

            return False
        
        try:
            if action_result['action'].split('_')[1].isdigit():
                selected_player_id = int(action_result['action'].split('_')[1])
                if selected_player_id in available_player_id:
                    if self.role == 'Guard':
                        self.fact.append(f'In night {round} round, you chose to protect player_{selected_player_id}.')
                        self.action_history.append({f'night {round}': f'You chose to proctect player_{selected_player_id}.'})
                        return selected_player_id
                    elif self.role == 'Seer':
                        self.action_history.append({f'night {round}': f'You chose to see player_{selected_player_id}.'})
                        return selected_player_id
                    elif self.role == 'Werewolf':
                        self.fact.append(f'In night {round} round, you chose to kill player_{selected_player_id}.')
                        self.action_history.append({f'night {round}': f'You chose to kill player_{selected_player_id}.'})
                        return selected_player_id
                    
            else:
                self.logger.error("Cannot parse the action")
                self.logger.info(action_result)
                return False

        except Exception as e:
            self.logger.error("Cannot parse the action")
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.info(action_result)
            return False


    def parse_statement_order(self, response: str, round: int, remaining_player):
        remaining_player_ids = [int(i.split('_')[1]) for i in remaining_player]
        your_idx = remaining_player_ids.index(self.player_id)
        if your_idx == 0:
            left_idx = -1
            right_idx = 1
        elif your_idx == len(remaining_player_ids) - 1:
            right_idx = 0
            left_idx = your_idx - 1
        else:
            left_idx = your_idx - 1
            right_idx = your_idx + 1
        
        left_player = remaining_player_ids[left_idx]
        right_player = remaining_player_ids[right_idx]
        try:
            json_strings = extract_json_strings(response)
            order_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.error('Invalid output of determining statement order')
            self.logger.info(response)

            self.fact.append(f'You chose player_{left_player} to make a statement first in day {round}')
            
            return self.take_order(remaining_player_ids, left_player, left = True), False, left_player
        
        try:
            if order_result['action'].split('_')[1].isdigit():
                selected_player = int(order_result['action'].split('_')[1])
                if selected_player == left_player:
                    self.fact.append(f'You chose player_{left_player} to make a statement first in day {round}')
                    self.action_history.append({f'day {round}': f'You chose player_{left_player} to make a statement first'})
                    return self.take_order(remaining_player_ids, left_player, left = True), True, left_player
                elif selected_player == right_player:
                    self.fact.append(f'You chose player_{right_player} to make a statement first in day {round}')
                    self.action_history.append({f'day {round}': f'You chose player_{right_player} to make a statement first'})
                    return self.take_order(remaining_player_ids, right_player, left = False), True, right_player
                else:
                    self.logger.error('Invalid decision on the order of statement.')
                    self.logger.info(order_result)

                    self.fact.append(f'You chose player_{left_player} to make a statement first in day {round}')
                    self.action_history.append({f'day {round}': f'You chose player_{left_player} to make a statement first'})
                    return self.take_order(remaining_player_ids, left_player, left = True), False, left_player
            else:
                self.logger.error('Invalid decision on the order of statement.')
                self.logger.info(order_result)
                self.fact.append(f'You chose player_{left_player} to make a statement first in day {round}')
                self.action_history.append({f'day {round}': f'You chose player_{left_player} to make a statement first'})
                return self.take_order(remaining_player_ids, left_player, left = True), False, left_player
        except Exception as e:
            self.logger.error('Invalid decision on the order of statement.')
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.info(order_result)
            self.fact.append(f'You chose player_{left_player} to make a statement first in day {round}')
            self.action_history.append({f'day {round}': f'You chose player_{left_player} to make a statement first'})
            return self.take_order(remaining_player_ids, left_player, left = True), False, left_player


    
    def parse_statement(self, response: str, round: int):
        try:
            json_strings = extract_json_strings(response)
            statement_result = json.loads(json_strings[0])
            words = statement_result['statement']
            self.fact.append(f'In day {round}, you say: {words}')
            self.statement_history.append({f'day {round}': words})

            return words
        except Exception as e:
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.info(response)

            return False


    def parse_pseudo_vote(self, response: str, remaining_player):
        available_player_id = [int(i.split('_')[1]) for i in remaining_player]
        try:
            json_strings = extract_json_strings(response)
            voting_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.error('Invalid output of voting')
            self.logger.info(response)

            return False

        try:
            if voting_result['action'] == 'do not vote':
                return -1
            elif voting_result['action'].split('_')[1].isdigit():
                selected_player = int(voting_result['action'].split('_')[1])
                if selected_player in available_player_id:
                    return selected_player
                else:
                    self.logger.error('Invalid voting actions')
                    self.logger.info(voting_result)

                    return False
            else:
                return False
        except Exception as e:
            self.logger.error('Invalid voting actions or do not vote')
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.info(voting_result)
            
            return False


    def parse_voting(self, response: str, round: int, remaining_player):
        available_player_id = [int(i.split('_')[1]) for i in remaining_player]
        try:
            json_strings = extract_json_strings(response)
            voting_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.error('Invalid output of voting')
            self.logger.info(response)

            return False

        try:
            if voting_result['action'] == 'do not vote':
                self.fact.append(f'You did not vote in day {round}')
                self.action_history.append({f'day {round}': 'You did not vote'})
                return -1
            elif voting_result['action'].split('_')[1].isdigit():
                selected_player = int(voting_result['action'].split('_')[1])
                if selected_player in available_player_id:
                    self.fact.append(f'You voted to eliminate player_{selected_player} in day {round}')
                    self.action_history.append({f'day {round}': f'You voted to eliminate player_{selected_player}'})
                    return selected_player
                else:
                    self.logger.error('Invalid voting actions')
                    self.logger.info(voting_result)

                    return False
            else:
                return False
        except Exception as e:
            self.logger.error('Invalid voting actions or do not vote')
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.info(voting_result)
            
            return False


    def parse_election_voting(self, response: str, sheriff_candidate):
        available_player_id = [int(i.split('_')[1]) for i in sheriff_candidate]
        try:
            json_strings = extract_json_strings(response)
            voting_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.error('Invalid output of voting')
            self.logger.info(response)

            return False

        try:
            if voting_result['action'] == 'do not vote':
                self.fact.append('You did not vote during the election phase of day 1')
                self.action_history.append({'election phase': 'You did not vote'})
                return -1
            elif voting_result['action'].split('_')[1].isdigit():
                selected_player = int(voting_result['action'].split('_')[1])
                if selected_player in available_player_id:
                    self.fact.append(f'You voted for player_{selected_player} during the election phase of day 1')
                    self.action_history.append({'election phase': f'You voted for player_{selected_player}'})
                    return selected_player
                else:
                    self.logger.error('Invalid voting actions')
                    self.logger.info(voting_result)

                    return False
            else:
                return False
        except Exception as e:
            self.logger.error('Invalid voting actions or do not vote')
            self.logger.exception("An error occurred: {}".format(e))
            self.logger.info(voting_result)
            
            return False


    def parse_reasoning(self, response: str):
        try:
            json_strings = extract_json_strings(response)
            reasoning_result = json.loads(json_strings[0])
        except Exception as e:
            self.logger.exception('An error occurs: {}'.format(e))
            self.logger.error('Invalid output of reasoning')
            self.logger.info(response)

            return False
        

        for player in reasoning_result:
            if player not in self.reliability:
                self.reliability[player] = {}
            try:
                player_role = reasoning_result[player]['role']
                self.reliability[player]['role'] =  player_role
                if player_role == 'Werewolf' and self.role != 'Werewolf':
                    self.reliability[player]['reliability'] = 11 - int(reasoning_result[player]['confidence'])
                else:
                    self.reliability[player]['reliability'] = int(reasoning_result[player]['confidence'])
                self.reliability[player]['confidence'] = int(reasoning_result[player]['confidence'])
                    

            except Exception as e:
                self.logger.error(f'Failed to update reliability of player_{player}')
                self.logger.exception("An error occurred: {}".format(e))
                self.logger.info(reasoning_result)

                return False
            
        self.add_new_context(True)       
        return True

        
    def organize_context(self, remaining_player):
        str_remaining_palyer = ', '.join(remaining_player)
        a = f'All information you can leverage is listed below. \n Remaining Players: {str_remaining_palyer}. \n \n'
        b = 'The following information is true.\n'
        for i in range(1, len(self.fact) + 1):
            temp = f'[{i}] ' + self.fact[i-1] + '\n'
            b = b + temp 
        
        c = '\n Your reliability scores of other players are similar to your confidence rating when reasoning. Note that for the player you think is a werewolf, the reliability is 11 minus your confidence score, and the reliability of other player is your confidence score. The statements made by players with a reliability score higher than 6 are viewed as potentially true, and the rest are viewed as potentially false.\n \n'
        c += 'The following information might be true.\n'
        if len(self.potential_truth) == 0:
            c += 'None \n'
        i = len(self.fact)
        for j in range(1, len(self.potential_truth) + 1):
            temp = f'[{i+j}]' + list(self.potential_truth[j-1].values())[0] + '\n'
            c = c + temp

        d = '\n The following information might be false.\n'
        j = len(self.potential_truth)
        if len(self.potential_deception) == 0:
            d += 'None \n'
        for k in range(1, len(self.potential_deception) + 1):
            temp = f'[{i+j+k}]' + list(self.potential_deception[k-1].values())[0] + '\n'
            d = d + temp

        e = '\n The following public information still needs further clarification. \n'
        if len(self.public_info) == 0:
            e += 'None \n'
        for k in range(1, len(self.public_info) + 1):
            temp =  list(self.public_info[k-1].values())[0] + '\n \n'
            e = e + temp        

        return a + b + c + d + e

    
    def add_new_context(self, clear) -> None:
        '''
        Add public info into the context according to the reliability
        '''
        for info in self.public_info:
            player = list(info.keys())[0]
            if player not in self.reliability:
                self.potential_deception.append({player: info[player]})
                for j in self.potential_truth:
                    if player in j:
                        self.potential_deception.append(j)
                        self.potential_truth.remove(j)
                continue

            if self.reliability[player]['reliability'] >= 6:
                self.potential_truth.append({player: info[player]})
                for j in self.potential_deception:
                    if player in j:
                        self.potential_truth.append(j)
                        self.potential_deception.remove(j)
            else:
                self.potential_deception.append({player: info[player]})
                for j in self.potential_truth:
                    if player in j:
                        self.potential_deception.append(j)
                        self.potential_truth.remove(j)
        if clear:
            self.public_info_history.extend(self.public_info)
            self.public_info = []


    def save_log(self, experiment_start_time = None):
        player_log = {'model_path': "Human Player",
                      'model_name': "Human Player",
                      'player_id': self.player_id,
                      'role': self.role,
                      'fact': self.fact,
                      'potential_truth': self.potential_truth,
                      'potential_deception': self.potential_deception,
                      'reliability': self.reliability,
                      'action_history': self.action_history, 
                      'statement_history': self.statement_history,
                      'reasoning_history': self.reasoning_history,
                      'public_info_history': self.public_info_history}
        
        if experiment_start_time is not None:
            dir_path = os.path.join(self.log_path, experiment_start_time)
        else:
            dir_path = self.log_path
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        
        with open(os.path.join(dir_path, f'{self.player_id}_{self.role}.json'), "w", encoding = 'utf8') as fp:
            json.dump(player_log, fp, indent = 4)
