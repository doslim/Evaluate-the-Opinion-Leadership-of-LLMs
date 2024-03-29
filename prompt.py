GAME_RULE_TEMPLATE = '''
You are playing a game called the Werewolf with some other players. This game is based on text conversations. There are seven players including player_1, player_2, player_3, player_4, player_5, player_6, and player_7. Here are the game rules: 

Roles: The moderator is also host, he organised this game and you need to answer his instructions correctly. Don't talk with the moderator. There are seven roles in the game, including two Werewolves, three Villagers, one Seer, and one Guard. At the beginning of the game, each player is assigned a hidden role which divides them into the Werewolves and the Villagers (Seer, Guard, Villagers). Then the game alternates between the night round and the day round until one side wins the game.

In the night round: the Werewolves recognize each other and choose one player to kill; the Seer chooses one player to see if they are a Werewolf; the Guard chooses one player including themselves to protect without knowing who is chosen by the Werewolves; the Villagers do nothing.

In the day round: three phases including an announcement phase, a discussion phase, and a voting phase are performed in order.
In the announcement phase, an announcement of last night's result is made to all players. If player_i was killed and not protected last night, the announcement will be "player_i was killed"; if a player was killed and protected last night, the announcement will be "no player was killed"
Only the first day will have an election phase. In the election phase, you can decide whether or not to run for Sheriff based on your identity. Players running for Sheriff take turns stating their reasons. Then each player votes for one Sheriff or chooses not to vote. The Sheriff is a special role that does not conflict with the players' roles. The elected Sheriff has the authority to choose which player starts the next round of discussions. And, as the last speaker, he can provide a summary to persuade other players to vote in alignment with him. 
In the discussion phase, each remaining player speaks only once in order determined by the Sheriff to discuss who might be the Werewolves.
In the voting phase, each player votes for one player or chooses not to vote. The player with the most votes is eliminated and the game continues to the next night round.
The Werewolves win the game if the number of remaining Werewolves is equal to the number of remaining Seer, Guard, and Villagers. The Seer, Guard, and Villagers win the game if all Werewolves are eliminated.
You are player_{}, the {}. You're playing with 6 other players.
'''


GAME_RULE = '''
You are playing a game called the Werewolf with some other players. This game is based on text conversations. There are seven players including player_1, player_2, player_3, player_4, player_5, player_6, and player_7. Here are the game rules: 

Roles: The moderator is also host, he organised this game and you need to answer his instructions correctly. Don't talk with the moderator. There are seven roles in the game, including two Werewolves, three Villagers, one Seer, and one Guard. At the beginning of the game, each player is assigned a hidden role which divides them into the Werewolves and the Villagers (Seer, Guard, Villagers). Then the game alternates between the night round and the day round until one side wins the game.

In the night round: the Werewolves recognize each other and choose one player to kill; the Seer chooses one player to see if they are a Werewolf; the Guard chooses one player including themselves to protect without knowing who is chosen by the Werewolves; the Villagers do nothing.

In the day round: three phases including an announcement phase, a discussion phase, and a voting phase are performed in order.
In the announcement phase, an announcement of last night's result is made to all players. If player_i was killed and not protected last night, the announcement will be "player_i was killed"; if a player was killed and protected last night, the announcement will be "no player was killed"
Only the first day will have an election phase. In the election phase, you can decide whether or not to run for Sheriff based on your identity. Players running for Sheriff take turns stating their reasons. Then each player votes for one Sheriff or chooses not to vote. The Sheriff is a special role that does not conflict with the players' roles. The elected Sheriff has the authority to choose which player starts the next round of discussions. And, as the last speaker, he can provide a summary to persuade other players to vote in alignment with him. 
In the discussion phase, each remaining player speaks only once in order determined by the Sheriff to discuss who might be the Werewolves.
In the voting phase, each player votes for one player or chooses not to vote. The player with the most votes is eliminated and the game continues to the next night round.
The Werewolves win the game if the number of remaining Werewolves is equal to the number of remaining Seer, Guard, and Villagers. The Seer, Guard, and Villagers win the game if all Werewolves are eliminated.
'''

WEREWOLF_ACTION_TEMPLATE = '''
Now it is night {} round. As player_{} and a {}, you should choose one player to kill. You should first reason about the current situation, then choose one from the following actions: {}. If your choice does not match that of your teammate, no player will be killed.
You should only respond in JSON format as described below.
Response Format:
{{
    "reasoning": "reason about the current situation",
    "action": "choose one from {}"
}}
Ensure the response is in English and can be parsed by Python json.loads.
'''


SEER_ACTION_TEMPLATE = '''
Now it is night {} round. As player_{} and a {}, you should choose one player to see. You should first reason about the current situation, then choose one from the following actions: {}.
You should only respond in JSON format as described below.
Response Format:
{{
    "reasoning": "reason about the current situation",
    "action": "choose one from {}"
}}
Ensure the response is in English and can be parsed by Python json.loads.
'''


GUARD_ACTION_TEMPLATE = '''
Now it is night {} round. As player_{} and a {}, you should choose one player to protect. You should first reason about the current situation, then choose one from the following actions: {}.
You should only respond in JSON format as described below.
Response Format:
{{
    "reasoning": "reason about the current situation",
    "action": "choose one from {}"
}}
Ensure the response is in English and can be parsed by Python json.loads.
'''



SHERRIF_STATEMENT_TEMPLATE = '''
Now it is day {} discussion phase and it is your turn to speak. As player_{} and a {}, before speaking to the other players, you should first reason the current situation only to yourself, and then speak to all other players. As the Sheriff, you can summarize the discussion and provide advice for voting. Please be cautious about revealing your role in this phase.
You should only respond in JSON format as described below.
Response Format:
{{
    "reasoning": "reason about the current situation only to yourself",
    "statement": "your statement that will be public to all platers"
}}
Ensure the response is in English and can be parsed by Python json.loads. Please be cautious about revealing your role in this phase.
'''


STATEMENT_TEMPLATE = '''
Now it is day {} discussion phase and it is your turn to speak. As player_{} and a {}, before speaking to the other players, you should first reason the current situation only to yourself, and then speak to all other players.
You should only respond in JSON format as described below.
Response Format:
{{
    "reasoning": "reason about the current situation only to yourself",
    "statement": "your statement that will be public to all platers"
}}
Ensure the response is in English and can be parsed by Python json.loads. Please be cautious about revealing your role in this phase.
'''



WEREWOLF_VOTING_TMEPLATE = '''
Now it is day {} voting phase, you should vote to eliminate one player or do not vote to maximize the Werewolves' benefit. As player_{} and a {}, you should first reason about the current situation, and then choose from the following actions: do no vote, {}.
You should only respond in JSON format as described below.
Response Format:
{{
    "reasoning": "reason about the current situation",
    "action": "vote to eliminate player_i"
}}
Ensure the response is in English and can be parsed by Python json.loads.
'''


VILLAGER_VOTING_TMEPLATE = '''
Now it is day {} voting phase, you should vote to eliminate one player that is most likely to be a werewolf or do not vote. As player_{} and a {}, you should first reason about the current situation, and then choose from the following actions: do no vote, {}.
You should only respond in JSON format as described below.
Response Format:
{{
    "reasoning": "reason about the current situation",
    "action": "vote to eliminate player_i"
}}
Ensure the response is in English and can be parsed by Python json.loads.
'''
# Ensure the response can be parsed by Python json.loads.
# '''


REASONING_TEMPLATE = '''
Now it is {}, As player_{} and a {}, you should reflect on your previous deduction and reconsider the hidden roles of {}. You should provide your reasoning, rate your confidence, and cite all key information as evidence to support your deduction.

You should only respond in JSON format as described below.
Response Format:
{{
    "{}": {{
        "role": select the most likely hidden role of this player from ["Werewolf", "Seer", "Doctor", "Villager", "Uncertain"],
        "reasoning": your reflection and reasoning,
        "confidence": use an integer from 5 (pure guess) to 10 (absolutely sure) to rate the confidence of your deduction,
        "evidence": list of integers that cite the key information
    }}
}}
'''


STATEMENT_ORDER_TEMPLATE = '''
Now it is day {} discussion phase and you are the Sheriff . As player_{}, a {} and the Sheriff, you should first reason the current situation only to yourself, and then decide on the first player to speak. 
You should only respond in JSON format as described below.
Response Format:
{{
    "reasoning": "reason about the current situation only to yourself",
    "action": "choose one from {}"
}}
'''


ELECTION_STATEMENT_TEMPLATE = '''
Now it is day 1 election phase, you run for the Sheriff, who can decide the order of statements, summarize the discussion, and provide advice for voting. As player_{} and a {}, you should first reason about the current situation, and then explain why you are qualified to be the Sheriff. 
You should only respond in JSON format as described below.
Response Format:
{{
    "reasoning": "reason about the current situation only to yourself",
    "statement": "your statement that will be public to all platers"
}}
Ensure the response is in English and can be parsed by Python json.loads. Do NOT reveal your role in this phase.
'''

ELECTION_VOTE_TEMPLATE = '''Now it is day 1 election phase, {} are running for the Sheriff,  who can decide the order of statements, summarize the discussion, and provide advice for voting. As player_{} and a {}, you should first reason about the current situation, and then vote for a player to be the Sheriff or choose not to vote to maximize the interests of your team. You should choose from the following actions: do no vote, {}.
Response Format:
{{
    "reasoning": "reason about the current situation",
    "action": "vote for player_i"
}}
Ensure the response is in English and can be parsed by Python json.loads.
'''







# Instruction template for the human player
SEER_ACTION_HUMAN_TEMPLATE = '''
Now it is night {} round. As player_{} and a {}, you should choose one player to see. Please respond in JSON format as described below.
Response Format:

{{
    "action": "choose one from {}"
}}

Ensure the response is in English and can be parsed by Python json.loads.
'''


WEREWOLF_ACTION_HUMAN_TEMPLATE = '''
Now it is night {} round. As player_{} and a {}, you should choose one player to kill. Please respond in JSON format as described below.
Response Format:

{{
    "action": "choose one from {}"
}}

Ensure the response is in English and can be parsed by Python json.loads.
'''


GUARD_ACTION_HUMAN_TEMPLATE = '''
Now it is night {} round. As player_{} and a {}, you should choose one player to protect. Please respond in JSON format as described below.
Response Format:
{{
    "action": "choose one from {}"

}}

Ensure the response is in English and can be parsed by Python json.loads.
'''

STATEMENT_HUMAN_TEMPLATE = '''
Now it is day {} discussion phase and it is your turn to speak. You are player_{} and a {}. Please respond in JSON format as described below.
Response Format:

{{
    "statement": "your statement that will be public to all platers"
}}

Ensure the response is in English and can be parsed by Python json.loads.
'''

WEREWOLF_VOTING_HUMAN_TMEPLATE = '''
Now it is day {} voting phase, as player_{} and a {}, you should vote to eliminate one player or do not vote to maximize the Werewolves' benefit. You choose from the following actions: do no vote, {}. Please respond in JSON format as described below.
Response Format:

{{
    "action": "vote to eliminate player_i or do not vote"
}}
Ensure the response is in English and can be parsed by Python json.loads.
'''


VILLAGER_VOTING_HUMAN_TMEPLATE = '''
Now it is day {} voting phase,  as player_{} and a {}, you should vote to eliminate one player that is most likely to be a werewolf or do not vote. You should  choose from the following actions: do no vote, {}. Please respond in JSON format as described below.
Response Format:

{{
    "action": "vote to eliminate player_i or do not vote"
}}

Ensure the response is in English and can be parsed by Python json.loads.
'''

WEREWOLF_VOTING_HUMAN_PSEUDO_TMEPLATE = '''
Before the Sheriff speaks, we need to collect your votes. Note that this vote will NOT be counted and will only be used for analysis and will NOT affect the progress of the game. Now it is day {} voting phase, as player_{} and a {}, you should vote to eliminate one player or do not vote to maximize the Werewolves' benefit. You choose from the following actions: do no vote, {}. Please respond in JSON format as described below.
Response Format:

{{
    "action": "vote to eliminate player_i or do not vote"
}}
Ensure the response is in English and can be parsed by Python json.loads.
'''


VILLAGER_VOTING_HUMAN_PSEUDO_TMEPLATE = '''
Before the Sheriff speaks, we need to collect your votes. Note that this vote will NOT be counted and will only be used for analysis and will NOT affect the progress of the game. Now it is day {} voting phase,  as player_{} and a {}, you should vote to eliminate one player that is most likely to be a werewolf or do not vote. You should  choose from the following actions: do no vote, {}. Please respond in JSON format as described below.
Response Format:

{{
    "action": "vote to eliminate player_i or do not vote"
}}

Ensure the response is in English and can be parsed by Python json.loads.
'''


REASONING_HUMAN_TEMPLATE = '''
Now it is {}, As player_{} and a {}, you can reflect on your previous deduction and reconsider the hidden roles of other players. You should provide your reasoning, and rate your confidence. Please respond in JSON format as described below.
Response Format:

{{
    "player_i": {{
        "role": select the most likely hidden role of this player from ["Werewolf", "Seer", "Doctor", "Villager", "Uncertain"],
        "confidence": use an integer to  from 5 (pure guess) to 10 (absolutely sure) rate the confidence of your deduction,
    }}
}}

Ensure the response is in English and can be parsed by Python json.loads. If you don't want to update the reliablity, please directly input 'None'. 
'''


ELECTION_VOTE_HUMAN_TEMPLATE = '''Now it is day 1 election phase, {} are running for the Sheriff,  who can decide the order of statements, summarize the discussion, and provide advice for voting. As player_{} and a {}, you should vote for a player to be the Sheriff or choose not to vote to maximize the interests of your team. You should choose from the following actions: do no vote, {}.  Please respond in JSON format as described below.
Response Format:

{{
    "action": "vote for player_i"
}}

Ensure the response is in English and can be parsed by Python json.loads.
'''

