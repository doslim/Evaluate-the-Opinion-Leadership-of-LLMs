import openai
import requests
import random
import re
import time
import datetime
import math
import json
from tqdm import tqdm

url = 'https://api.openai.com/v1/chat/completions'

api_key = 'xxxxxxxxxxxxx'

model = "gpt-3.5-turbo-16k"


def generate_question(game_rules, rule_based=True, rounds=10, burn_in=3, gen_size=10):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    questions_list = []

    if rule_based == True:
        question_type = "rule-based"
        requirement = 'Questions should be centered around rule details rather than scenario-based decisions.'
        examples = [
            'What abilities does the Guard have?',
            'How many people can the Werewolves kill in one night?',
            'Do villagers have any special abilities?',
            'If the sheriff dies, how should the new Sheriff be determined?',
            'Can players abstain from voting during the voting phase?'
        ]
    else:
        question_type = "situation-based"
        requirement = 'You need to construct a simple scenario within my guidelines and propose deductive questions. The questions should be centered around situation-based decisions rather than rule details.'
        examples = [
            'If one player was killed by the Werewolves was protected by the Guard on the same night, then how many players will die that night?',
            'Which player should be eliminated if, in a single vote, player_1 receives 2 votes, player_3 receives 3 votes, and player_6 receives 1 vote?',
            'If on the first night, the werewolves killed Player_2, the Guardian protected Player_2, and the Seer verified Player_2\'s identity, how many deaths would occur on the first day?',
            'If, after the voting phase on a certain day, there are only 4 players left - 2 werewolves and 2 villagers, can the game be declared over?',
            'If, after the discussion phase on a certain day, players 1, 2, 3, 4, 5, and 7 all consider player_6 suspicious, but players 3, 4, and 5 change their minds after hearing the Sheriff\'s statement, while players 1, 2, and 7 remain unchanged, does it necessarily mean that player_6 will be voted out in this round?'
        ]

    for i in tqdm(range(0, rounds)):
        if i < burn_in:
            example_q1, example_q2, example_q3, example_q4, example_q5 = examples
        else:
            example_q1, example_q2 = random.sample(examples, 2)
            example_q3, example_q4, example_q5 = random.sample(questions_list, 3)

        prompt1 = f'''
        # 01 You are a Q&A pair dataset processing expert.

        # 02 Your task is to generate questions that are suitable as a quiz pair dataset based on the rules I gave about the Werewolf game. The questions should be about the rules of the game of werewolf.

        # 03 The questions should be as short as possible, not too long.

        # 04 There should be only one question in a sentence.

        # 05 {requirement}

        # 06 Examples of generated questions:

        """
        {example_q1}

        {example_q2}

        {example_q3}

        {example_q4}

        {example_q5}

        """

        # 07 Here are the game rules I gave:

        """

        {{GAMERULES}}

        """

        # 08 WARNING: You mustn't codify any characters or rules! The characters and rules in your answer must come from the instructions I give you.

        # 09 You can only answer me in English.
        '''

        prompt = prompt1.replace("{{GAMERULES}}", game_rules)

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user",
                 "content": f"Now generate {gen_size} {question_type} questions about our Werewolf game."}
            ]

        }

        response = requests.post(url, headers=headers, json=data, verify=False)

        if response.status_code == 200:
            question_text = response.json()["choices"][0]["message"]['content']
        else:
            print(f"Error: {response.status_code}")
            # print(response.content)
            continue

        clean_question = [re.sub(r'^\d+\.\s*', '', item) for item in
                          question_text.replace('" ', '').replace('"', '').split('\n') if
                          item and not any(ch > '\u4e00' and ch < '\u9fff' for ch in item) and '?' in item]
        if len(clean_question) == gen_size:
            questions_list.extend(clean_question)
        else:
            continue

        if i == rounds - 1:
            questions_list = list(set(questions_list))
            print(len(questions_list))
            if len(questions_list) < gen_size * rounds:
                i == i - 1
                continue
        else:
            continue

    return questions_list


def generate_answers(game_rules, questions_list, batch_size=10):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    answers_list = []
    num_batches = math.ceil(len(questions_list) / batch_size)

    for i in tqdm(range(0, num_batches)):

        prompt2 = f'''
        # 01 You're an expert at answering questions about Werewolf game.

        # 02 Your task is to answer my questions about a specific Werewolf game based on the instructions I give you.

        # 03 You mustn't codify any roles or rules!

        # 04 The answers should be simple, direct but make full use of my information, and be more informative.

        # 05 You must delete "based on the rules ...", "given the rules of ...", etc. from your answers.

        # 06 You must provide your answer directly without introductory phrases such as "Here are your answers.", "# Answers to your questions:", etc.

        # 07 Here are the game rules I gave:

        """

        {{GAMERULES}}

        """

        # 08 WARNING: You mustn't codify any characters or rules! The characters and rules in your answer must come from the instructions I give you.

        # 09 You can only answer me in English.
        '''

        batch = questions_list[i * batch_size:(i + 1) * batch_size]
        batch_str = '\n'.join(batch)
        current_questions = f'''
        # Here are my questions.

        """
        {batch_str}
        """
        '''

        prompt = prompt2.replace("{{GAMERULES}}", game_rules)

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": current_questions}
            ]

        }

        response = requests.post(url, headers=headers, json=data, verify=False)

        if response.status_code == 200:
            answers_text = response.json()["choices"][0]["message"]['content']
        else:
            print(f"Error: {response.status_code}")
            # print(response.content)
            continue

        clean_answer = [re.sub(r'^\s*\d+\.\s*', '', item) for item in
                        answers_text.replace('" ', '').replace('"', '').replace('#', '').split('\n') if
                        item.strip() != ""]
        if len(clean_answer) == batch_size:
            answers_list.extend(clean_answer)
        else:
            print(f"Error: Missing answers for question {i * batch_size}:{(i + 1) * batch_size}")
            answers_list.extend(["Error"] * batch_size)
            continue

    return answers_list


def generate_binary(game_rules, rounds=10, burn_in=3, batch_size=10):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    qa_list = []

    examples = [
        '{"question": "Do villagers have any special abilities?","answer": “No"}',
        '{"question": "Is the role assignment of Werewolves and Villagers random?","answer": "Yes"}',
        '{"question": "Can the Werewolves be killed by another Werewolf?","answer": "Yes"}',
        '{"question": "Can the Werewolves kill two players in one night?","answer": "No"}',
        '{"question": "Can the Werewolves pretend to be Villagers during the day round?","answer": "Yes"}'
    ]

    for i in tqdm(range(0, rounds)):
        if i < burn_in:
            example_q1, example_q2, example_q3, example_q4, example_q5 = examples
        else:
            example_q1, example_q2 = random.sample(examples, 2)
            example_q3, example_q4, example_q5 = random.sample(questions_list, 3)

        prompt3 = f'''
        #01 You are a Q&A pair dataset processing expert.

        #02 Your task is to generate {batch_size} Yes-or-No questions and their corresponding answers that are suitable as a quiz pair dataset based on the rules I gave about the Werewolf game. 

        #03 The questions should be as short as possible, not too long.

        #04 There should be only one question in a sentence.

        #05 You should only respond in JSON format as described below.
        Response Format:
        '{{"question": your question, "answer": choose from ['Yes', 'No']}};'
        Please separate different replies with ";". Ensure the response can be parsed by Python json.loads.

        #06 Examples of generated Q&A pairs:

        """
        {example_q1}

        {example_q2}

        {example_q3}

        {example_q4}

        {example_q5}

        """

        # 07 Here are the game rules I gave:

        """

        {{GAMERULES}}

        """

        # 08 WARNING: You mustn't codify any characters or rules! The characters and rules in your answer must come from the instructions I give you.

        # 09 You can only answer me in English.
        '''

        prompt = prompt3.replace("{{GAMERULES}}", game_rules)

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user",
                 "content": f"Now generate {batch_size} Q&A pairs."}
            ]

        }

        response = requests.post(url, headers=headers, json=data, verify=False)

        if response.status_code == 200:
            qa_pairs = response.json()["choices"][0]["message"]['content']
            qa_pairs_list = [t.replace("\n", "") for t in qa_pairs.split(";\n")]
            if len(qa_pairs_list) == batch_size:
                qa_list.extend(qa_pairs_list)
        else:
            print(f"Error: {response.status_code}")
            # print(response.content)
            continue

    return qa_list


def read_file(file_name):
    try:
        with open(file_name, "r") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")


def write_to_file(content):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"new_file_{timestamp}.txt"
    with open(file_name, 'w', newline='') as file:
        file.write('\n'.join(content))
    print("File has been created and written.")


def write_to_json(content):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"binary_{timestamp}.json"
    parsed_data = [json.loads(item.replace(';', '')) for item in list(set(content))]
    with open(file_name, "w") as json_file:
        json.dump(parsed_data, json_file, indent=4)


def qa_pairing(questions_list, answers_list):
    pairs = []
    if len(questions_list) == len(answers_list):
        for i in range(len(questions_list)):
            item = {"question": questions_list[i], "answer": answers_list[i]}
            data.append(item)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"pairs_{timestamp}.txt"
        with open(file_name, "w") as file:
            json.dump(data, file, indent=4)
    else:
        print("Error: Mismatch in length!")


def main():
    game_rules = read_file("input_file.txt")
    # my_questions = generate_question(game_rules = game_rules, rule_based=True, rounds=100, burn_in=3, gen_size=10)
    # write_to_file(my_questions)

    # my_questions = read_file("test_questions.txt").split('\n')
    # my_answers = generate_answers(game_rules = game_rules, questions_list = my_questions, batch_size = 10)
    # write_to_file(my_answers)

    # my_questions = read_file("test_questions.txt").split('\n')
    # my_answers = read_file("test_answers.txt").split('\n')
    # qa_pairing(my_questions, my_answers)

    my_binary = generate_binary(game_rules=game_rules, rounds=3, burn_in=1, batch_size=10)
    write_to_json(my_binary)


if __name__ == "__main__":
    main()