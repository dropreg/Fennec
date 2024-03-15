import pandas as pd
import json
import pdb
import uuid
import random

def get_fennec_branch_system_message():
    message = "You are a fair, faithful, and helpful content evaluation assistant. Kindly assist me in accomplishing the assigned task by creating Evaluation Criteria for the provided dialogue."
    return message

def get_fennec_scoring_system_message():
    message = "You are a fair, faithful, and helpful content evaluation assistant. Kindly assist me finalize the assigned task by developing Evaluation Criteria into comprehensive Scoring Guidelines."
    return message

def get_fennec_pairwise_solving_system_message():
    message = "You are a fair, faithful, and helpful content evaluation assistant. Kindly assist me in finishing the assigned task by providing Pairwise Evaluations for the given dialogue. (Tips: This entails evaluating responses through the comparison of two distinct replies.)"
    return message

def get_fennec_correction_message():
    message = "You are an assistant capable of assisting in content modification. It is necessary to correct and refine the dialogue based on User Queries, Responses, and corresponding Evaluations results."
    return message

def get_fennec_branching_prompt():
    prompt = "For evaluating human satisfaction with responses from an AI assistant based on a [User Query], we need to brainstorm and establish five [Evaluation Criteria] directly linked to the user's query. These criteria play a crucial role in objectively assessing response content, with higher priority and greater evaluation weight.\n***\nAs an illustration:\n1. Relevance: Evaluate whether the response is directly related to the user's query.\n2. Criterion: Assess the correctness of the information provided in the response. etc.\n***\n[User Query]:\n{query}\n***\nPlease return five [Evaluation Criteria]:\n"
    return prompt

def get_fennec_scoring_prompt():
    prompt = "Consider a [User Query] and [Evaluation Criteria] for evaluating response satisfaction. Reflect on these criteria and offer a comprehensive [Scoring Guideline] on a scale of 1-5 (1 represents 'Not at all satisfactory' and 5 represents 'Extremely satisfactory'). Ensure that these guidelines are closely tied to both the user query and the assessment criteria, allowing for a precise evaluation of possible responses to the user query. Conduct a detailed comparison of the [Scoring Guideline] to ease adherence and assist individuals in assigning reasonable scores.\n***\n[User Query]:\n{query}\n***\n[Evaluation Criteria]:\n{criteria}\n***\nPlease return detailed [Scoring Guideline]:\n"
    return prompt

def get_fennec_pairwise_sovling_prompt():
    prompt = "Given a [User Query], please score the responses (A and B) from two AI assistants according to the [Evaluation Criteria] and [Scoring Guideline]. Ensure a comparative and objective assessment based on the evaluation criteria and scoring guideline, aiming to identify deficiencies in the response content. Provide a final score of 1-5 along with relevant explanations.\n***\n[User Query]:\n{query}\n***\n[Evaluation Criteria]:\n{criteria}\n***\n[Scoring Guideline]:\n{scoring}\n***\n[The Start of Response A]:\n{response1}\n[The End of Response A]\n***\n[The Start of Response B]:\n{response2}\n[The End of Response B]\n***\nPlease return [Judge Result] as follows:\nResponse A Score: 3\nExplanation: Explanation of the score for the Response A.\nResponse B Score: 3\nExplanation: Explanation of the score for the Response B.\nComparison: The comparison of the Response A and Response B.\n[Judge Result]:\n"
    return prompt

def get_fennec_reverse_pairwise_sovling_prompt():
    prompt = "Given a [User Query], please score the responses (A and B) from two AI assistants according to the [Evaluation Criteria] and [Scoring Guideline]. Ensure a comparative and objective assessment based on the evaluation criteria and scoring guideline, aiming to identify deficiencies in the response content. Provide a final score of 1-5 along with relevant explanations.\n***\n[User Query]:\n{query}\n***\n[Evaluation Criteria]:\n{criteria}\n***\n[Scoring Guideline]:\n{scoring}\n***\n[The Start of Response B]:\n{response2}\n[The End of Response B]\n***\n[The Start of Response A]:\n{response1}\n[The End of Response A]\n***\nPlease return [Judge Result] as follows:\nResponse A Score: 3\nExplanation: Explanation of the score for the Response A.\nResponse B Score: 3\nExplanation: Explanation of the score for the Response B.\nComparison: The comparison of the Response A and Response B.\n[Judge Result]:\n"
    return prompt

def get_fennec_correction_prompt():
    prompt = "Provided with a [User Query], the AI assistant's [Original Response], and a comprehensive objective evaluation of the response, please attend to the identified shortcomings in the original response according to the [Judge Result]. Make certain that the modified response remains objective, non-harmful, and constructive in addressing the user's query intent, while also aligning with human behavioral norms. \n***\n[User Query]:\n{query}\n***\n[The Start of Original Response]:\n{response}\n[The End of Original Response]\n***\n[The Start of Judge Result]:\n{judge}\n[The End of Judge Result]. Kindly return one final [Modified Response] for user query directly without additional information.\nPlease return [Modified Response]:\n"
    return prompt


def prepare_fennec():

    fennec_json_data = []
    branch_data_number = 0
    scoring_data_number = 0
    solving_data_number = 0
    correction_data_number = 0
    # df = pd.read_parquet("data/fennec_train_data/final/fennec_train_113.parquet")
    df = pd.read_parquet("/public/home/ljt/lxb/Fennec/data/fennec_train_data/final/train/fennec_train_correction.parquet")
    for _, json_item in df.iterrows():
        
        query = json_item['query']
        response_1 = json_item['response_1']
        response_2 = json_item['response_2']
        branch_list = json_item['branch']
        branch_list = [b.replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "").replace("5.", "") for b in branch_list]
        scoring_list = json_item['scoring']
        solving_list = json_item['solving']

        branch_system = get_fennec_branch_system_message()
        branch_prompt = get_fennec_branching_prompt().format(query=query)

        fennec_json_data.append({
            "prompt": query,
            "prompt_id": str(uuid.uuid4()),
            "messages": [
                {"content": branch_system, "role": "system"},
                {"content": branch_prompt,  "role": "user"},
                {"content": "\n".join(branch_list).strip(), "role": "assistant"}
            ]
        })
        branch_data_number += 1

        for idx, branch in enumerate(branch_list):
            
            scoring = scoring_list[idx].replace("[Scoring Guideline]:", "").strip()
            if scoring:

                scoring_system = get_fennec_scoring_system_message()
                scoring_prompt = get_fennec_scoring_prompt().format(query=query, criteria=branch)
                fennec_json_data.append({
                    "prompt": query,
                    "prompt_id": str(uuid.uuid4()),
                    "messages": [
                        {"content": scoring_system, "role": "system"},
                        {"content": scoring_prompt,  "role": "user"},
                        {"content": scoring, "role": "assistant"}
                    ]
                })
                scoring_data_number += 1

                solving = solving_list[idx].strip()
                if solving:
                    
                    pairwise_solving_system = get_fennec_pairwise_solving_system_message()
                    pairwise_sovling_prompt = get_fennec_pairwise_sovling_prompt().format(query=query, criteria=branch, scoring=scoring, response1=response_1, response2=response_2)
                    fennec_json_data.append({
                        "prompt": query,
                        "prompt_id": str(uuid.uuid4()), 
                        "messages": [
                            {"content": pairwise_solving_system, "role": "system"},
                            {"content": pairwise_sovling_prompt,  "role": "user"},
                            {"content": solving, "role": "assistant"}
                        ]
                    })

                    reverse_pairwise_sovling_prompt = get_fennec_reverse_pairwise_sovling_prompt().format(query=query, criteria=branch, scoring=scoring, response1=response_1, response2=response_2)
                    fennec_json_data.append({
                        "prompt": query,
                        "prompt_id": str(uuid.uuid4()),
                        "messages": [
                            {"content": pairwise_solving_system, "role": "system"},
                            {"content": reverse_pairwise_sovling_prompt,  "role": "user"},
                            {"content": solving, "role": "assistant"}
                        ]
                    })

                    solving_data_number += 2

        judge_a = []
        judge_b = []
        for branch, solving in zip(branch_list, solving_list):
            a_pos = solving.find("Response A Score:")
            b_pos = solving.find("Response B Score:")
            e_pos = solving.find("Comparison:")
            if a_pos < b_pos < e_pos:
                j_a = solving[a_pos:b_pos].replace("Response A", "Response")
                judge_a.append(branch)
                judge_a.append(j_a)
                j_b = solving[b_pos:e_pos].replace("Response B", "Response")
                judge_b.append(branch)
                judge_b.append(j_b)
        
        correction_system = get_fennec_correction_message()

        if json_item['correction_1'] and len(judge_a):

            correction_prompt = get_fennec_correction_prompt().format(query=query, response=response_1, judge ="".join(judge_a))
            fennec_json_data.append({
                "prompt": query,
                "prompt_id": str(uuid.uuid4()),
                "messages": [
                    {"content": correction_system, "role": "system"},
                    {"content": correction_prompt,  "role": "user"},
                    {"content": json_item['correction_1'], "role": "assistant"}
                ]
            })
            correction_data_number += 1

        if json_item['correction_2'] and len(judge_b):
            
            correction_prompt = get_fennec_correction_prompt().format(query=query, response=response_2, judge ="".join(judge_b))
            fennec_json_data.append({
                "prompt": query,
                "prompt_id": str(uuid.uuid4()),
                "messages": [
                    {"content": correction_system, "role": "system"},
                    {"content": correction_prompt,  "role": "user"},
                    {"content": json_item['correction_2'], "role": "assistant"}
                ]
            })
            correction_data_number += 1

    print("branch_data_number", branch_data_number)
    print("scoring_data_number", scoring_data_number)
    print("solving_data_number", solving_data_number)
    print("correction_data_number", correction_data_number)
    return fennec_json_data

def get_autoj_pairwise_solving_system_message():
    message = "You are a fair, faithful, and helpful content evaluation assistant. Kindly assist me in finishing the assigned task by providing Pairwise Evaluations for the given dialogue. (Tips: This entails evaluating responses through the comparison of two distinct replies.)"
    return message

def get_autoj_pairwise_solving_prompt():
    prompt = "Given a [User Query], please score the responses from two AI assistants according to the [Evaluation Criteria] and [Scoring Guideline]. Ensure a comparative and objective [Judge Result] based on the evaluation criteria and scoring guideline, aiming to identify deficiencies in the response content. Here are the instructions to assess and compare the two responses:\n\n1. Pinpoint the key factors to distinguish these two responses.\n2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with \"So, the final decision is Response 1 / Response 2 / Tie\". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided.\n***\n[User Query]:\n{query}\n***\n[Evaluation Criteria]:\n{criteria}\n***\n[Scoring Guideline]:\n{scoring}\n***\n[The Start of Response 1]:\n{response1}\n[The End of Response 1]\n***\n[The Start of Response 2]:\n{response2}\n[The End of Response 2]\n***\nPlease return [Judge Result]:\n"
    return prompt

def get_autoj_reverse_pairwise_solving_prompt():
    prompt = "Given a [User Query], please score the responses from two AI assistants according to the [Evaluation Criteria] and [Scoring Guideline]. Ensure a comparative and objective [Judge Result] based on the evaluation criteria and scoring guideline, aiming to identify deficiencies in the response content. Here are the instructions to assess and compare the two responses:\n\n1. Pinpoint the key factors to distinguish these two responses.\n2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with \"So, the final decision is Response 1 / Response 2 / Tie\". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided.\n***\n[User Query]:\n{query}\n***\n[Evaluation Criteria]:\n{criteria}\n***\n[Scoring Guideline]:\n{scoring}\n***\n[The Start of Response 2]:\n{response2}\n[The End of Response 2]\n***\n[The Start of Response 1]:\n{response1}\n[The End of Response 1]\n***\nPlease return [Judge Result]:\n"
    return prompt

def prepare_autoj():

    autoj_json_data = []
    
    for json_item in json.load(open("data/fennec_train_data/final/train/fennec_reverse_autoj_pairwise.jsonl")):

        query = json_item['query']
        response_1 = json_item['response_a']
        response_2 = json_item['response_b']
        branch = json_item['branch'][0].strip()
        scoring = json_item['scoring'][0].replace("[Scoring Guideline]:", "").strip()
        solving = json_item['solving'].strip()

        if solving:
            
            pairwise_solving_system = get_autoj_pairwise_solving_system_message()
            pairwise_solving_prompt = get_autoj_pairwise_solving_prompt().format(query=query, criteria=branch, scoring=scoring, response1=response_1, response2=response_2)
            autoj_json_data.append({
                "prompt": query,
                "prompt_id": str(uuid.uuid4()), 
                "messages": [
                    {"content": pairwise_solving_system, "role": "system"},
                    {"content": pairwise_solving_prompt,  "role": "user"},
                    {"content": solving, "role": "assistant"}
                ]
            })

            reverse_pairwise_solving_prompt = get_autoj_reverse_pairwise_solving_prompt().format(query=query, criteria=branch, scoring=scoring, response1=response_1, response2=response_2)
            autoj_json_data.append({
                "prompt": query,
                "prompt_id": str(uuid.uuid4()),
                "messages": [
                    {"content": pairwise_solving_system, "role": "system"},
                    {"content": reverse_pairwise_solving_prompt,  "role": "user"},
                    {"content": solving, "role": "assistant"}
                ]
            })
    return autoj_json_data


def get_judgelm_pairwise_solving_system_message():
    message = "You are a fair, faithful, and helpful content evaluation assistant. Kindly assist me in finishing the assigned task by providing Pairwise Evaluations for the given dialogue. (Tips: This entails evaluating responses through the comparison of two distinct replies.)"
    return message

def get_judgelm_pairwise_solving_prompt():
    prompt = "Given a [User Query], please score the responses from two AI assistants according to the [Evaluation Criteria] and [Scoring Guideline]. Ensure a comparative and objective assessment based on the evaluation criteria and scoring guideline, aiming to identify deficiencies in the response content. Provide a final score of 1-10 along with relevant explanations.\n***\n[User Query]:\n{query}\n***\n[Evaluation Criteria]:\n{criteria}\n***\n[Scoring Guideline]:\n{scoring}\n***\n[The Start of Assistant 1's Response]:\n{response1}\n[The End of Assistant 1's Response]\n***\n[The Start of Assistant 2's Response]:\n{response2}\n[The End of Assistant 2's Response]\n***\nPlease return [Judge Result]:\n"
    return prompt

def get_judgelm_reverse_pairwise_solving_prompt():
    prompt = "Given a [User Query], please score the responses from two AI assistants according to the [Evaluation Criteria] and [Scoring Guideline]. Ensure a comparative and objective assessment based on the evaluation criteria and scoring guideline, aiming to identify deficiencies in the response content. Provide a final score of 1-10 along with relevant explanations.\n***\n[User Query]:\n{query}\n***\n[Evaluation Criteria]:\n{criteria}\n***\n[Scoring Guideline]:\n{scoring}\n***\n[The Start of Assistant 2's Response]:\n{response2}\n[The End of Assistant 2's Response]\n***\n[The Start of Assistant 1's Response]:\n{response1}\n[The End of Assistant 1's Response]\n***\nPlease return [Judge Result]:\n"
    return prompt

def prepare_judgelm():

    judgelm_json_data = []
    df = json.load(open("data/fennec_train_data/final/train/fennec_judgelm_100k_reverse.jsonl"))
    for _, json_item in df.items():
        
        query = json_item['query']
        response_1 = json_item['response_a']
        response_2 = json_item['response_b']
        branch = json_item['branch'][0].strip()
        scoring = json_item['scoring'][0].strip()
        solving = json_item['solving'].strip()
        
        pairwise_solving_system = get_judgelm_pairwise_solving_system_message()
        pairwise_solving_prompt = get_judgelm_pairwise_solving_prompt().format(query=query, criteria=branch, scoring=scoring, response1=response_1, response2=response_2)
        judgelm_json_data.append({
            "prompt": query,
            "prompt_id": str(uuid.uuid4()), 
            "messages": [
                {"content": pairwise_solving_system, "role": "system"},
                {"content": pairwise_solving_prompt,  "role": "user"},
                {"content": solving, "role": "assistant"}
            ]
        })
        reverse_pairwise_solving_prompt = get_judgelm_reverse_pairwise_solving_prompt().format(query=query, criteria=branch, scoring=scoring, response1=response_1, response2=response_2)
        judgelm_json_data.append({
            "prompt": query,
            "prompt_id": str(uuid.uuid4()),
            "messages": [
                {"content": pairwise_solving_system, "role": "system"},
                {"content": reverse_pairwise_solving_prompt,  "role": "user"},
                {"content": solving, "role": "assistant"}
            ]
        })
    return judgelm_json_data

def get_prometheus_single_solving_system_message():
    message = "You are a fair, faithful, and helpful content evaluation assistant. Please assist me in completing the assigned task by providing Single-Score Evaluations for the given dialogue. (Tips: This involves assessing individual responses independently.)"
    return message

def get_prometheus_single_solving_prompt():
    prompt = "Given a [User Query], please score the responses from AI assistants according to the [Evaluation Criteria] and [Scoring Guideline]. Ensure a comparative and objective assessment based on the evaluation criteria and scoring guideline, aiming to identify deficiencies in the response content. \n***\n[User Query]:\n{query}\n***\n[Evaluation Criteria]:\n{criteria}\n***\n[Scoring Guideline]:\n{scoring}\n***\n[The Start of Response]:\n{response}\n[The End of Response]\n***\nAssign a score as an integer between 1 and 5. Provide a detailed [Judge Result] strictly based on the given Scoring Guideline, refraining from a general evaluation. Please return [Judge Result] as follows:\nResponse Score: 3\nExplanation: Explanation of the score for the Response.\nPlease return [Judge Result]:\n"
    return prompt

def prepare_prometheus():

    prometheus_json_data = []
    for line in open("data/fennec_train_data/final/train/prometheus_filter.jsonl").readlines():
        json_line = json.loads(line)

        query = json_line['query']
        response = json_line['response']
        branch = json_line['branch'].replace("[", "").replace("]", "")
        scoring = json_line['scoring']
        solving = "Response Score: {}\nExplanation: {}".format(json_line['judge'], json_line['judgment'].strip())
        
        random_noise = random.random()
        if json_line['judge'] == "1" and random_noise > 0.23:
            continue
        elif json_line['judge'] == "5" and random_noise > 0.3:
            continue
        elif json_line['judge'] == "2" and random_noise > 0.36:
            continue
        elif json_line['judge'] == "3" and random_noise > 0.54:
            continue
        elif json_line['judge'] == "4" and random_noise > 0.5:
            continue

        single_solving_system = get_prometheus_single_solving_system_message()
        single_solving_prompt = get_prometheus_single_solving_prompt().format(query=query, criteria=branch, scoring=scoring, response=response)
        prometheus_json_data.append({
            "prompt": query,
            "prompt_id": str(uuid.uuid4()),
            "messages": [
                {"content": single_solving_system, "role": "system"},
                {"content": single_solving_prompt,  "role": "user"},
                {"content": solving, "role": "assistant"}
            ]
        })
    return prometheus_json_data

fennec_data = prepare_fennec()
print("prepare_fennec", len(fennec_data))

autoj_data = prepare_autoj()
print("prepare_autoj", len(autoj_data))

judgelm_data = prepare_judgelm()
print("prepare_judgelm", len(judgelm_data))

prometheus_data = prepare_prometheus()
print("prepare_prometheus", len(prometheus_data))


# json_data = fennec_data + autoj_data + judgelm_data + prometheus_data
json_data = fennec_data 
random.shuffle(json_data)
print("all data", len(json_data))

df = pd.DataFrame(json_data[1000:])
df.to_parquet("data/fennec_train_data/final/train/train.parquet")

df = pd.DataFrame(json_data[:1000])
df.to_parquet("data/fennec_train_data/final/train/test.parquet")
