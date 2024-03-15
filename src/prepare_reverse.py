import pandas as pd
import pdb
import json

result2criterion = "Given a [User Query] and the subsequent responses (A and B) provided by two AI assistants, along with an in-depth and impartial [Evaluation Result] of these responses, please formulate an [Evaluation Criterion] that demonstrates a high correlation with the relevance of the user's query and assistant's responses.\n***\n[User Query]:\n{query}\n***\n[The Start of Response A]:\n{response1}\n[The End of Response A]\n***\n[The Start of Response B]:\n{response2}\n[The End of Response B]\n***\n[The Start of Evaluation Result]:\n{result}\n[The End of Evaluation Result]\n***\nPlease return [Evaluation Criterion]:\n"

result2score = "Given a [User Query] and the subsequent responses (A and B) provided by two AI assistants, along with a comprehensive and unbiased [Evaluation Criteria] and [Evaluation Result] for these responses. Please contemplate and summarize the scoring scales associated with these [Evaluation Results] to develop a rational and highly relevant [Scoring Guideline].\n***\n[User Query]:\n{query}\n***\n[Evaluation Criteria]:\n{criteria}\n***\n[The Start of Response A]:\n{response1}\n[The End of Response A]\n***\n[The Start of Response B]:\n{response2}\n[The End of Response B]\n***\n[The Start of Evaluation Result]:\n{result}\n[The End of Evaluation Result]\n***\nPlease return [Scoring Guideline]:\n"


def reverse_inf():
    df = pd.read_parquet("data/eval_file/fennec_train_new.parquet")
    with open("data/fennec_eval_data/raw/autoj_bench_reverse/train.jsonl", "w") as fw:
        for idx, item in df.iterrows():
            candi_branch = []
            for solving_result in item["solving"].tolist():
                text = result2criterion.format(
                    query=item["query"],
                    response1=item["response_1"],
                    response2=item["response_2"],
                    result=solving_result,
                )
                candi_branch.append(text)

            candi_scoring = []
            for b_item in item["branch"].tolist():
                text = result2score.format(
                    query=item["query"],
                    criteria=b_item,
                    response1=item["response_1"],
                    response2=item["response_2"],
                    result=solving_result,
                )
                candi_scoring.append(text)

            new_data = {
                "idx": [idx],
                "query": [item["query"]],
                "response_1": [item["response_1"]],
                "response_2": [item["response_2"]],
                "branch": [item["branch"].tolist()],
                "candi_branch": [candi_branch],
                "scoring": [item["scoring"].tolist()],
                "candi_scoring": [candi_scoring],
                "solving": [item["solving"].tolist()],
            }
            fw.writelines(json.dumps(new_data) + "\n")


reverse_inf()
# import pyarrow as pa
# import pyarrow.parquet as pq

# df = pd.read_parquet("data/eval_file/fennec_train.parquet")
# json_dict = json.load(open("data/eval_file/fennec_train_reverse.jsonl"))
# new_table = None

# for idx, item in df.iterrows():
#     if item["query"] in json_dict:
#         new_data = pd.DataFrame(
#             {
#                 "query": [item["query"]],
#                 "response_1": [item["response_1"]],
#                 "response_2": [item["response_2"]],
#                 "branch": [item["branch"].tolist()],
#                 "scoring": [item["scoring"].tolist()],
#                 "solving": [item["solving"].tolist()],
#                 "reverse_branch": [json_dict[item["query"]]["branch"]],
#                 "reverse_scoring": [json_dict[item["query"]]["scoring"]],
#             }
#         )
#     else:
#         new_data = pd.DataFrame(
#             {
#                 "query": [item["query"]],
#                 "response_1": [item["response_1"]],
#                 "response_2": [item["response_2"]],
#                 "branch": [item["branch"].tolist()],
#                 "scoring": [item["scoring"].tolist()],
#                 "solving": [item["solving"].tolist()],
#                 "reverse_branch": [[""] * 5],
#                 "reverse_scoring": [[""] * 5],
#             }
#         )
#     table = pa.Table.from_pandas(new_data)
#     if new_table:
#         new_table = pa.concat_tables([new_table, table])
#     else:
#         new_table = table

# pq.write_table(new_table, "data/eval_file/fennec_train_new.parquet")
