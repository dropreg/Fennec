import pdb
import json

# with open("pandalm_test.json", 'w') as fw:
#     for json_item in json.load(open("data/fennec_eval_data/raw/pandalm/testset-v1.json")):

#         if json_item['annotator1'] == 1:
#             judge = "win"
#         elif json_item['annotator1'] == 2:
#             judge = "lose"
#         elif json_item['annotator1'] == 0:
#             judge = "tie"    
#         else:
#             print(json_item['annotator1'])
#             raise Exception("error")
        
#         dump_data = {
#             "id": json_item['idx'],
#             "query": "Instruction: {}\nInput: {}".format(json_item['instruction'], json_item['input']),
#             "response_1": json_item['response1'],
#             "response_2": json_item['response2'],
#             "judge": judge 
#         }
#         fw.writelines(json.dumps(dump_data) + "\n")

# with open("mt_bench.json", 'w') as fw:
#     for line in open("data/fennec_eval_data/raw/mt_bench/human_pair_3355.jsonl").readlines():
#         json_item = json.loads(line)        
#         if json_item['turn'] == 1:
            
#             if json_item['winner'] == "model_a":
#                 judge = "win"
#             elif json_item['winner'] == "model_b":
#                 judge = "lose"
#             elif "tie" in json_item['winner']:
#                 judge = "tie"
#             else:
#                 raise Exception("error")
            
#             dump_data = {
#                 "id": json_item['question_id'],
#                 "query": json_item['conversation_a'][0]['content'],
#                 "response_1": json_item['conversation_a'][1]['content'],
#                 "response_2": json_item['conversation_b'][1]['content'],
#                 "judge": judge 
#             }
#             fw.writelines(json.dumps(dump_data) + "\n")
