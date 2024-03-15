from mmap import ACCESS_READ
from os import access
import pdb
import json
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score


consistency = 0
agreement = 0
count = 0

pred = []
gold = []

for line0, line1, line2, line3, line4 in zip(open("mistral_single_0.jsonl").readlines(), open("mistral_single_1.jsonl").readlines(), open("mistral_single_2.jsonl").readlines(), open("mistral_no_example_3.jsonl").readlines(), open("mistral_no_example_4.jsonl").readlines()):
    
    data0 = json.loads(line0)
    data1 = json.loads(line1)
    data2 = json.loads(line2)
    data3 = json.loads(line3)
    data4 = json.loads(line4)

    if data0['label'] == data1['label']:
        consistency += 1
    assert data0['question_id'] == data1['question_id'] == data2['question_id'] 
    scores = {v:k for k, v in Counter([data0['label'], data1['label']]).items()}

    if 5 in scores:
        pred_label = scores[5]
    elif 4 in scores:
        pred_label = scores[4]
    elif 3 in scores:
        pred_label = scores[3]
    elif 2 in scores:
        pred_label = scores[2]
    elif 1 in scores:
        pred_label = scores[1]
    else:
        pred_label = -1
    # pred_label = data2['label']
    
    if pred_label >= 0:
        pred.append(pred_label)
        if data0['gold'] == "positive":
            gold.append(1)
        else:
            gold.append(0)
        if pred_label == 1 and data0['gold'] == "positive":
            agreement += 1
        elif pred_label == 0 and data0['gold'] != "positive":
            agreement += 1
    else:
        continue
    # pdb.set_trace()   
    count += 1
    
print(consistency, count, consistency/count)
print(agreement, count, agreement/count)

print("F1 score = {}".format(f1_score(gold, pred)))
print("Precision score = {}".format(precision_score(gold, pred)))
print("Recall score = {}".format(recall_score(gold, pred)))
