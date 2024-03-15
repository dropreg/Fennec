import pdb
import time
from config.task_config import TaskConfig
from .meta import Evaluation
from .registry import auto_register
from actions.tool_utils import ScenarioJudgeAction
from sklearn.metrics import f1_score, precision_score, recall_score
import json


@auto_register("tool_awareness_eval")
class ToolAwarenessEvaluation(Evaluation):
    task_name = "tool_awareness_eval"

    def __init__(self, config: TaskConfig, task_func) -> None:
        super().__init__(config)

        self.task_func = task_func
        self.eval_score = {}

    def result_format(self, tmp_result):
        tmp_result = tmp_result.lower()
        pred_label = -1
        if "\"yes.\"" in tmp_result or "\"yes\"" in tmp_result or "yes," in tmp_result or "yes." in tmp_result:
            pred_label = 1
        elif "\"no.\"" in tmp_result or "\"no\"" in tmp_result or "no," in tmp_result or "no." in tmp_result:
            pred_label = 0
        elif "yes" in tmp_result and "no " not in tmp_result and "not" not in tmp_result and "don't" not in tmp_result and "\"no\"" not in tmp_result:
            pred_label = 1
        elif ("it is necessary to" in tmp_result or "it's necessary to" in tmp_result or "would recommend using external tools" in tmp_result or "external tools may be necessary to" in tmp_result or "think it is necessary" in tmp_result or "may need to" in tmp_result or "would need to" in tmp_result or "believe it is necessary to" in tmp_result or "would need access" in tmp_result or "might be necessary to" in tmp_result or "might need to" in tmp_result or "tools would be necessary" in tmp_result or "may be necessary to" in tmp_result or "be beneficial to" in tmp_result or "will need to" in tmp_result or "might need to" in tmp_result or "would need to rely"  in tmp_result or "may need to access" in tmp_result or "may be necessary to" in tmp_result):
            pred_label = 1
        elif ("not seem necessary" in tmp_result or "not think it is necessary" in tmp_result or "not need to" in tmp_result or "not necessary to" in tmp_result or "do not think i need to" in tmp_result or "may not be necessary to" in tmp_result or "not necessary to" in tmp_result or "does not explicitly require" in tmp_result or "do not believe it is necessary to" in tmp_result or "it does not indicate a need" in tmp_result) and "yes" not in tmp_result:
            pred_label = 0
        elif "no " in tmp_result or "no." in tmp_result or "don't" in tmp_result or "\"no\"" in tmp_result:
            pred_label = 0   
        else:
            pass
        return pred_label

    def eval(self, eval_event):
        dialogue = eval_event.get_dialogue()

        if self.task_func == "single_eval_func" or self.task_func == "demon_eval_func":
            meta_info = dialogue.get_meta_info()
            turn = meta_info["turn"] - 1

            sj_action_feedback = eval_event.get_memories(
                self.task_name,
                ScenarioJudgeAction.action_name,
                str("turn{}".format(turn)),
            )
            pred_label = self.result_format(sj_action_feedback['result'])

            with open("mistral_single_2.jsonl", 'a') as fw:
                fw.writelines(json.dumps({
                    "question_id": meta_info['question_id'],
                    "result": sj_action_feedback['result'],
                    "label": pred_label,
                    "gold": meta_info['judge'][0],
                }) + "\n")
            
            if pred_label >= 0:
                if "pred" not in self.eval_score:
                    self.eval_score['pred'] = []
                    self.eval_score['ground_truth'] = []
                    self.eval_score['pred_neg'] = 0
                    self.eval_score['pred_pos'] = 0
                    self.eval_score['acc'] = 0

                if (pred_label == 0 and meta_info['judge'][0] == 'negative') or (
                    pred_label == 1 and meta_info['judge'][0] != 'negative'
                ):
                    self.eval_score['acc'] += 1

                self.eval_score['pred'].append(pred_label)
                if meta_info['judge'][0] == 'negative':
                    self.eval_score['ground_truth'].append(0)
                    if pred_label == 0:
                        self.eval_score['pred_neg'] += 1
                    else:
                        pass
                        # pdb.set_trace()
                else:
                    self.eval_score['ground_truth'].append(1)
                    if pred_label == 1:
                        self.eval_score['pred_pos'] += 1
                    else:
                        pass
                        # pdb.set_trace()
        else:
            raise Exception("Not Support")

    def serialize(self):
        time.sleep(1)
        if len(self.eval_score):
            if self.task_func == "single_eval_func" or self.task_func == "demon_eval_func":
                pred = self.eval_score['pred']
                ground_truth = self.eval_score['ground_truth']
                acc = self.eval_score['acc'] / len(pred)
                self.logger.info("All Data Number = {}".format(len(pred)))
                self.logger.info("Pred_pos = {}, Pred_neg = {}".format(self.eval_score['pred_pos'], self.eval_score['pred_neg']))
                self.logger.info("F1 score = {}".format(f1_score(ground_truth, pred)))
                self.logger.info("Precision score = {}".format(precision_score(ground_truth, pred)))
                self.logger.info("Recall score = {}".format(recall_score(ground_truth, pred)))
                self.logger.info("Acc score = {}".format(acc))
