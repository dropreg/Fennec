import json
import pdb
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from data.eval_event import EvalEvent
from .meta import Evaluation
from .registry import auto_register
from actions.dialogue_gen import DialogueGenAction
from config.task_config import TaskConfig
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import threading


write_lock = threading.Lock()


@auto_register("dialogue_gen")
class DialogueGenEvaluation(Evaluation):
    task_name = "dialogue_gen"

    def __init__(self, config: TaskConfig, task_func) -> None:
        super().__init__(config)

        self.task_func = task_func
        self.dialogue_generation_file = self.config.get_eval_json_file(self.task_name)

        self.gen_results = []

    def eval(self, eval_event: EvalEvent):
        dialogue = eval_event.get_dialogue()
        meta_info = dialogue.get_meta_info()

        dg_action_feedback = eval_event.get_memories(
            self.task_name,
            DialogueGenAction.action_name,
            str("turn{}".format(0)),
        )
        
        r = {
            "query": dialogue.get_query_by_idx(0)["content"],
            "original_response": dialogue.get_response_by_idx(0)["content"],
            "{}_response".format(dg_action_feedback["eval_model"]): dg_action_feedback["result"],
            "meta_info": meta_info,
        }
        self.dialogue_generation_file = self.dialogue_generation_file.replace("model", dg_action_feedback["eval_model"])
        self.gen_results.append(r)

    def serialize(self):
        with open(self.dialogue_generation_file, 'w') as o:
            for line in self.gen_results:
                o.write(json.dumps(line, ensure_ascii=False) + '\n')
