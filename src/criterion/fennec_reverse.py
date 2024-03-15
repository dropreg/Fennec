import json
import pdb
from tkinter import E
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from data.eval_event import EvalEvent
from .meta import Evaluation
from .registry import auto_register
from actions.fennec_reverse import FennecReverseBranchAction, FennecReverseScoringAction
from config.task_config import TaskConfig


@auto_register("fennec_reverse")
class FennecReverseEvaluation(Evaluation):
    task_name = "fennec_reverse"

    def __init__(self, config: TaskConfig, task_func) -> None:
        super().__init__(config)

        self.task_func = task_func
        self.train_parquet_file = self.config.get_train_parquet_file(self.task_name)
        self.eval_gen = []

    def eval(self, eval_event: EvalEvent):
        dialogue = eval_event.get_dialogue()

        if self.task_func == "pairwise_gen_func":
            meta_info = dialogue.get_meta_info()
            turn = meta_info["turn"] - 1

            frb_action_feedback = eval_event.get_memories(
                self.task_name,
                FennecReverseBranchAction.action_name,
                str("turn{}".format(turn)),
            )

            frs_action_feedback = eval_event.get_memories(
                self.task_name,
                FennecReverseScoringAction.action_name,
                str("turn{}".format(turn)),
            )
            
            self.eval_gen.append({
                "query": dialogue.get_query_by_idx(0)["content"],
                "response_a": dialogue.get_pairwise_response_by_idx(0, "model_a")[
                    "content"
                ],
                "response_b": dialogue.get_pairwise_response_by_idx(0, "model_b")[
                    "content"
                ],
                "branch": frb_action_feedback["branch_list"],
                "scoring": frs_action_feedback["scoring_list"],
                "solving": eval_event.get_memories("memory", "result"),
            })

    def serialize(self):
        if len(self.eval_gen):
            if self.task_func == "pairwise_gen_func":
                print(len(self.eval_gen))
                json.dump(
                    self.eval_gen,
                    open(self.train_parquet_file, "w"),
                    ensure_ascii=False,
                )
