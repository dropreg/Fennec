import pdb
import time
from config.task_config import TaskConfig
from .meta import Evaluation
from .registry import auto_register
from actions.scoring import SingleScoringAction, PairwiseSingleScoringAction
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


@auto_register("prometheus_eval")
class PrometheusEvaluation(Evaluation):
    task_name = "prometheus_eval"

    def __init__(self, config: TaskConfig, task_func) -> None:
        super().__init__(config)

        self.task_func = task_func
        self.eval_json_file = self.config.get_eval_json_file(self.task_name)
        self.eval_score = {}
        self.eval_gen = {"table": None}

    def eval(self, eval_event):
        dialogue = eval_event.get_dialogue()

        if self.task_func == "single_eval_func":
            meta_info = dialogue.get_meta_info()
            turn = meta_info["turn"] - 1
            
            ss_action_feedback = eval_event.get_memories(
                self.task_name,
                SingleScoringAction.action_name,
                str("turn{}".format(turn)),
            )
            query = dialogue.get_query_by_idx(0)["content"]
            response = dialogue.get_response_by_idx(0)["content"]

            if "acc" not in self.eval_score:
                self.eval_score["acc"] = 0
                self.eval_score["count"] = 0
            if meta_info["judge"][0] == str(ss_action_feedback["rating"]):
                self.eval_score['acc'] += 1
                
                new_data = pd.DataFrame(
                    {
                        "question_id": [meta_info['question_id']],
                        "query": [query],
                        "response": [response],
                        "judge": [meta_info["judge"][0]],
                        "judgment": [meta_info['judgment'][0]],
                        "instruction": [meta_info['instruction']],
                        "reference": [meta_info['reference']],
                    }
                )
                table = pa.Table.from_pandas(new_data)
                if self.eval_gen["table"] is None:
                    self.eval_gen["table"] = table
                else:
                    self.eval_gen["table"] = pa.concat_tables(
                        [self.eval_gen["table"], table]
                    )

            self.eval_score["count"] += 1
        elif self.task_func == "pairwise_single_eval_func":
            meta_info = dialogue.get_meta_info()
            turn = meta_info["turn"] - 1
            
            pss_action_feedback = eval_event.get_memories(
                self.task_name,
                PairwiseSingleScoringAction.action_name,
                str("turn{}".format(turn)),
            )
            rating = pss_action_feedback["rating"]
            judge = meta_info["judge"][0]

            if isinstance(judge, str):
                if judge == "model_a":
                    judge = 0
                elif judge == "model_b":
                    judge = 1
                elif "tie" in judge:
                    judge = 2
            elif len(meta_info["judge"]) == 3:
                # panda lm
                if judge == 0:
                    judge = 2
                elif judge == 1:
                    judge = 0
                else:
                    judge = 1

            if "acc" not in self.eval_score:
                self.eval_score["acc"] = 0
                self.eval_score["count"] = 0
            
            if "agreement" not in self.eval_score:
                self.eval_score["agreement"] = []
                self.eval_score["error"] = 0

            if rating["model_a"] > rating["model_b"] and judge == 0:
                self.eval_score["agreement"].append(1)
            elif rating["model_a"] < rating["model_b"] and judge == 1:
                self.eval_score["agreement"].append(1)
            elif rating["model_a"] == rating["model_b"] and judge == 2:
                self.eval_score["agreement"].append(1)
            else:
                if rating["model_a"] == -1 or rating["model_b"] == -1:
                    self.eval_score["error"] += 1
                self.eval_score["agreement"].append(0)
        else:
            raise Exception("Not Support")

    def serialize(self):
        time.sleep(1)
        if len(self.eval_score):
            if self.task_func == "single_eval_func":
                acc = self.eval_score["acc"]
                count = self.eval_score["count"]
                self.logger.info(
                    "Acc Score = {} = {} / {}".format(
                        str(acc / count),
                        str(acc),
                        str(count),
                    )
                )
                pq.write_table(self.eval_gen["table"], self.eval_json_file)
            elif self.task_func == "pairwise_single_eval_func":
                agreement = self.eval_score["agreement"]
                self.logger.info("Error = {}".format(self.eval_score["error"]))
                self.logger.info(
                    "Agreement Average Score = {} = {} / {}".format(
                        str(
                            sum(agreement) / (len(agreement) - self.eval_score["error"])
                        ),
                        str(sum(agreement)),
                        str(len(agreement) - self.eval_score["error"]),
                    )
                )
