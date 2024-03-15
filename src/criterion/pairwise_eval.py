import pdb
import time
from config.task_config import TaskConfig
from .meta import Evaluation
from .registry import auto_register
from actions.comparing import PairwiseComparingAction
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pearsonr


@auto_register("pairwise_eval")
class PairwiseEvaluation(Evaluation):
    task_name = "pairwise_eval"

    def __init__(self, config: TaskConfig, task_func) -> None:
        super().__init__(config)

        self.task_func = task_func
        self.eval_json_file = self.config.get_eval_json_file(self.task_name)
        self.eval_score = {}

    def eval(self, eval_event):
        dialogue = eval_event.get_dialogue()
        
        if self.task_func == "pairwise_eval_func":
            meta_info = dialogue.get_meta_info()
            turn = meta_info["turn"] - 1
            pc_action_feedback = eval_event.get_memories(
                self.task_name,
                PairwiseComparingAction.action_name,
                str("turn{}".format(turn)),
            )
            pred = pc_action_feedback["prediction"]
            ex_pred = pc_action_feedback["ex_prediction"]
            judge = meta_info["judge"][0]

            if "agreement" not in self.eval_score:
                self.eval_score["model_a"] = 0
                self.eval_score["model_b"] = 0
                self.eval_score["tie"] = 0
                self.eval_score["agreement"] = []
                self.eval_score["consistency"] = []
                self.eval_score["error"] = 0
                self.eval_score["single_agreement"] = []
                self.eval_score["single_error"] = 0

            if isinstance(judge, str):
                if judge == "model_a":
                    judge = 0
                elif judge == "model_b":
                    judge = 1
                elif "tie" in judge:
                    judge = 2
            # elif len(meta_info["judge"]) == 3:
            #     # panda lm
            #     if judge == 0:
            #         judge = 2
            #     elif judge == 1:
            #         judge = 0
            #     else:
            #         judge = 1

            if len(meta_info["judge"]) > 2:
                if pred == 0:
                    self.eval_score["model_a"] += 1
                elif pred == 1:
                    self.eval_score["model_b"] += 1
                elif pred == 2:
                    self.eval_score["tie"] += 1
                    
                if pred == 0 and judge == 1:
                    self.eval_score["single_agreement"].append(1)
                elif pred == 1 and judge == 2:
                    self.eval_score["single_agreement"].append(1)
                elif pred == 2 and judge == 0:
                    self.eval_score["single_agreement"].append(1)
                else:
                    if pred == -1:
                        self.eval_score["single_error"] += 1
                    self.eval_score["single_agreement"].append(0)

                if pred == 0 and ex_pred == 0:
                    if judge == 1:
                        self.eval_score["agreement"].append(1)
                    else:
                        self.eval_score["agreement"].append(0)
                    self.eval_score["consistency"].append(1)
                elif pred == 1 and ex_pred == 1:
                    if judge == 2:
                        self.eval_score["agreement"].append(1)
                    else:
                        self.eval_score["agreement"].append(0)
                    self.eval_score["consistency"].append(1)
                elif pred == 2 and ex_pred == 2:
                    if judge == 0:
                        self.eval_score["agreement"].append(1)
                    else:
                        self.eval_score["agreement"].append(0)
                    self.eval_score["consistency"].append(1)
                else:
                    if pred == -1 or ex_pred == -1:
                        self.eval_score["error"] += 1
                        
                    self.eval_score["agreement"].append(0)
                    self.eval_score["consistency"].append(0)
            else:
                if pred == 0:
                    self.eval_score["model_a"] += 1
                elif pred == 1:
                    self.eval_score["model_b"] += 1
                elif pred == 2:
                    self.eval_score["tie"] += 1

                if pred == 0 and judge == 0:
                    self.eval_score["single_agreement"].append(1)
                elif pred == 1 and judge == 1:
                    self.eval_score["single_agreement"].append(1)
                elif pred == 2 and judge == 2:
                    self.eval_score["single_agreement"].append(1)
                else:
                    if pred == -1:
                        self.eval_score["single_error"] += 1
                    self.eval_score["single_agreement"].append(0)

                if pred == 0 and ex_pred == 0:
                    if judge == 0:
                        self.eval_score["agreement"].append(1)
                    else:
                        self.eval_score["agreement"].append(0)
                    self.eval_score["consistency"].append(1)
                elif pred == 1 and ex_pred == 1:
                    if judge == 1:
                        self.eval_score["agreement"].append(1)
                    else:
                        self.eval_score["agreement"].append(0)
                    self.eval_score["consistency"].append(1)
                elif pred == 2 and ex_pred == 2:
                    if judge == 2:
                        self.eval_score["agreement"].append(1)
                    else:
                        self.eval_score["agreement"].append(0)
                    self.eval_score["consistency"].append(1)
                else:
                    if pred == -1 or ex_pred == -1:
                        self.eval_score["error"] += 1
                    self.eval_score["agreement"].append(0)
                    self.eval_score["consistency"].append(0)
            # pdb.set_trace()

    def serialize(self):
        time.sleep(1)
        if len(self.eval_score):
            self.logger.info("Error = {}".format(self.eval_score["error"]))
            agreement = self.eval_score["agreement"]
            self.logger.info(
                "Agreement Average Score = {} = {} / {}".format(
                    str(sum(agreement) / (len(agreement) - self.eval_score["error"])),
                    str(sum(agreement)),
                    str(len(agreement) - self.eval_score["error"]),
                )
            )
            consistency = self.eval_score["consistency"]
            self.logger.info(
                "Consistency Average Score = {} = {} / {}".format(
                    str(
                        sum(consistency) / (len(consistency) - self.eval_score["error"])
                    ),
                    str(sum(consistency)),
                    str(len(consistency) - self.eval_score["error"]),
                )
            )

            self.logger.info(
                "Single Error = {}".format(self.eval_score["single_error"])
            )
            single_agreement = self.eval_score["single_agreement"]
            self.logger.info(
                "Single Agreement Average Score = {} = {} / {}".format(
                    str(
                        sum(single_agreement)
                        / (len(single_agreement) - self.eval_score["single_error"])
                    ),
                    str(sum(single_agreement)),
                    str(len(single_agreement) - self.eval_score["single_error"]),
                )
            )

            self.logger.info(
                "Model a {} Model b {} tie {}".format(
                    self.eval_score["model_a"],
                    self.eval_score["model_b"],
                    self.eval_score["tie"],
                )
            )
