import pdb
from .meta import Evaluation
from .registry import auto_register
import pandas as pd
from actions.bsm import (
    BSMPairwiseMergeAction,
    BSMSingleMergeAction,
)
from config.task_config import TaskConfig
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pearsonr


@auto_register("bsm_eval")
class BranchSolveMergeEvaluation(Evaluation):
    task_name = "bsm_eval"

    def __init__(self, config: TaskConfig, task_func) -> None:
        super().__init__(config)

        self.task_func = task_func
        self.eval_json_file = self.config.get_eval_json_file(self.task_name)
        self.eval_score = {}

    def eval(self, eval_event):
        dialogue = eval_event.get_dialogue()

        if self.task_func == "pairwise_bsm_eval_func":
            meta_info = dialogue.get_meta_info()
            turn = meta_info["turn"] - 1

            bsmpm_action_feedback = eval_event.get_memories(
                self.task_name,
                BSMPairwiseMergeAction.action_name,
                str("turn{}".format(turn)),
            )
            judge = meta_info["judge"][0]
            if "agreement" not in self.eval_score:
                self.eval_score["model_a"] = 0
                self.eval_score["model_b"] = 0
                self.eval_score["tie"] = 0

                self.eval_score["agreement"] = []
                self.eval_score["consistency"] = []
                self.eval_score["error"] = 0
                self.eval_score["single_agreement"] = []

            if (
                bsmpm_action_feedback["ex_model_a"]
                < bsmpm_action_feedback["ex_model_b"]
                and judge == 0
            ):
                self.eval_score["single_agreement"].append(1)
            elif (
                bsmpm_action_feedback["ex_model_a"]
                > bsmpm_action_feedback["ex_model_b"]
                and judge == 1
            ):
                self.eval_score["single_agreement"].append(1)
            elif (
                bsmpm_action_feedback["ex_model_a"]
                == bsmpm_action_feedback["ex_model_b"]
                and judge == 2
            ):
                self.eval_score["single_agreement"].append(1)
            else:
                self.eval_score["single_agreement"].append(0)

            if (
                bsmpm_action_feedback["model_a"] > bsmpm_action_feedback["model_b"]
                and bsmpm_action_feedback["ex_model_a"]
                < bsmpm_action_feedback["ex_model_b"]
            ):
                if judge == 0:
                    self.eval_score["agreement"].append(1)
                else:
                    self.eval_score["agreement"].append(0)
                self.eval_score["consistency"].append(1)
            elif (
                bsmpm_action_feedback["model_a"] < bsmpm_action_feedback["model_b"]
                and bsmpm_action_feedback["ex_model_a"]
                > bsmpm_action_feedback["ex_model_b"]
            ):
                if judge == 1:
                    self.eval_score["agreement"].append(1)
                else:
                    self.eval_score["agreement"].append(0)
                self.eval_score["consistency"].append(1)
            elif (
                bsmpm_action_feedback["model_a"] == bsmpm_action_feedback["model_b"]
                and bsmpm_action_feedback["ex_model_a"]
                == bsmpm_action_feedback["ex_model_b"]
            ):
                if judge == 2:
                    self.eval_score["agreement"].append(1)
                else:
                    self.eval_score["agreement"].append(0)
                self.eval_score["consistency"].append(1)
            else:
                if (
                    bsmpm_action_feedback["model_a"] == 0
                    or bsmpm_action_feedback["model_b"] == 0
                    or bsmpm_action_feedback["ex_model_a"] == 0
                    or bsmpm_action_feedback["ex_model_b"] == 0
                ):
                    self.eval_score["error"] += 1
                self.eval_score["agreement"].append(0)
                self.eval_score["consistency"].append(0)
        elif self.task_func == "single_bsm_eval_func":
            meta_info = dialogue.get_meta_info()
            turn = meta_info["turn"] - 1

            bsmsm_action_feedback = eval_event.get_memories(
                self.task_name,
                BSMSingleMergeAction.action_name,
                str("turn{}".format(turn)),
            )
            
            judge = meta_info["judge"][0]
            if "agreement" not in self.eval_score:
                self.eval_score["model_a"] = 0
                self.eval_score["model_b"] = 0
                self.eval_score["tie"] = 0

                self.eval_score["agreement"] = []
                self.eval_score["consistency"] = []
                self.eval_score["error"] = 0
                self.eval_score["single_agreement"] = []

            if (
                bsmsm_action_feedback["ex_model_a"]
                > bsmsm_action_feedback["ex_model_b"]
                and judge == 0
            ):
                self.eval_score["single_agreement"].append(1)
            elif (
                bsmsm_action_feedback["ex_model_a"]
                < bsmsm_action_feedback["ex_model_b"]
                and judge == 1
            ):
                self.eval_score["single_agreement"].append(1)
            elif (
                bsmsm_action_feedback["ex_model_a"]
                == bsmsm_action_feedback["ex_model_b"]
                and judge == 2
            ):
                self.eval_score["single_agreement"].append(1)
            else:
                self.eval_score["single_agreement"].append(0)

            if (
                bsmsm_action_feedback["model_a"] > bsmsm_action_feedback["model_b"]
                and bsmsm_action_feedback["ex_model_a"]
                > bsmsm_action_feedback["ex_model_b"]
            ):
                if judge == 0:
                    self.eval_score["agreement"].append(1)
                else:
                    self.eval_score["agreement"].append(0)
                self.eval_score["consistency"].append(1)
            elif (
                bsmsm_action_feedback["model_a"] < bsmsm_action_feedback["model_b"]
                and bsmsm_action_feedback["ex_model_a"]
                < bsmsm_action_feedback["ex_model_b"]
            ):
                if judge == 1:
                    self.eval_score["agreement"].append(1)
                else:
                    self.eval_score["agreement"].append(0)
                self.eval_score["consistency"].append(1)
            elif (
                bsmsm_action_feedback["model_a"] == bsmsm_action_feedback["model_b"]
                and bsmsm_action_feedback["ex_model_a"]
                == bsmsm_action_feedback["ex_model_b"]
            ):
                if judge == 2:
                    self.eval_score["agreement"].append(1)
                else:
                    self.eval_score["agreement"].append(0)
                self.eval_score["consistency"].append(1)
            else:
                if (
                    bsmsm_action_feedback["model_a"] == 0
                    or bsmsm_action_feedback["model_b"] == 0
                    or bsmsm_action_feedback["ex_model_a"] == 0
                    or bsmsm_action_feedback["ex_model_b"] == 0
                ):
                    self.eval_score["error"] += 1
                self.eval_score["agreement"].append(0)
                self.eval_score["consistency"].append(0)

    def serialize(self):
        if len(self.eval_score):
            self.logger.info("Error = {}".format(self.eval_score["error"]))
            agreement = self.eval_score["agreement"]
            self.logger.info(
                "Agreement Average Score = {} = {} / {}".format(
                    str(sum(agreement) / len(agreement)),
                    str(sum(agreement)),
                    str(len(agreement)),
                )
            )
            consistency = self.eval_score["consistency"]
            self.logger.info(
                "Consistency Average Score = {} = {} / {}".format(
                    str(sum(consistency) / len(consistency)),
                    str(sum(consistency)),
                    str(len(consistency)),
                )
            )
            single_agreement = self.eval_score["single_agreement"]
            self.logger.info(
                "Single Agreement Average Score = {} = {} / {}".format(
                    str(sum(single_agreement) / (len(single_agreement))),
                    str(sum(single_agreement)),
                    str(len(single_agreement)),
                )
            )
