import pdb
import time
from config.task_config import TaskConfig
from .meta import Evaluation
from .registry import auto_register
from actions.scoring import SingleScoringAction, PairwiseSingleScoringAction
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pearsonr


@auto_register("single_eval")
class SingleEvaluation(Evaluation):
    task_name = "single_eval"

    def __init__(self, config: TaskConfig, task_func) -> None:
        super().__init__(config)

        self.task_func = task_func
        self.eval_json_file = self.config.get_eval_json_file(self.task_name)
        self.eval_score = {}

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
            model_id = meta_info["model"]
            if model_id not in self.eval_score:
                self.eval_score[model_id] = {"prediction": [], "label": []}

            if str(turn) not in self.eval_score[model_id]:
                self.eval_score[model_id][str(turn)] = [ss_action_feedback["rating"]]
            else:
                self.eval_score[model_id][str(turn)].append(
                    ss_action_feedback["rating"]
                )

            self.eval_score[model_id]["prediction"].append(ss_action_feedback["rating"])
            self.eval_score[model_id]["label"].append(meta_info["judge"][0])

        elif self.task_func == "pairwise_single_eval_func":
            meta_info = dialogue.get_meta_info()

            if meta_info["turn"] == 2:
                return
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
            
            if "agreement" not in self.eval_score:
                self.eval_score["agreement"] = []
                self.eval_score["prediction"] = []
                self.eval_score["label"] = []
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
                sorted_pred_model = {}
                sorted_label_model = {}
                for model_id, model_value in self.eval_score.items():
                    self.logger.info("#" * 20)
                    self.logger.info(model_id)

                    all_prediction = []
                    all_label = []

                    turn_sum = []
                    for turn_key, turn_value_tmp in model_value.items():
                        if turn_key == "prediction":
                            all_prediction.extend(turn_value_tmp)
                            continue
                        if turn_key == "label":
                            all_label.extend(turn_value_tmp)
                            continue

                        turn_value = []
                        error = 0
                        for v in turn_value_tmp:
                            if v > 0:
                                turn_value.append(v)
                            else:
                                error += 1
                        turn_sum.extend(turn_value)

                        self.logger.info(
                            "Turn {} Score = {} Error = {}".format(
                                turn_key,
                                str(sum(turn_value) / len(turn_value)),
                                str(error),
                            )
                        )

                    self.logger.info(
                        "Average Score = {} = {} / {}".format(
                            str(sum(turn_sum) / len(turn_sum)),
                            str(sum(turn_sum)),
                            str(len(turn_sum)),
                        )
                    )
                    sorted_pred_model[model_id] = sum(turn_sum) / len(turn_sum)
                    sorted_label_model[model_id] = sum(all_label) / len(all_label)

                    p_coef, p_value = pearsonr(all_prediction, all_label)
                    self.logger.info("Pearsonr = {} P_value {}".format(p_coef, p_value))
                    tau, p_value = kendalltau(all_prediction, all_label)
                    self.logger.info("Kendall tau = {} P_value {}".format(tau, p_value))
                    rho, p_value = spearmanr(all_prediction, all_label)
                    self.logger.info("Spearmanr = {} P_value {}".format(rho, p_value))

                self.logger.info("#" * 20)
                self.logger.info(
                    dict(sorted(sorted_pred_model.items(), key=lambda item: item[1]))
                )
                self.logger.info("#" * 20)
                self.logger.info(
                    dict(sorted(sorted_label_model.items(), key=lambda item: item[1]))
                )

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
