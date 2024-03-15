import json
import pdb
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from data.eval_event import EvalEvent
from .meta import Evaluation
from .registry import auto_register
from actions.fennec import (
    FennecBranchAction,
    FennecScoringAction,
    FennecPairwiseSolvingAction,
    FennecPairwiseMergeAction,
    FennecCorrectionAction,
)
from config.task_config import TaskConfig
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import threading


write_lock = threading.Lock()


@auto_register("fennec")
class FennecEvaluation(Evaluation):
    task_name = "fennec"

    def __init__(self, config: TaskConfig, task_func) -> None:
        super().__init__(config)

        self.task_func = task_func
        self.train_parquet_file = self.config.get_train_parquet_file(self.task_name)
        self.train_filter_parquet_file = self.config.get_train_filter_parquet_file(self.task_name)

        self.eval_score = {}
        self.eval_gen = {}
        self.eval_filter_gen = {}

        self.correction = {}

    def eval(self, eval_event: EvalEvent):
        dialogue = eval_event.get_dialogue()

        if self.task_func == "pairwise_eval_func":
            meta_info = dialogue.get_meta_info()
            turn = meta_info["turn"] - 1

            fb_action_feedback = eval_event.get_memories(
                self.task_name,
                FennecBranchAction.action_name,
                str("turn{}".format(turn)),
            )
            fs_action_feedback = eval_event.get_memories(
                self.task_name,
                FennecScoringAction.action_name,
                str("turn{}".format(turn)),
            )
            fps_action_feedback = eval_event.get_memories(
                self.task_name,
                FennecPairwiseSolvingAction.action_name,
                str("turn{}".format(turn)),
            )
            fpm_action_feedback = eval_event.get_memories(
                self.task_name,
                FennecPairwiseMergeAction.action_name,
                str("turn{}".format(turn)),
            )
            
            judge = meta_info["judge"][0]
            # import pdb; pdb.set_trace()
            # print(fpm_action_feedback)

            # print(len(fb_action_feedback['branch_list']))
            # fc_action_feedback = eval_event.get_memories(
            #     self.task_name,
            #     FennecCorrectionAction.action_name,
            #     str("turn{}".format(turn)),
            # )
            
            # query = dialogue.get_query_by_idx(0)["content"]
            # response_1 = dialogue.get_pairwise_response_by_idx(0, "model_a")["content"]
            # response_2 = dialogue.get_pairwise_response_by_idx(0, "model_b")["content"]
            # self.correction[meta_info['question_id']] = {
            #     "query": query,
            #     "response_1": response_1,
            #     "correction_1": fc_action_feedback['result_a'].replace("\n<|assistant|>\n", "") if "result_a" in fc_action_feedback else "",
            #     "response_2": response_2,
            #     "correction_2": fc_action_feedback['result_b'].replace("\n<|assistant|>\n", "") if "result_b" in fc_action_feedback else "",
            #     "judge": judge,
            #     "score_1": fpm_action_feedback["model_a"],
            #     "score_2": fpm_action_feedback["model_b"],
            # }

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
                self.eval_score["consistency"] = []
                self.eval_score["single_agreement"] = []
                self.eval_score["error"] = 0
                self.eval_score["hit"] = 0

                self.eval_score["g_win"] = 0
                self.eval_score["win"] = 0
                self.eval_score["g_lose"] = 0
                self.eval_score["lose"] = 0
                self.eval_score["g_tie"] = 0
                self.eval_score["tie"] = 0

            for rating_a, rating_b in zip(fps_action_feedback['rating_a'], fps_action_feedback['rating_b']):
                if rating_a > rating_b and judge == 0:
                    self.eval_score["hit"] += 1
                    break
                elif rating_a < rating_b and judge == 1:
                    self.eval_score["hit"] += 1
                    break
                elif rating_a == rating_b and judge == 2:
                    self.eval_score["hit"] += 1
                    break
            
            if judge == 0:
                self.eval_score["g_win"] += 1
            elif judge == 1:
                self.eval_score["g_lose"] += 1
            elif judge == 2:
                self.eval_score["g_tie"] += 1
            
            if fpm_action_feedback["model_a"] > fpm_action_feedback["model_b"]:
                self.eval_score["win"] += 1
            elif fpm_action_feedback["model_a"] < fpm_action_feedback["model_b"]:
                self.eval_score["lose"] += 1
            elif fpm_action_feedback["model_a"] == fpm_action_feedback["model_b"]:
                self.eval_score["tie"] += 1
                # pdb.set_trace()

            if judge == 2:
                return

            if (
                fpm_action_feedback["model_a"] > fpm_action_feedback["model_b"]
                and judge == 0
            ):
                self.eval_score["single_agreement"].append(1)
            elif (
                fpm_action_feedback["model_a"] < fpm_action_feedback["model_b"]
                and judge == 1
            ):
                self.eval_score["single_agreement"].append(1)
            elif (
                fpm_action_feedback["model_a"] == fpm_action_feedback["model_b"]
                and judge == 2
            ):
                self.eval_score["single_agreement"].append(1)
            else:
                self.eval_score["single_agreement"].append(0)

            no_skip = False
            if (
                fpm_action_feedback["model_a"] > fpm_action_feedback["model_b"]
                and fpm_action_feedback["ex_model_a"]
                > fpm_action_feedback["ex_model_b"]
            ):
                if judge == 0:
                    no_skip = True
                    self.eval_score["agreement"].append(1)
                else:
                    self.eval_score["agreement"].append(0)
                self.eval_score["consistency"].append(1)
            elif (
                fpm_action_feedback["model_a"] < fpm_action_feedback["model_b"]
                and fpm_action_feedback["ex_model_a"]
                < fpm_action_feedback["ex_model_b"]
            ):
                if judge == 1:
                    no_skip = True
                    self.eval_score["agreement"].append(1)
                else:
                    self.eval_score["agreement"].append(0)
                self.eval_score["consistency"].append(1)
            elif (
                fpm_action_feedback["model_a"] == fpm_action_feedback["model_b"]
                and fpm_action_feedback["ex_model_a"]
                == fpm_action_feedback["ex_model_b"]
            ):
                if judge == 2:
                    no_skip = True
                    self.eval_score["agreement"].append(1)
                else:
                    self.eval_score["agreement"].append(0)
                self.eval_score["consistency"].append(1)
            else:
                if (
                    fpm_action_feedback["model_a"] == 0
                    or fpm_action_feedback["model_b"] == 0
                ):
                    self.eval_score["error"] += 1
                self.eval_score["consistency"].append(0)
                self.eval_score["agreement"].append(0)
            
            # if no_skip:
            #     query = dialogue.get_query_by_idx(0)["content"]
            #     self.eval_filter_gen[meta_info['question_id']] = {
            #         "query": query,
            #         "judge": judge,
            #     }
        elif self.task_func == "pairwise_single_eval_func":

            meta_info = dialogue.get_meta_info()
            turn = meta_info["turn"] - 1

            fpm_action_feedback = eval_event.get_memories(
                self.task_name,
                FennecPairwiseMergeAction.action_name,
                str("turn{}".format(turn)),
            )
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

            if "single_agreement" not in self.eval_score:
                self.eval_score["single_agreement"] = []
                self.eval_score["error"] = 0
                self.eval_score["single_score"] = {}

            if meta_info['category'] not in self.eval_score["single_score"]:
                self.eval_score["single_score"][meta_info['category']] = [fpm_action_feedback["model_a"] / 5]
            else:
                self.eval_score["single_score"][meta_info['category']].append(fpm_action_feedback["model_a"] / 5)

            if (
                fpm_action_feedback["model_a"] > fpm_action_feedback["model_b"]
                and judge == 0
            ):
                self.eval_score["single_agreement"].append(1)
            elif (
                fpm_action_feedback["model_a"] < fpm_action_feedback["model_b"]
                and judge == 1
            ):
                self.eval_score["single_agreement"].append(1)
            elif (
                fpm_action_feedback["model_a"] == fpm_action_feedback["model_b"]
                and judge == 2
            ):
                self.eval_score["single_agreement"].append(1)
            else:
                self.eval_score["single_agreement"].append(0)
            
            if (
                fpm_action_feedback["model_a"] <= 0
                or fpm_action_feedback["model_b"] <= 0
            ):
                self.eval_score["error"] += 1

        elif self.task_func == "pairwise_gen_func":
            meta_info = dialogue.get_meta_info()
            turn = meta_info["turn"] - 1

            if not eval_event.not_empty():
                return

            fb_action_feedback = eval_event.get_memories(
                self.task_name,
                FennecBranchAction.action_name,
                str("turn{}".format(turn)),
            )

            query = dialogue.get_query_by_idx(0)["content"]
            response_1 = dialogue.get_pairwise_response_by_idx(0, "model_a")["content"]
            response_2 = dialogue.get_pairwise_response_by_idx(0, "model_b")["content"]

            fs_action_feedback = eval_event.get_memories(
                self.task_name,
                FennecScoringAction.action_name,
                str("turn{}".format(turn)),
            )
            fps_action_feedback = eval_event.get_memories(
                self.task_name,
                FennecPairwiseSolvingAction.action_name,
                str("turn{}".format(turn)),
            )
            fps_action_feedback = eval_event.get_memories(
                self.task_name,
                FennecPairwiseSolvingAction.action_name,
                str("turn{}".format(turn)),
            )
            fc_action_feedback = eval_event.get_memories(
                self.task_name,
                FennecCorrectionAction.action_name,
                str("turn{}".format(turn)),
            )

            if "error" not in self.eval_gen:
                self.eval_gen["error"] = 0
                self.eval_gen["server_error"] = 0
                self.eval_gen["count"] = 0
                self.eval_gen["collect"] = []
                self.eval_gen["table"] = None
                self.eval_gen["aggrement"] = []

            if (
                len(fb_action_feedback["branch_list"]) == 0
                or len(fs_action_feedback["result"]) == 0
                or len(fps_action_feedback["result"]) == 0
            ):
                self.eval_gen["server_error"] += 1
                return
            
            for r in (
                fb_action_feedback["branch_list"]
                + fs_action_feedback["result"]
                + fps_action_feedback["result"]
            ):
                if r == "" or "" == "server error":
                    self.eval_gen["error"] += 1
                
            with write_lock:
                judge = meta_info["judge"][0]
                if (
                    sum(fps_action_feedback["rating_a"])
                    > sum(fps_action_feedback["rating_b"])
                    and judge == 0
                ):
                    self.eval_gen["aggrement"].append(1)
                elif (
                    sum(fps_action_feedback["rating_a"])
                    < sum(fps_action_feedback["rating_b"])
                    and judge == 1
                ):
                    self.eval_gen["aggrement"].append(1)
                elif (
                    sum(fps_action_feedback["rating_a"])
                    == sum(fps_action_feedback["rating_b"])
                    and judge == 2
                ):
                    self.eval_gen["aggrement"].append(1)
                else:
                    self.eval_gen["aggrement"].append(0)

                if (
                    0 in fps_action_feedback["rating_a"]
                    or 0 in fps_action_feedback["rating_b"]
                ):
                    self.eval_gen["error"] += 1
                    return
                
                # pdb.set_trace()
                correction_1 = ""
                correction_2 = ""
                if "result_a" in fc_action_feedback:
                    correction_1 = fc_action_feedback['result_a']
                if "result_b" in fc_action_feedback:
                    correction_2 = fc_action_feedback['result_b']
        
                new_data = pd.DataFrame(
                    {
                        "query": [query],
                        "judge": [meta_info["judge"][0]],
                        "rating_a": [fps_action_feedback['rating_a']],
                        "rating_b": [fps_action_feedback['rating_b']],
                        "response_1": [response_1],
                        "response_2": [response_2],
                        "branch": [fb_action_feedback["branch_list"]],
                        "scoring": [fs_action_feedback["result"]],
                        "solving": [fps_action_feedback["result"]],
                        "correction_1": [correction_1],
                        "correction_2": [correction_2],
                    }
                )
                table = pa.Table.from_pandas(new_data)
                if self.eval_gen["table"] is None:
                    self.eval_gen["table"] = table
                else:
                    self.eval_gen["table"] = pa.concat_tables(
                        [self.eval_gen["table"], table]
                    )

                # pdb.set_trace()

    def serialize(self):
        if len(self.eval_score):
            if self.task_func == "pairwise_eval_func":

                if "single_agreement" in self.eval_score:
                    single_agreement = self.eval_score["single_agreement"]
                    self.logger.info(
                        "Single Agreement Average Score = {} = {} / {}".format(
                            str(
                                sum(single_agreement)
                                / (len(single_agreement) - self.eval_score["error"])
                            ),
                            str(sum(single_agreement)),
                            str(len(single_agreement) - self.eval_score["error"]),
                        )
                    )

                    self.logger.info("G win {} lose {} tie {}".format(
                        self.eval_score['g_win'], self.eval_score['g_lose'], self.eval_score['g_tie']
                    ))
                    self.logger.info("win {} lose {} tie {}".format(
                        self.eval_score['win'], self.eval_score['lose'], self.eval_score['tie']
                    ))

                if "agreement" in self.eval_score:
                    agreement = self.eval_score["agreement"]
                    consistency = self.eval_score["consistency"]
                    self.logger.info("Hit = {}".format(self.eval_score["hit"]))
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
                    self.logger.info(
                        "Consistency Average Score = {} = {} / {}".format(
                            str(
                                sum(consistency)
                                / (len(consistency) - self.eval_score["error"])
                            ),
                            str(sum(consistency)),
                            str(len(consistency) - self.eval_score["error"]),
                        )
                    )
                
                # print(len(self.eval_filter_gen))
                # json.dump(self.eval_filter_gen, open(self.train_filter_parquet_file, 'w'))
                # json.dump(self.correction, open("correction_llama2_7bchat_test.json", 'w'))
            elif self.task_func == "pairwise_single_eval_func":
                self.logger.info("Error = {}".format(self.eval_score["error"]))
                if "single_agreement" in self.eval_score:
                    single_agreement = self.eval_score["single_agreement"]
                    self.logger.info(
                        "Single Agreement Average Score = {} = {} / {}".format(
                            str(
                                sum(single_agreement)
                                / (len(single_agreement) - self.eval_score["error"])
                            ),
                            str(sum(single_agreement)),
                            str(len(single_agreement) - self.eval_score["error"]),
                        )
                    )

                    for key, item in self.eval_score["single_score"].items():
                        self.logger.info("{} score = {}".format(key, sum(item)/len(item)))    
        if len(self.eval_gen):
            if self.task_func == "pairwise_gen_func":
                server_error = self.eval_gen["server_error"]
                self.logger.info("Server Error = {}".format(server_error))
                error = self.eval_gen["error"]
                self.logger.info("Error = {}".format(error))
                count = self.eval_gen["count"]
                self.logger.info("Count = {}".format(count))
                
                self.logger.info("Table size = {}".format(len(self.eval_gen["table"])))
                pq.write_table(self.eval_gen["table"], self.train_parquet_file)

                aggrement = self.eval_gen["aggrement"]
                self.logger.info(
                    "Aggreement Average Score = {} = {} / {}".format(
                        str(sum(aggrement) / len(aggrement)),
                        str(sum(aggrement)),
                        str(len(aggrement)),
                    )
                )
