import pdb
from data.eval_event import EvalEvent
from .registry import auto_register
from .meta import Task
from actions.fennec import (
    FennecBranchAction,
    FennecScoringAction,
    FennecPairwiseSolvingAction,
    FennecPairwiseMergeAction,
    FennecPairwiseSingleSolvingAction,
    FennecCorrectionAction,
    FennecBranchSelectionAction,
    FennecBranchQuickSortAction,
)
import re

@auto_register("fennec")
class Fennec(Task):
    task_name = "fennec"

    def __init__(self, config, task_func, db_server, llm_server) -> None:
        super().__init__(config, task_func, db_server, llm_server)

        self.logger.info("Init Task {}".format(self.task_name))

    def annotation(self):
        pass

    def starts_with_numbered_list(self, text):
        pattern = r"^\d+\.\s"
        return bool(re.match(pattern, text))
        
    def criteria_extraction(self, criteria):
        criteria_list = []
        for line in criteria.split("\n"):
            if self.starts_with_numbered_list(line):
                criteria_list.append(line)
        if criteria_list == []:
            criteria_list = [c for c in criteria.split("\n") if c.replace('\n', '').strip() != ""]
        return criteria_list
    
    def extract_branch_list(self, result):
        if "\n<|assistant|>\n" in result:
            result = result.replace("\n<|assistant|>\n", "")
        elif "[/INST]" in result:
            result = result.split("[/INST]")[-1]
        # branch_list = []
        # for branch in result.split("\n"):
        #     if branch:
        #         branch_list.append(branch)
        
        branch_list = [b.replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "").replace("5.", "")
                       .replace("6.", "").replace("7.", "").replace("8.", "").replace("9.", "").replace("10.", "") for b in self.criteria_extraction(result)]

        return branch_list
    
    def evaluation(self, eval_event: EvalEvent, server, eval_model):
        dialogue = eval_event.get_dialogue()
        session_id = dialogue.get_session_id()
        meta_info = dialogue.get_meta_info()

        if self.task_func == "pairwise_eval_func":
            turn_idx = meta_info["turn"] - 1

            fb_action_feedback = self.run_action(
                FennecBranchAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "server": server,
                    "eval_model": eval_model,
                    "context": meta_info["context"] if "context" in meta_info.keys() else None,
                },
            )
            eval_event.update_memory(
                self.task_name,
                FennecBranchAction.action_name,
                fb_action_feedback,
            )

            fs_action_feedback = self.run_action(
                FennecScoringAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
                        "branch_list"
                    ],
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                FennecScoringAction.action_name,
                fs_action_feedback,
            )
            fps_action_feedback = self.run_action(
                FennecPairwiseSolvingAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
                        "branch_list"
                    ],
                    "scoring_list": fs_action_feedback[str("turn{}".format(turn_idx))][
                        "result"
                    ],
                    # "scoring_list": ["", "", "", "", ""],
                    "eval": True,
                    "server": server,
                    "eval_model": eval_model,
                    "context": meta_info["context"] if "context" in meta_info.keys() else None
                },
            )
            eval_event.update_memory(
                self.task_name,
                FennecPairwiseSolvingAction.action_name,
                fps_action_feedback,
            )

            fpm_action_feedback = self.run_action(
                FennecPairwiseMergeAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
                        "branch_list"
                    ],
                    "solving_list": fps_action_feedback[str("turn{}".format(turn_idx))][
                        "result"
                    ],
                    "result": fps_action_feedback[str("turn{}".format(turn_idx))],
                    "judge": [meta_info["judge"][0]],
                    "server": server,
                    "eval_model": eval_model,
                },
                force=True,
            )
            eval_event.update_memory(
                self.task_name,
                FennecPairwiseMergeAction.action_name,
                fpm_action_feedback,
            )
            # fbs_action_feedback = self.run_action(
            #     FennecBranchSelectionAction.action_name,
            #     session_id,
            #     turn_idx=str("turn{}".format(turn_idx)),
            #     action_input={
            #         "dialogue": dialogue,
            #         "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
            #             "branch_list"
            #         ],
            #         "scoring_list": fs_action_feedback[str("turn{}".format(turn_idx))][
            #             "result"
            #         ],
            #         "solving_list": fps_action_feedback[str("turn{}".format(turn_idx))],
            #         "server": server,
            #         "eval_model": eval_model,
            #     },
            #     # force=True
            # )
            # eval_event.update_memory(
            #     self.task_name,
            #     FennecBranchSelectionAction.action_name,
            #     fbs_action_feedback,
            # )
            
            # fbqs_action_feedback = self.run_action(
            #     FennecBranchQuickSortAction.action_name,
            #     session_id,
            #     turn_idx=str("turn{}".format(turn_idx)),
            #     action_input={
            #         "dialogue": dialogue,
            #         "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
            #             "branch_list"
            #         ],
            #         "scoring_list": fs_action_feedback[str("turn{}".format(turn_idx))][
            #             "result"
            #         ],
            #         "solving_list": fps_action_feedback[str("turn{}".format(turn_idx))],
            #         "server": server,
            #         "eval_model": eval_model,
            #     },
            #     # force=True
            # )
            # eval_event.update_memory(
            #     self.task_name,
            #     FennecBranchQuickSortAction.action_name,
            #     fbqs_action_feedback,
            # )
            
            
            # import pdb;pdb.set_trace()
            # eval_event.update_memory(
            #     self.task_name,
            #     FennecCorrectionAction.action_name,
            #     fc_action_feedback,
            # )
            # fc_action_feedback = self.run_action(
            #     FennecCorrectionAction.action_name,
            #     session_id,
            #     turn_idx=str("turn{}".format(turn_idx)),
            #     action_input={
            #         "dialogue": dialogue,
            #         "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
            #             "branch_list"
            #         ],
            #         "solving_list": fps_action_feedback[str("turn{}".format(turn_idx))],
            #         "server": server,
            #         "eval_model": eval_model,
            #     },
            # )
            # eval_event.update_memory(
            #     self.task_name,
            #     FennecCorrectionAction.action_name,
            #     fc_action_feedback,
            # )
            self.logger.info(
                "Fennec Execute {} with Turn {}".format(self.task_name, turn_idx)
            )

        elif self.task_func == "pairwise_single_eval_func":

            turn_idx = meta_info["turn"] - 1
            fb_action_feedback = self.run_action(
                FennecBranchAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                FennecBranchAction.action_name,
                fb_action_feedback,
            )

            fs_action_feedback = self.run_action(
                FennecScoringAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
                        "branch_list"
                    ],
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                FennecScoringAction.action_name,
                fs_action_feedback,
            )
            fpss_action_feedback = self.run_action(
                FennecPairwiseSingleSolvingAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
                        "branch_list"
                    ],
                    "scoring_list": fs_action_feedback[str("turn{}".format(turn_idx))][
                        "result"
                    ],
                    "eval": True,
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                FennecPairwiseSingleSolvingAction.action_name,
                fpss_action_feedback,
            )

            fpm_action_feedback = self.run_action(
                FennecPairwiseMergeAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
                        "branch_list"
                    ],
                    "result": fpss_action_feedback[str("turn{}".format(turn_idx))],
                    "server": server,
                    "eval_model": eval_model,
                },
                force=True,
            )
            eval_event.update_memory(
                self.task_name,
                FennecPairwiseMergeAction.action_name,
                fpm_action_feedback,
            )
            self.logger.info(
                "Fennec Execute {} with Turn {}".format(self.task_name, turn_idx)
            )
        else:
            raise Exception("Not Support Task Function {}".format(self.task_func))

    def generation(self, eval_event: EvalEvent, server, eval_model):
        dialogue = eval_event.get_dialogue()
        session_id = dialogue.get_session_id()
        meta_info = dialogue.get_meta_info()

        if self.task_func == "pairwise_gen_func":
            turn_idx = meta_info["turn"] - 1
            fb_action_feedback = self.run_action(
                FennecBranchAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "server": server,
                    "context": meta_info["context"] if "context" in meta_info.keys() else None,
                    "eval_model": eval_model,
                },
            )
            
            # fb_action_feedback[str("turn{}".format(turn_idx))]['branch_list'] = self.extract_branch_list(fb_action_feedback[str("turn{}".format(turn_idx))]["result"])
            
            eval_event.update_memory(
                self.task_name,
                FennecBranchAction.action_name,
                fb_action_feedback,
            )

            self.logger.info(
                "Fennec Execute {} with Turn {}".format(self.task_name, turn_idx)
            )

            fs_action_feedback = self.run_action(
                FennecScoringAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
                        "branch_list"
                    ],
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                FennecScoringAction.action_name,
                fs_action_feedback,
            )
            self.logger.info(
                "BranchSolveMergeEval Execute {} with Turn {}".format(
                    self.task_name, turn_idx
                )
            )

            fps_action_feedback = self.run_action(
                FennecPairwiseSolvingAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
                        "branch_list"
                    ],
                    "scoring_list": fs_action_feedback[str("turn{}".format(turn_idx))][
                        "result"
                    ],
                    # "scoring_list": ["", "", "", "", ""],
                    "server": server,
                    "eval_model": eval_model,
                    "context": meta_info["context"] if "context" in meta_info.keys() else None
                },
                
            )
            eval_event.update_memory(
                self.task_name,
                FennecPairwiseSolvingAction.action_name,
                fps_action_feedback,
            )
            
            # fc_action_feedback = self.run_action(
            #     FennecCorrectionAction.action_name,
            #     session_id,
            #     turn_idx=str("turn{}".format(turn_idx)),
            #     action_input={
            #         "dialogue": dialogue,
            #         "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
            #             "branch_list"
            #         ],
            #         "solving_list": fps_action_feedback[str("turn{}".format(turn_idx))],
            #         "server": server,
            #         "eval_model": eval_model,
            #     },
            # )
            # eval_event.update_memory(
            #     self.task_name,
            #     FennecCorrectionAction.action_name,
            #     fc_action_feedback,
            # )
            # import pdb;pdb.set_trace()
            self.logger.info(
                "Fennec Execute {} with Turn {}".format(
                    self.task_name, turn_idx
                )
            )
        else:
            raise Exception("Not Support Task Function {}".format(self.task_func))
