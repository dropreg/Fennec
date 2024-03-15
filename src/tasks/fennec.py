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
)


@auto_register("fennec")
class Fennec(Task):
    task_name = "fennec"

    def __init__(self, config, task_func, db_server, llm_server) -> None:
        super().__init__(config, task_func, db_server, llm_server)

        self.logger.info("Init Task {}".format(self.task_name))

    def annotation(self):
        pass

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
                    "eval": True,
                    "server": server,
                    "eval_model": eval_model,
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
            
            fc_action_feedback = self.run_action(
                FennecCorrectionAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
                        "branch_list"
                    ],
                    "solving_list": fps_action_feedback[str("turn{}".format(turn_idx))],
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                FennecCorrectionAction.action_name,
                fc_action_feedback,
            )
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
                    "eval_model": eval_model,
                },
            )
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
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                FennecPairwiseSolvingAction.action_name,
                fps_action_feedback,
            )
            
            fc_action_feedback = self.run_action(
                FennecCorrectionAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "branch_list": fb_action_feedback[str("turn{}".format(turn_idx))][
                        "branch_list"
                    ],
                    "solving_list": fps_action_feedback[str("turn{}".format(turn_idx))],
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                FennecCorrectionAction.action_name,
                fc_action_feedback,
            )

            self.logger.info(
                "Fennec Execute {} with Turn {}".format(
                    self.task_name, turn_idx
                )
            )
        else:
            raise Exception("Not Support Task Function {}".format(self.task_func))
