import pdb
from data.eval_event import EvalEvent
from .registry import auto_register
from .meta import Task
from actions.bsm import (
    BSMBranchAction,
    BSMPairwiseSolvingAction,
    BSMPairwiseMergeAction,
    BSMSingleBranchAction,
    BSMSingleSolvingAction,
    BSMSingleMergeAction,
)


@auto_register("bsm_eval")
class BranchSolveMergeEval(Task):
    task_name = "bsm_eval"

    def __init__(self, config, task_func, db_server, llm_server) -> None:
        super().__init__(config, task_func, db_server, llm_server)

        self.logger.info("Init Task {}".format(self.task_name))

    def annotation(self):
        pass

    def generation(self):
        pass

    def evaluation(self, eval_event: EvalEvent, server, eval_model):
        dialogue = eval_event.get_dialogue()
        session_id = dialogue.get_session_id()
        meta_info = dialogue.get_meta_info()

        if self.task_func == "pairwise_bsm_eval_func":
            turn_idx = meta_info["turn"] - 1
            bsmb_action_feedback = self.run_action(
                BSMBranchAction.action_name,
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
                BSMBranchAction.action_name,
                bsmb_action_feedback,
            )
            self.logger.info(
                "BranchSolveMergeEval Execute {} with Turn {}".format(
                    self.task_name, turn_idx
                )
            )

            bsmps_action_feedback = self.run_action(
                BSMPairwiseSolvingAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "branch_list": bsmb_action_feedback[str("turn{}".format(turn_idx))][
                        "branch_list"
                    ],
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                BSMPairwiseSolvingAction.action_name,
                bsmps_action_feedback,
            )
            self.logger.info(
                "BranchSolveMergeEval Execute {} with Turn {}".format(
                    self.task_name, turn_idx
                )
            )
            bsmpm_action_feedback = self.run_action(
                BSMPairwiseMergeAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "solving_score": bsmps_action_feedback[
                        str("turn{}".format(turn_idx))
                    ]["solving_score"],
                    "ex_solving_score": bsmps_action_feedback[
                        str("turn{}".format(turn_idx))
                    ]["ex_solving_score"],
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                BSMPairwiseMergeAction.action_name,
                bsmpm_action_feedback,
            )
            self.logger.info(
                "BranchSolveMergeEval Execute {} with Turn {}".format(
                    self.task_name, turn_idx
                )
            )
        elif self.task_func == "single_bsm_eval_func":
            turn_idx = meta_info["turn"] - 1
            bsmsb_action_feedback = self.run_action(
                BSMSingleBranchAction.action_name,
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
                BSMSingleBranchAction.action_name,
                bsmsb_action_feedback,
            )
            self.logger.info(
                "BranchSolveMergeEval Execute {} with Turn {}".format(
                    self.task_name, turn_idx
                )
            )

            bsmpss_action_feedback = self.run_action(
                BSMSingleSolvingAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "branch_list": bsmsb_action_feedback[
                        str("turn{}".format(turn_idx))
                    ]["branch_list"],
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                BSMSingleSolvingAction.action_name,
                bsmpss_action_feedback,
            )
            self.logger.info(
                "BranchSolveMergeEval Execute {} with Turn {}".format(
                    self.task_name, turn_idx
                )
            )

            bsmsm_action_feedback = self.run_action(
                BSMSingleMergeAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "solving_score": bsmpss_action_feedback[
                        str("turn{}".format(turn_idx))
                    ]["solving_score"],
                    "solving_score2": bsmpss_action_feedback[
                        str("turn{}".format(turn_idx))
                    ]["solving_score2"],
                    "ex_solving_score": bsmpss_action_feedback[
                        str("turn{}".format(turn_idx))
                    ]["ex_solving_score"],
                    "ex_solving_score2": bsmpss_action_feedback[
                        str("turn{}".format(turn_idx))
                    ]["ex_solving_score2"],
                    "server": server,
                    "eval_model": eval_model,
                },
                force=True,
            )
            eval_event.update_memory(
                self.task_name,
                BSMSingleMergeAction.action_name,
                bsmsm_action_feedback,
            )
            self.logger.info(
                "BranchSolveMergeEval Execute {} with Turn {}".format(
                    self.task_name, turn_idx
                )
            )
        else:
            raise Exception("Not Support Task Function {}".format(self.task_func))
