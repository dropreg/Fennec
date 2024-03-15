import pdb
from .registry import auto_register
from .meta import Task
from actions.scoring import SingleScoringAction, PairwiseSingleScoringAction


@auto_register("single_eval")
class SingleEval(Task):
    task_name = "single_eval"

    def __init__(self, config, task_func, db_server, llm_server) -> None:
        super().__init__(config, task_func, db_server, llm_server)

        self.logger.info("Init Task {}".format(self.task_name))

    def annotation(self):
        pass

    def generation(self):
        pass

    def evaluation(self, eval_event, server, eval_model):
        dialogue = eval_event.get_dialogue()
        session_id = dialogue.get_session_id()
        meta_info = dialogue.get_meta_info()

        if self.task_func == "single_eval_func":
            turn_idx = meta_info["turn"] - 1
            ss_action_feedback_dict = self.run_action(
                SingleScoringAction.action_name,
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
                SingleScoringAction.action_name,
                ss_action_feedback_dict,
            )
            self.logger.info(
                "SingleEval Execute {} with Turn {}".format(self.task_name, turn_idx)
            )
        elif self.task_func == "pairwise_single_eval_func":
            turn_idx = meta_info["turn"] - 1
            pss_action_feedback_dict = self.run_action(
                PairwiseSingleScoringAction.action_name,
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
                PairwiseSingleScoringAction.action_name,
                pss_action_feedback_dict,
            )
            self.logger.info(
                "SingleEval Execute {} with Turn {}".format(self.task_name, turn_idx)
            )
        else:
            raise Exception("Not Support Task Function {}".format(self.task_func))
