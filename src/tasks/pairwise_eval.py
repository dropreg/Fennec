import pdb
from data.eval_event import EvalEvent
from .registry import auto_register
from .meta import Task
from actions.comparing import PairwiseComparingAction


@auto_register("pairwise_eval")
class PairwiseEval(Task):
    task_name = "pairwise_eval"

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

        if self.task_func == "pairwise_eval_func":
            turn_idx = meta_info["turn"] - 1
            pc_action_feedback = self.run_action(
                PairwiseComparingAction.action_name,
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
                PairwiseComparingAction.action_name,
                pc_action_feedback,
            )
            self.logger.info(
                "PairwiseEval Execute {} with Turn {}".format(self.task_name, turn_idx)
            )

        else:
            raise Exception("Not Support Task Function {}".format(self.task_func))
