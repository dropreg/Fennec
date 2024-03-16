import pdb
from .registry import auto_register
from .meta import Task
from actions.dialogue_gen import DialogueGenAction


@auto_register("dialogue_gen")
class DialogueGenEval(Task):
    task_name = "dialogue_gen"
    
    def __init__(self, config, task_func, db_server, llm_server) -> None:
        super().__init__(config, task_func, db_server, llm_server)

        self.logger.info("Init Task {}".format(self.task_name))

    def annotation(self):
        pass

    def evaluation(self):
        pass

    def generation(self, eval_event, server, eval_model):
        dialogue = eval_event.get_dialogue()
        session_id = dialogue.get_session_id()
        meta_info = dialogue.get_meta_info()
        
        if self.task_func == "gen_func":
            turn_idx = meta_info["turn"] - 1

            dg_action_feedback_dict = self.run_action(
                DialogueGenAction.action_name,
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
                DialogueGenAction.action_name,
                dg_action_feedback_dict,
            )
            self.logger.info(
                "DialogueGen Action Execute {} with Turn {}".format(self.task_name, turn_idx)
            )
        else:
            raise Exception("Not Support Task Function {}".format(self.task_func))
