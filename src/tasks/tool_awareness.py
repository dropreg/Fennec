import pdb
from .registry import auto_register
from .meta import Task
from actions.tool_utils import ScenarioJudgeAction, GenerateDemonAction


@auto_register("tool_awareness_eval")
class ToolAwarenessEval(Task):
    task_name = "tool_awareness_eval"

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

            sj_action_feedback_dict = self.run_action(
                ScenarioJudgeAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "example": None,
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                ScenarioJudgeAction.action_name,
                sj_action_feedback_dict,
            )
            self.logger.info(
                "Scenario Judge Execute {} with Turn {}".format(self.task_name, turn_idx)
            )
        elif self.task_func == "demon_eval_func":
            turn_idx = meta_info["turn"] - 1

            gd_action_feedback_dict = self.run_action(
                GenerateDemonAction.action_name,
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
                GenerateDemonAction.action_name,
                gd_action_feedback_dict,
            )
            
            sj_action_feedback_dict = self.run_action(
                ScenarioJudgeAction.action_name,
                session_id,
                turn_idx=str("turn{}".format(turn_idx)),
                action_input={
                    "dialogue": dialogue,
                    "example": gd_action_feedback_dict[str("turn{}".format(turn_idx))]["result"],
                    "server": server,
                    "eval_model": eval_model,
                },
            )
            eval_event.update_memory(
                self.task_name,
                ScenarioJudgeAction.action_name,
                sj_action_feedback_dict,
            )
            self.logger.info(
                "Scenario Judge Execute {} with Turn {}".format(self.task_name, turn_idx)
            )
        else:
            raise Exception("Not Support Task Function {}".format(self.task_func))
