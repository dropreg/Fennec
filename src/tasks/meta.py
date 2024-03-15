import pdb
from actions.meta import Action
from server.llm_server import LLMServer
from utils import common_utils
from abc import ABC
from abc import abstractmethod
from persistence.task_handler import TaskHandler
from actions.registry import ActionRegistry
from config.task_config import TaskConfig


class Task(ABC):
    task_name = "meta_task"

    def __init__(
        self, config: TaskConfig, task_func, db_server, llm_server: LLMServer
    ) -> None:
        self.logger = common_utils.get_loguru()
        self.config = config

        self.handler = TaskHandler(db_server)
        self.llm_server = llm_server
        self.task_func = task_func
        self.action_space = {}
        for name, action_config in self.config.get_actions(self.task_name).items():
            action = ActionRegistry.create_instance(
                name, action_config, self.llm_server
            )
            self.action_space[name] = action

        self.handler.build_action_db(self.task_name, list(self.action_space.keys()))
        self.logger.info(
            "Meta Task {} build action db {}".format(
                self.task_name, list(self.action_space.keys())
            )
        )

    @abstractmethod
    def annotation(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def generation(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def evaluation(self, *args, **kwargs):
        return NotImplemented

    def post_process(self, action: Action, action_input: dict, action_feedback: dict):
        action_feedback["time"] = common_utils.get_current_time()
        action_feedback["eval_model"] = action_input["eval_model"]
        return action_feedback

    def run_action(self, action_name, session_id, turn_idx, action_input, force=False):
        action_feedback_dict = self.handler.load_action_feedback(
            session_id, self.task_name, action_name
        )

        if (
            force
            or action_feedback_dict is None
            or turn_idx not in action_feedback_dict
        ):
            action_feedback = self.action_space[action_name].execute(**action_input)
            self.post_process(
                self.action_space[action_name], action_input, action_feedback
            )

            if action_feedback_dict is None:
                action_feedback_dict = {turn_idx: action_feedback}
                self.handler.save_action_feedback(
                    session_id, self.task_name, action_name, action_feedback_dict
                )
            elif turn_idx not in action_feedback_dict:
                action_feedback_dict[turn_idx] = action_feedback
                self.handler.update_action_feedback(
                    session_id, self.task_name, action_name, action_feedback_dict
                )
            else:
                if force:
                    action_feedback_dict[turn_idx] = action_feedback
                    self.handler.update_action_feedback(
                        session_id, self.task_name, action_name, action_feedback_dict
                    )
                else:
                    raise ValueError("Not Support Update action_feedback Value !")
        return action_feedback_dict

    def run_action_with_ekey(
        self, action_name, session_id, turn_idx, extra_key, action_input
    ):
        action_feedback_dict = self.handler.load_action_feedback(
            session_id, self.task_name, action_name
        )

        if (
            action_feedback_dict is None
            or turn_idx not in action_feedback_dict
            or extra_key not in action_feedback_dict[turn_idx]
        ):
            action_feedback = self.action_space[action_name].execute(**action_input)
            self.post_process(
                self.action_space[action_name], action_input, action_feedback
            )

            if action_feedback_dict is None:
                action_feedback_dict = {turn_idx: {extra_key: action_feedback}}
                self.handler.save_action_feedback(
                    session_id, self.task_name, action_name, action_feedback_dict
                )
            elif turn_idx not in action_feedback_dict:
                action_feedback_dict[turn_idx] = {extra_key: action_feedback}
                self.handler.update_action_feedback(
                    session_id, self.task_name, action_name, action_feedback_dict
                )
            elif extra_key not in action_feedback_dict[turn_idx]:
                action_feedback_dict[turn_idx][extra_key] = action_feedback
                self.handler.update_action_feedback(
                    session_id, self.task_name, action_name, action_feedback_dict
                )
            else:
                raise ValueError("Not Support Update action_feedback Value !")
        return action_feedback_dict
