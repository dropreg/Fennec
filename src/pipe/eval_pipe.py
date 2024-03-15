from tasks.registry import TaskRegistry
from criterion.registry import EvalRegistry
from config.auto_eval_config import AutoEvalConfig
from db.sqlalchemy_db import TaskAlchemyDatabase
from server.llm_server import LLMServer
from utils import common_utils
from data.eval_event import EvalEvent
import time


class EvalPipe:
    def __init__(self, config: AutoEvalConfig) -> None:
        self.logger = common_utils.get_loguru()

        self.config = config
        self.db_server = TaskAlchemyDatabase(self.config.get_eval_db())
        self.llm_server = LLMServer(self.config.get_server_config())
        self.eval_func = self.config.get_eval_func()

        task_name = self.config.get_task()
        task_config = self.config.get_task_config()
        task_func = self.config.get_task_func()
        self.task = TaskRegistry.create_instance(
            task_name, task_config, task_func, self.db_server, self.llm_server
        )
        self.logger.info("Init Eval Task name = {}".format(task_name))

        if EvalRegistry.need_eval(task_name):
            self.criterion = EvalRegistry.create_instance(
                task_name,
                task_config,
                task_func,
            )
            self.logger.info("Init Criterion for {}".format(task_name))
        else:
            self.criterion = None
            self.logger.info("Not Need Criterion for {}".format(task_name))

    def run(self, dialogue_idx, dialogue, queue_id=0):
        self.logger.info("Eval Pipe Handle Dialogue Index {}...".format(dialogue_idx))
        start_time = time.time()

        eval_event = EvalEvent(dialogue, queue_id)
        session_id = dialogue.get_session_id()[:4]

        self.logger.info(
            "Start Execute Task {} Session ID {} From Queue {}".format(
                self.task.task_name, session_id, queue_id
            )
        )

        server = self.config.get_server()
        eval_model = self.config.get_eval_model()

        if self.eval_func == "annotation":
            self.task.annotation(eval_event, server, eval_model)
        elif self.eval_func == "generation":
            self.task.generation(eval_event, server, eval_model)
        elif self.eval_func == "evaluation":
            self.task.evaluation(eval_event, server, eval_model)
        else:
            raise ValueError("Not Support Execute Function = {}".format(self.eval_func))

        if self.criterion:
            self.criterion.eval(eval_event)

        self.logger.info(
            "Eval Pipe End with Cost Time = {}/s".format(
                round(time.time() - start_time, 3)
            )
        )

    def serilaize_eval(self):
        if self.criterion:
            self.criterion.serialize()
