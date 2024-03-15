from utils import common_utils
from .dialogue_config import DialogueConfig
from .server_config import ServerConfig
from .task_config import TaskConfig
import yaml
import pdb


class AutoEvalConfig(object):
    def __init__(self, config_file):
        self.logger = common_utils.get_loguru()

        with open(config_file, "r") as stream:
            self.config = yaml.safe_load(stream)

        self.load_dialogue_config()
        self.load_task_config()
        self.load_server_config()

        self.load_eval_config()
        self.logger.info("Auto Eval Config Loaded.")

    def load_dialogue_config(self):
        self.logger.info("Load dialogue config ...")
        self.dialogue_config = DialogueConfig(self.config["dialogue_conf"])

        self.dialogue = self.config["dialogue"]
        self.dialogue_dataset = self.config["dialogue_dataset"]
        self.dialogue_db = self.dialogue_config.get_bench_db(
            self.dialogue, self.dialogue_dataset
        )
        self.logger.info("Current dialogue -> {}".format(self.dialogue))
        self.logger.info("Current dialogue dataset -> {}".format(self.dialogue_dataset))
        self.logger.info("Current dialogue db file -> {}".format(self.dialogue_db))

    def get_dialogue_db(self):
        return self.dialogue_db

    def load_task_config(self):
        self.logger.info("Load task config ...")
        self.task_config = TaskConfig(self.config["task_conf"])
        self.task = self.config["task"]
        self.task_func = self.config["task_func"]
        self.logger.info(
            "Current task -> {} func -> {}".format(
                self.config["task_conf"], self.task_func
            )
        )

    def get_task_config(self):
        return self.task_config

    def get_task(self):
        return self.task

    def get_task_func(self):
        return self.task_func

    def load_server_config(self):
        self.logger.info("Load Server config ...")
        self.server_config = ServerConfig(self.config["server_conf"])
        self.server = self.config["server"]
        self.model_id = self.config["model_id"]
        self.logger.info(
            "Current server -> {} id -> {}".format(self.server, self.model_id)
        )

    def get_server_config(self):
        return self.server_config

    def get_server(self):
        return self.server

    def get_eval_model(self):
        return self.model_id

    def load_eval_config(self):
        self.logger.info("load eval config ...")
        self.eval_func = self.config["eval_func"]
        self.eval_db = self.config["eval_db_file"].format(
            dialogue=self.dialogue,
            dialogue_dataset=self.dialogue_dataset,
            model_id=self.model_id,
            task_func=self.task_func,
        )
        self.logger.info("current eval func -> {}".format(self.eval_func))
        self.logger.info("current eval db file -> {}".format(self.eval_db))

    def get_eval_db(self):
        return self.eval_db

    def get_eval_func(self):
        return self.eval_func
