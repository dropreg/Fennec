import pdb
from utils import common_utils
import yaml


class TaskConfig:
    def __init__(self, config_path):
        self.logger = common_utils.get_loguru()

        self.config_path = config_path
        self.load_task_config()

    def load_task_config(self):
        with open(self.config_path, "r") as stream:
            self.config = yaml.safe_load(stream)

        self.task_map = {}
        for task, task_config in self.config.items():
            with open(task_config["config_file"], "r") as stream:
                self.task_map[task] = yaml.safe_load(stream)
            self.logger.info(
                "load task config from path {}".format(task_config["config_file"])
            )

    def get_actions(self, task):
        return self.task_map[task]["actions"]

    def get_eval_json_file(self, task):
        return self.task_map[task]["eval_json"]

    def get_train_parquet_file(self, task):
        return self.task_map[task]["train_parquet_file"]
    
    def get_train_filter_parquet_file(self, task):
        return self.task_map[task]["train_filter_parquet_file"]

    def get_eval_source(self, task):
        return self.task_map[task]["eval_source"]

    def get_eval_target(self, task):
        return self.task_map[task]["eval_target"]
