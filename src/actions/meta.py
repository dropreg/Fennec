from utils import common_utils
from abc import ABC
from abc import abstractmethod
import json


class Action(ABC):
    action_name = "meta_action"

    def __init__(self, config) -> None:
        self.logger = common_utils.get_loguru()
        self.config = config

        self.load_config()

    def load_config(self):
        self.name = self.config["name"]
        self.description = self.config["description"]
        self.lang = self.config["lang"]
        self.template_json = json.load(open(self.config["template_file"]))

    def get_template_json(self):
        return self.template_json

    def get_lang(self):
        return self.lang

    @abstractmethod
    def execute(self, *args, **kwargs):
        return NotImplemented
