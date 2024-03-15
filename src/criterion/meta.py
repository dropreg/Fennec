from utils import common_utils
from abc import ABC
from abc import abstractmethod
from config.task_config import TaskConfig


class Evaluation(ABC):
    def __init__(self, config) -> None:
        self.logger = common_utils.get_loguru()
        self.config = config

    @abstractmethod
    def eval(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def serialize(self, *args, **kwargs):
        return NotImplemented
