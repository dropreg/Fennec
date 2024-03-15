from utils import common_utils
from abc import ABC
from abc import abstractmethod


class Server(ABC):
    eval_model = "meta_server"

    def __init__(self, config) -> None:
        self.logger = common_utils.get_loguru()

        self.config = config

    @abstractmethod
    def message_warpper(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def chat_compeletion(self, *args, **kwargs):
        return NotImplemented
