from utils import common_utils
from abc import ABC
from abc import abstractmethod
from config.dialogue_config import DialogueConfig
from persistence.dialogue_handler import DialogueHandler
from db.sqlalchemy_db import DialogueAlchemyDatabase


class Bench(ABC):
    bench_name = "meta_bench"

    def __init__(self, config: DialogueConfig) -> None:
        self.logger = common_utils.get_loguru()
        self.meta_config = config
        self.config = self.meta_config.get_bench_config(self.bench_name)

    def get_zookeeper(self):
        return self.meta_config.get_zookeeper()

    def load_db_handler(self, db_file):
        db = DialogueAlchemyDatabase(db_file)
        db_handler = DialogueHandler(db)
        return db_handler

    @abstractmethod
    def prepare(self, *args, **kwargs):
        return NotImplemented
