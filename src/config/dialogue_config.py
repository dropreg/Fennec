from utils import common_utils
import yaml


class DialogueConfig:
    def __init__(self, config_path):
        self.logger = common_utils.get_loguru()

        self.config_path = config_path
        self.load_dialogue_config()

    def load_dialogue_config(self):
        with open(self.config_path, "r") as stream:
            self.config = yaml.safe_load(stream)
        self.logger.info("Load Dialogue Config from {}".format(self.config_path))

    def get_bench_candidates(self):
        return set(self.config["bench_candidates"])

    def get_bench_config(self, bench_name):
        return self.config[bench_name]

    def get_bench_db(self, bench_name, dataset_name):
        return self.config[bench_name]["datasets"][dataset_name]["db_file"]

    def get_zookeeper(self):
        return self.config["zookeeper"]
