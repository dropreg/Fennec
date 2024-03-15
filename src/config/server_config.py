from utils import common_utils
import yaml
import pdb


class ServerConfig(object):
    def __init__(self, config_file):
        self.logger = common_utils.get_loguru()

        with open(config_file, "r") as stream:
            self.config = yaml.safe_load(stream)

        self.load_server_config()
        self.logger.info("Server config loaded.")

    def load_server_config(self):
        self.logger.info("Load server config ...")

        self.server_map = {}
        for _, server in self.config.items():
            self.server_map[server["eval_model"]] = server["servers"]

    def get_server_config(self, eval_model, server):
        if server in self.server_map:
            return self.server_map[server][eval_model]
        else:
            # print(self.server_map)
            raise Exception("Not Support Eval Model {}".format(eval_model))
