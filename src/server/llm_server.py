from utils import common_utils
from config.server_config import ServerConfig
from .registry import ServerRegistry


class LLMServer:
    def __init__(self, server_config: ServerConfig) -> None:
        self.logger = common_utils.get_loguru()
        self.config = server_config
        self.server = None

    def lazy_load(self, eval_model, server):
        self.server_config = self.config.get_server_config(eval_model, server)
        self.server = ServerRegistry.create_instance(server, self.server_config)

    def chat_compeletion(
        self,
        eval_model,
        server,
        query,
        system=None,
        context=None,
        temperature=None,
        top_p=None,
        max_tokens=None,
    ):
        if self.server is None:
            self.lazy_load(eval_model, server)
            self.logger.info(
                "LLM Server  eval_model = {} server = {}".format(eval_model, server)
            )

        return self.server.chat_compeletion(
            query,
            system,
            context,
            temperature,
            top_p,
            max_tokens,
        )
