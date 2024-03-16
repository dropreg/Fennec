from data.dialogue import Dialogue, PairwiseDialogue
from server.llm_server import LLMServer
from .registry import auto_register
from .meta import Action
import pdb
import json
import random


@auto_register("dialogue_gen")
class DialogueGenAction(Action):
    action_name = "dialogue_gen"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)
        self.llm_server = llm_server
    
    def execute(self, **action_input):
        action_feedback = {}
        
        result = self.get_result_from_server(
            action_input["dialogue"],
            action_input["server"],
            action_input["eval_model"],
        )
        action_feedback["result"] = result
        return action_feedback

    def get_result_from_server(self, dialogue, server, eval_model):
        meta_info = dialogue.get_meta_info()
        result = self.llm_server.chat_compeletion(
            eval_model,
            server,
            query=dialogue.get_query_by_idx(0)["content"],
            system="",
            context=meta_info['context'],
        )
        if isinstance(result, str):
            if "[/INST]" in result:
                pos = result.rfind("[/INST]")
                return result[pos:].strip()
            return result.strip()
        else:
            return result[0]["generated_text"]

