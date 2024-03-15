from data.dialogue import Dialogue, PairwiseDialogue
from server.llm_server import LLMServer
from .registry import auto_register
from .meta import Action
import pdb
import json
import random


@auto_register("scenario_judge")
class ScenarioJudgeAction(Action):
    action_name = "scenario_judge"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)
        self.llm_server = llm_server
    
    def execute(self, **action_input):
        action_feedback = {}
        result = self.get_result_from_server(
            action_input["dialogue"],
            action_input["example"],
            action_input["server"],
            action_input["eval_model"],
        )
        action_feedback["result"] = result
        return action_feedback
        
    def build_prompt(self, dialogue, example, eval_model):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()]
        instruction = template_json["common"]['raw']
        if example is None:
            example = meta_info['example']
        if meta_info["turn"] == 1:
            prompt = instruction.format(
                query=dialogue.get_query_by_idx(0)["content"], example=example
            )
        else:
            raise Exception("Not Support")
        return None, prompt

    def get_result_from_server(self, dialogue, example, server, eval_model):
        system_message, prompt = self.build_prompt(dialogue, example, eval_model)
        result = self.llm_server.chat_compeletion(
            eval_model,
            server,
            query=prompt,
            system=system_message,
            context=None,
        )
        if isinstance(result, str):
            if "[/INST]" in result:
                pos = result.rfind("[/INST]")
                return result[pos:].strip()
            return result.strip()
        else:
            return result[0]["generated_text"]


@auto_register("generate_demon")
class GenerateDemonAction(Action):
    action_name = "generate_demon"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {}

        dialogue = action_input["dialogue"]
        meta_info = dialogue.get_meta_info()

        result_list = []
        
        for i in range(3):
            
            query = dialogue.get_query_by_idx(0)["content"]
            result = self.get_result_from_server(
                dialogue,
                query.strip(),
                action_input["server"],
                action_input["eval_model"],
            )
            result_list.append("\nThought {}: ".format(i) + result)

        action_feedback["result"] = "\n".join(result_list)
        return action_feedback
        
    def build_prompt(self, dialogue, query, eval_model):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()]
        instruction = template_json["common"]['gen_demon']
        # example = meta_info['example']
        prompt = instruction.format(
            query=query
        )
        return None, prompt

    def get_result_from_server(self, dialogue, query, server, eval_model):
        system_message, prompt = self.build_prompt(dialogue, query, eval_model)
        result = self.llm_server.chat_compeletion(
            eval_model,
            server,
            query=prompt,
            system=system_message,
            context=None,
        )
        if isinstance(result, str):
            if "[/INST]" in result:
                pos = result.rfind("[/INST]")
                return result[pos:].strip().replace("[/INST]", "")
            return result.strip()
        else:
            return result[0]["generated_text"]