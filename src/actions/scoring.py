import pdb
from data.dialogue import Dialogue, PairwiseDialogue
from server.llm_server import LLMServer
from .registry import auto_register
from .meta import Action
import re
import ast


@auto_register("single_scoring")
class SingleScoringAction(Action):
    action_name = "single_scoring"

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
        action_feedback["rating"] = self.rating_format(
            result, action_input["eval_model"]
        )
        return action_feedback

    def build_prompt(self, dialogue: Dialogue, eval_model: str):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        system_message = template_json["system"]
        

        if "instruction" in meta_info:
            # prompt = meta_info['instruction']
            end_pos = meta_info['instruction'].find("###Reference Answer")
            start_pos = meta_info['instruction'].find("###Score Rubrics:")
            prompt = meta_info['instruction'][:end_pos] + meta_info['instruction'][start_pos:]

        else:
            prompt = template_json["instruction"]
            if meta_info["turn"] == 1:
                prompt += template_json["query"].format(
                    query=dialogue.get_query_by_idx(0)["content"]
                )
                prompt += template_json["response"].format(
                    response=dialogue.get_response_by_idx(0)["content"]
                )
            else:
                prompt += template_json["history"].format(
                    query=dialogue.get_query_by_idx(0)["content"],
                    response=dialogue.get_response_by_idx(0)["content"],
                )
                prompt += template_json["query"].format(
                    query=dialogue.get_query_by_idx(1)["content"]
                )
                prompt += template_json["response"].format(
                    response=dialogue.get_response_by_idx(1)["content"]
                )
            if "reference" in template_json and len(meta_info["reference"]):
                prompt += template_json["reference"].format(
                    reference=meta_info["reference"][0]
                )
            if "score" in template_json and meta_info["score"]:
                prompt += template_json["score"].format(score=meta_info["score"])
            prompt += template_json["format"]

        return system_message, prompt

    def rating_format(self, result, eval_model):
        if "prometheus" == eval_model:
            pos = result.rfind("###Feedback:")
            x = result[pos:].split("[RESULT]")
            if len(x) >= 2:
                rating = int(x[1].strip())
            else:
                rating = -1
        else:
            score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
            score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
            match = re.search(score_pattern, result)
            if not match:
                match = re.search(score_pattern_backup, result)

            if match:
                rating = ast.literal_eval(match.groups()[0])
            else:
                rating = -1
        return rating

    def get_result_from_server(self, dialogue, server, eval_model):
        system_message, prompt = self.build_prompt(dialogue, eval_model)
        result = self.llm_server.chat_compeletion(
            eval_model,
            server,
            query=prompt,
            system=system_message,
            context=None,
        )
        if isinstance(result, str):
            return result
        else:
            return result[0]["generated_text"]


@auto_register("pairwise_single_scoring")
class PairwiseSingleScoringAction(Action):
    action_name = "pairwise_single_scoring"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {"result": {}, "rating": {}}
        for key in action_input["dialogue"].get_pairwise_key():
            result = self.get_result_from_server(
                action_input["dialogue"],
                action_input["server"],
                action_input["eval_model"],
                key,
            )
            action_feedback["result"][key] = result
            action_feedback["rating"][key] = self.rating_format(
                result, action_input["eval_model"]
            )
        return action_feedback

    def build_prompt(
        self, dialogue: PairwiseDialogue, eval_model: str, pairwise_key: str
    ):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        system_message = template_json["system"]
        prompt = template_json["instruction"]
        
        if meta_info["turn"] == 1:
            prompt += template_json["query"].format(
                query=dialogue.get_query_by_idx(0)["content"]
            )
            prompt += template_json["response"].format(
                response=dialogue.get_pairwise_response_by_idx(0, pairwise_key)[
                    "content"
                ]
            )
        else:
            prompt += template_json["history"].format(
                query=dialogue.get_query_by_idx(0)["content"],
                response=dialogue.get_pairwise_response_by_idx(0, pairwise_key)[
                    "content"
                ],
            )
            prompt += template_json["query"].format(
                query=dialogue.get_query_by_idx(1)["content"]
            )
            prompt += template_json["response"].format(
                response=dialogue.get_pairwise_response_by_idx(1, pairwise_key)[
                    "content"
                ]
            )
        if "reference" in template_json and len(meta_info["reference"]):
            prompt += template_json["reference"].format(
                reference=meta_info["reference"][0]
            )
        # if "score" in template_json:
        #     if meta_info["score"]:
        #         prompt += template_json["score"].format(score=meta_info["score"])
        #     else:
        #         prompt += template_json["score_candi"]
        #         prompt += template_json["format"]
        # else:
        prompt += template_json["score_candi"]
        prompt += template_json["format"]
        return system_message, prompt

    def rating_format(self, result, eval_model):
        if "prometheus" == eval_model:
            pos = result.rfind("###Feedback:")
            x = result[pos:].split("[RESULT]")
            if len(x) >= 2:
                rating = float(x[-1].strip())
            else:
                rating = -1
        else:
            score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
            score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
            match = re.search(score_pattern, result)
            if not match:
                match = re.search(score_pattern_backup, result)

            if match:
                rating = ast.literal_eval(match.groups()[0])
            else:
                rating = -1
        return rating

    def get_result_from_server(self, dialogue, server, eval_model, pairwise_key):
        system_message, prompt = self.build_prompt(dialogue, eval_model, pairwise_key)
        result = self.llm_server.chat_compeletion(
            eval_model,
            server,
            query=prompt,
            system=system_message,
            context=None,
        )
        if isinstance(result, str):
            return result
        else:
            return result[0]["generated_text"]
