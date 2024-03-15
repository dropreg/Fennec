import pdb
from urllib import response

from sympy import EX
from data.dialogue import PairwiseDialogue
from server.llm_server import LLMServer
from .registry import auto_register
from .meta import Action
import re
import ast


@auto_register("fennec_branch")
class FennecBranchAction(Action):
    action_name = "fennec_branch"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server
        self.max_branch = 5

    def execute(self, **action_input):
        action_feedback = {}
        result = self.get_result_from_server(
            action_input["dialogue"],
            action_input["server"],
            action_input["eval_model"],
        )
        action_feedback["result"] = result
        action_feedback["branch_list"] = self.extract_branch_list(result)
        return action_feedback

    def extract_branch_list(self, result):
        if "\n<|assistant|>\n" in result:
            result = result.replace("\n<|assistant|>\n", "")
        branch_list = []
        for branch in result.split("\n"):
            if branch:
                branch_list.append(branch)
        
        branch_list = [b.replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "").replace("5.", "") for b in branch_list]

        return branch_list[: self.max_branch]

    def build_prompt(self, dialogue, eval_model):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        system_message = template_json["branch_message"]

        if meta_info["turn"] == 1:
            prompt = template_json["branch"].format(
                query=dialogue.get_query_by_idx(0)["content"]
            )
        else:
            raise Exception("Not Support")
        return system_message, prompt

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
            raise Exception("error")


@auto_register("fennec_scoring")
class FennecScoringAction(Action):
    action_name = "fennec_scoring"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {"result": []}

        for branch in action_input["branch_list"]:
            result = self.get_result_from_server(
                action_input["dialogue"],
                action_input["server"],
                action_input["eval_model"],
                branch,
            )
            result = result.replace("[Scoring Guideline]:", "")
            action_feedback["result"].append(result)
        return action_feedback

    def build_prompt(self, dialogue: PairwiseDialogue, eval_model: str, branch: str):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        system_message = template_json["scoring_message"]
        
        if meta_info["turn"] == 1:
            prompt = template_json["scoring"].format(
                query=dialogue.get_query_by_idx(0)["content"], criteria=branch
            )
        else:
            raise Exception("Not Support")
        return system_message, prompt

    def get_result_from_server(self, dialogue, server, eval_model, branch):
        system_message, prompt = self.build_prompt(dialogue, eval_model, branch)
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


@auto_register("fennec_pairwise_solving")
class FennecPairwiseSolvingAction(Action):
    action_name = "fennec_pairwise_solving"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {
            "result": [],
            "rating_a": [],
            "rating_b": [],
            "ex_result": [],
            "ex_rating_a": [],
            "ex_rating_b": [],
        }

        for branch, scoring in zip(
            action_input["branch_list"], action_input["scoring_list"]
        ):
            branch = branch.strip()
            scoring = scoring.strip()

            result = self.get_result_from_server(
                action_input["dialogue"],
                action_input["server"],
                action_input["eval_model"],
                branch,
                scoring,
            )
            action_feedback["result"].append(result)
            rating_a, rating_b = self.rating_format(result, action_input["eval_model"])
            action_feedback["rating_a"].append(rating_a)
            action_feedback["rating_b"].append(rating_b)

            if "eval" in action_input and action_input["eval"]:
                ex_result = self.get_result_from_server(
                    action_input["dialogue"],
                    action_input["server"],
                    action_input["eval_model"],
                    branch,
                    scoring,
                    exchange=True,
                )
                action_feedback["ex_result"].append(ex_result)
                ex_rating_a, ex_rating_b = self.rating_format(
                    ex_result, action_input["eval_model"]
                )
                action_feedback["ex_rating_a"].append(ex_rating_a)
                action_feedback["ex_rating_b"].append(ex_rating_b)

        return action_feedback

    def build_prompt(
        self,
        dialogue: PairwiseDialogue,
        eval_model: str,
        branch: str,
        scoring: str,
        exchange: bool,
    ):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        system_message = template_json["system"]
        prompt = template_json["sovling"]
        ex_prompt = template_json["ex_sovling"]
        
        if meta_info["turn"] == 1:
            if exchange:
                if "info" in meta_info and meta_info['info']['correction_1']:
                    prompt += ex_prompt.format(
                        query=dialogue.get_query_by_idx(0)["content"],
                        criteria=branch,
                        scoring=scoring,
                        response1=meta_info['info']['correction_1'],
                        response2=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                            "content"
                        ],
                    )
                else:
                    prompt += ex_prompt.format(
                        query=dialogue.get_query_by_idx(0)["content"],
                        criteria=branch,
                        scoring=scoring,
                        response1=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                            "content"
                        ],
                        response2=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                            "content"
                        ],
                    )
            else:
                if "info" in meta_info and meta_info['info']['correction_1']:
                    pdb.set_trace()
                    prompt += prompt.format(
                        query=dialogue.get_query_by_idx(0)["content"],
                        criteria=branch,
                        scoring=scoring,
                        response1=meta_info['info']['correction_1'],
                        response2=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                            "content"
                        ],
                    )
                else:
                    prompt += prompt.format(
                        query=dialogue.get_query_by_idx(0)["content"],
                        criteria=branch,
                        scoring=scoring,
                        response1=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                            "content"
                        ],
                        response2=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                            "content"
                        ],
                    )
        else:
            raise Exception("Not Support")
        return system_message, prompt

    def rating_format(self, result, eval_model):

        if "Response A Score: " in result:
            try:
                pos = result.find("Response A Score: ")
                x = result[pos + len("Response A Score: ")]
                rating_a = int(x.strip())
                
                pos = result.rfind("Response B Score: ")
                x = result[pos + len("Response B Score: ")]
                rating_b = int(x.strip())
            except:
                rating_a = 0
                rating_b = 0
        elif "Assistant 1's Response Score: " in result:
            try:
                pos = result.find("Assistant 1's Response Score: ")
                x = result[pos + len("Assistant 1's Response Score: ")]
                rating_a = int(x.strip())
                
                pos = result.rfind("Assistant 2's Response Score: ")
                x = result[pos + len("Assistant 2's Response Score: ")]
                rating_b = int(x.strip())
            except:
                rating_a = 0
                rating_b = 0
        else:
            raise Exception("Format Error")

        return rating_a, rating_b

    def get_result_from_server(
        self, dialogue, server, eval_model, branch, scoring, exchange=False
    ):
        system_message, prompt = self.build_prompt(
            dialogue, eval_model, branch, scoring, exchange
        )
        result = self.llm_server.chat_compeletion(
            eval_model,
            server,
            query=prompt,
            system=system_message,
            context=None,
            temperature=0.1,
            top_p=0.99,
        )
        if isinstance(result, str):
            if "<|im_end|>" in result:
                pos = result.find("<|im_end|>")
                return result[pos:]
            return result
        else:
            return result[0]["generated_text"]


@auto_register("fennec_pairwise_single_solving")
class FennecPairwiseSingleSolvingAction(Action):
    action_name = "fennec_pairwise_single_solving"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server
        
    def execute(self, **action_input):
        action_feedback = {
            "result_a": [],
            "result_b": [],
            "rating_a": [],
            "rating_b": [],
        }

        for branch, scoring in zip(
            action_input["branch_list"], action_input["scoring_list"]
        ):
            branch = branch.strip()
            scoring = scoring.strip()

            result_a = self.get_result_from_server(
                action_input["dialogue"],
                action_input["server"],
                action_input["eval_model"],
                branch,
                scoring,
            )
            action_feedback["result_a"].append(result_a)
            rating_a = self.rating_format(result_a, action_input["eval_model"])
            action_feedback["rating_a"].append(rating_a)

            result_b = self.get_result_from_server(
                action_input["dialogue"],
                action_input["server"],
                action_input["eval_model"],
                branch,
                scoring,
                exchange=True,
            )
            action_feedback["result_b"].append(result_b)
            rating_b = self.rating_format(result_b, action_input["eval_model"])
            action_feedback["rating_b"].append(rating_b)

        return action_feedback

    def build_prompt(
        self,
        dialogue: PairwiseDialogue,
        eval_model: str,
        branch: str,
        scoring: str,
        exchange: bool,
    ):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]

        system_message = template_json["single_system_message"]
        prompt = template_json["single_solving"]
        
        if meta_info["turn"] == 1:
            if exchange:
                prompt += prompt.format(
                    query=dialogue.get_query_by_idx(0)["content"],
                    criteria=branch,
                    scoring=scoring,
                    response=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                        "content"
                    ],
                )
            else:
                prompt += prompt.format(
                    query=dialogue.get_query_by_idx(0)["content"],
                    criteria=branch,
                    scoring=scoring,
                    response=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                        "content"
                    ],
                )
        else:
            raise Exception("Not Support")
        return system_message, prompt

    def rating_format(self, result, eval_model):

        if "Response Score: " in result:
            try:
                pos = result.find("Response Score: ")
                x = result[pos + len("Response Score: ")]
                rating = int(x.strip())
            except:
                rating = 0
        elif "Response A Score: " in result:
            try:
                pos = result.find("Response A Score: ")
                x = result[pos + len("Response A Score: ")]
                rating = int(x.strip())
            except:
                rating = 0
        else:
            rating = -1
            # raise Exception("Format Error")

        return rating

    def get_result_from_server(
        self, dialogue, server, eval_model, branch, scoring, exchange=False
    ):
        system_message, prompt = self.build_prompt(
            dialogue, eval_model, branch, scoring, exchange
        )
        result = self.llm_server.chat_compeletion(
            eval_model,
            server,
            query=prompt,
            system=system_message,
            context=None,
            temperature=0.1,
            top_p=0.99,
        )
        if isinstance(result, str):
            return result
        else:
            return result[0]["generated_text"]


@auto_register("fennec_pairwise_merge")
class FennecPairwiseMergeAction(Action):
    action_name = "fennec_pairwise_merge"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {"model_a": 0, "model_b": 0, "ex_model_a": 0, "ex_model_b": 0}
        idx = 0
        
        if "result" in action_input["result"]:
            for model_a, model_b, ex_model_a, ex_model_b in zip(
                action_input["result"]["rating_a"],
                action_input["result"]["rating_b"],
                action_input["result"]["ex_rating_a"],
                action_input["result"]["ex_rating_b"],
            ):
                
                if model_a <= 0 or model_b <= 0 or ex_model_a <= 0 or ex_model_b <= 0:
                    idx += 1
                    continue

                if idx < 5:
                    action_feedback["model_a"] += model_a
                    action_feedback["model_b"] += model_b
                    action_feedback["ex_model_a"] += ex_model_a
                    action_feedback["ex_model_b"] += ex_model_b
                idx += 1
        
        elif "result_a" in action_input["result"]:
            for model_a, model_b in zip(
                action_input["result"]["rating_a"],
                action_input["result"]["rating_b"],
            ):
                if model_a <= 0 or model_b <= 0:
                    idx += 1
                    continue
                
                if idx < 5:
                    action_feedback["model_a"] += model_a
                    action_feedback["model_b"] += model_b
                idx += 1
        
        return action_feedback


@auto_register("fennec_correction")
class FennecCorrectionAction(Action):
    action_name = "fennec_correction"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {}

        query = action_input["dialogue"].get_query_by_idx(0)["content"]
        response_a = action_input["dialogue"].get_pairwise_response_by_idx(0, "model_a")["content"]
        response_b = action_input["dialogue"].get_pairwise_response_by_idx(0, "model_b")["content"]

        branch_list = [b.replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "").replace("5.", "") for b in action_input["branch_list"]]
        solving_list = action_input["solving_list"]["result"]

        rating_a_list = action_input["solving_list"]["rating_a"]
        rating_b_list = action_input["solving_list"]["rating_b"]
        
        judge_a = []
        judge_b = []

        for b, a_score, b_score, solving in zip(branch_list, rating_a_list, rating_b_list, solving_list):
            a_pos = solving.find("Response A Score:")
            b_pos = solving.find("Response B Score:")
            e_pos = solving.find("Comparison:")
            if a_pos < b_pos < e_pos:
                if a_score <= 3:
                    j_a = solving[a_pos:b_pos].replace("Response A", "Response")
                    judge_a.append(b)
                    judge_a.append(j_a)
                if b_score <= 3:
                    j_b = solving[b_pos:e_pos].replace("Response B", "Response")
                    judge_b.append(b)
                    judge_b.append(j_b)
        
        if len(judge_a):
            result_a = self.get_result_from_server(
                action_input,
                query,
                response_a,
                "".join(judge_a),
                action_input["server"],
                action_input["eval_model"],
            )
            action_feedback["result_a"] = result_a
        if len(judge_b):
            result_b = self.get_result_from_server(
                action_input,
                query,
                response_b,
                "".join(judge_b),
                action_input["server"],
                action_input["eval_model"],
            )
            action_feedback["result_b"] = result_b
        return action_feedback

    def build_prompt(self, query, response, judge, eval_model):
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        system_message = template_json["correction_message"]
        
        prompt = template_json["correction"].format(
            query=query,
            response=response,
            judge=judge,
        )
        return system_message, prompt

    def get_result_from_server(self, action_input, query, response, judge, server, eval_model):
        system_message, prompt = self.build_prompt(query, response, judge, eval_model)
        if action_input["dialogue"].get_meta_info()['category'] == "writing" or action_input["dialogue"].get_meta_info()['category'] == "roleplay":
            temperature = 0.1
            top_p=0.95
        elif action_input["dialogue"].get_meta_info()['category'] == "stem" or action_input["dialogue"].get_meta_info()['category'] == "humanities":
            temperature = 0.1
            top_p=0.95
        else:
            temperature = 0.0
            top_p=1.0
        
        result = self.llm_server.chat_compeletion(
            eval_model,
            server,
            query=prompt,
            system=system_message,
            context=None,
            temperature=temperature,
            top_p=top_p,
        )
        if isinstance(result, str):
            return result
        else:
            raise Exception("error")

