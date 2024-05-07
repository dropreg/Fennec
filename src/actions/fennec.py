import pdb
from urllib import response
from data.dialogue import PairwiseDialogue
from server.llm_server import LLMServer
from .registry import auto_register
from .meta import Action
from cachetools import FIFOCache
import pickle
import os
import re
import random

@auto_register("fennec_branch")
class FennecBranchAction(Action):
    action_name = "fennec_branch"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server
        self.max_branch = 10

    def execute(self, **action_input):
        action_feedback = {}
        result = self.get_result_from_server(
            action_input["dialogue"],
            action_input["context"],
            action_input["server"],
            action_input["eval_model"],
        )
        action_feedback["result"] = result
        action_feedback["branch_list"] = self.extract_branch_list(result)
        return action_feedback

    def starts_with_numbered_list(self, text):
        pattern = r"^\d+\.\s"
        return bool(re.match(pattern, text))
        
    def criteria_extraction(self, criteria):
        criteria_list = []
        for line in criteria.split("\n"):
            if self.starts_with_numbered_list(line):
                criteria_list.append(line)
        if criteria_list == []:
            criteria_list = [c for c in criteria.split("\n") if c.replace('\n', '').strip() != ""]
        return criteria_list
    
    def extract_branch_list(self, result):
        if "\n<|assistant|>\n" in result:
            result = result.replace("\n<|assistant|>\n", "")
        elif "[/INST]" in result:
            result = result.split("[/INST]")[-1]
        # branch_list = []
        # for branch in result.split("\n"):
        #     if branch:
        #         branch_list.append(branch)
        
        branch_list = [b.replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "").replace("5.", "")
                       .replace("6.", "").replace("7.", "").replace("8.", "").replace("9.", "").replace("10.", "") for b in self.criteria_extraction(result)]

        return branch_list[: self.max_branch]
    
    def build_context(
        self,
        context,
    ):
        formatted_str = ""
        for i, text in enumerate(context):
            if i % 2 == 0:
                formatted_str += "[Query]: " + text
            else:
                formatted_str += "[Response]: " + text
            if i < len(context) - 1:
                formatted_str += "\n"
        return formatted_str
    
    def build_prompt(self, dialogue, context, eval_model):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        if "branch_message" not in template_json and "system" not in template_json:
            system_message = None
        else:
            system_message = template_json["branch_message"] if "branch_message" in template_json.keys() else template_json["system"]
        # system_message = None
        if meta_info["turn"] == 1:
            prompt = template_json["branch"].format(
                query=dialogue.get_query_by_idx(0)["content"],
                response_a=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                    "content"
                ],
                response_b=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                    "content"
                ],
                # context=self.build_context(context) if context and context != "" else "None.",
            )
        else:
            raise Exception("Not Support")
        return system_message, prompt

    def get_result_from_server(self, dialogue, context, server, eval_model):
        system_message, prompt = self.build_prompt(dialogue, context, eval_model)
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
            if "[/INST]" in result:
                result = result.split("[/INST]")[-1]
            if "\n<|assistant|>\n" in result:
                result = result.split("\n<|assistant|>\n")[-1]
            result = result.replace("[Scoring Guideline]:", "")
            action_feedback["result"].append(result)
        return action_feedback

    def build_prompt(self, dialogue: PairwiseDialogue, eval_model: str, branch: str):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        if "branch_message" not in template_json and "system" not in template_json:
            system_message = None
        else:
            system_message = template_json["system"]
        # system_message = None
        
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
        self.cache = FIFOCache(maxsize=500)

    def save_to_cache(self, key, value):
        self.cache[key] = value

    def load_from_cache(self, key):
        return self.cache.get(key)

    def save_cache_to_file(self, action_name, dialogue_idx):
        filename = "tmp/{}_{}.pkl".format(action_name, dialogue_idx)
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.cache), f)

    def load_cache_from_file(self, action_name, dialogue_idx):
        filename = "tmp/{}_{}.pkl".format(action_name, dialogue_idx)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.cache.update(pickle.load(f))
        return filename

    def delete_cache(self, action_name, dialogue_idx):
        filename = "tmp/{}_{}.pkl".format(action_name, dialogue_idx)
        if os.path.exists(filename):
            os.remove(filename)

    def execute(self, **action_input):
        action_feedback = {
            "result": [],
            "rating_a": [],
            "rating_b": [],
            "ex_result": [],
            "ex_rating_a": [],
            "ex_rating_b": [],
        }
        meta_info = action_input["dialogue"].get_meta_info()
        dialogue_idx = "{}_{}_{}".format(meta_info['question_id'], meta_info['model_a'], meta_info['model_b'])
        if action_input["eval_model"] == "one_chat":
            filename = self.load_cache_from_file("pairwise_solving", dialogue_idx)
            self.logger.info("load cache {}, length = {}".format(filename, len(self.cache)))
        branch_idx = 0
        cache_key_list = []
        for branch, scoring in zip(
            action_input["branch_list"], action_input["scoring_list"]
        ):
            branch_idx += 1
            branch = branch.strip()
            scoring = scoring.strip()
            
            if action_input["eval_model"] == "one_chat":
                cache_key = filename + "_" + str(branch_idx)
            if action_input["eval_model"] == "one_chat" and cache_key in self.cache:
                result = self.load_from_cache(cache_key)
            else:
                result = self.get_result_from_server(
                    action_input["dialogue"],
                    action_input["server"],
                    action_input["eval_model"],
                    action_input["context"],
                    branch,
                    scoring,
                )
            if action_input["eval_model"] == "one_chat":
                self.save_to_cache(cache_key, result)
                self.save_cache_to_file("pairwise_solving", dialogue_idx)
                self.logger.info("save cache {} and cache_key = {}".format(filename, cache_key))
                cache_key_list.append(cache_key)
            if "[/INST]" in result:
                result = result.split("[/INST]")[-1]
            if "\n<|assistant|>\n" in result:
                result = result.split("\n<|assistant|>\n")[-1]
            action_feedback["result"].append(result)
            rating_a, rating_b = self.rating_format(result, action_input["eval_model"])
            action_feedback["rating_a"].append(rating_a)
            action_feedback["rating_b"].append(rating_b)

            if "eval" in action_input and action_input["eval"]:
                ex_result = self.get_result_from_server(
                    action_input["dialogue"],
                    action_input["server"],
                    action_input["eval_model"],
                    action_input["context"],
                    branch,
                    scoring,
                    exchange=True,
                )
                if "[/INST]" in ex_result:
                    ex_result = ex_result.split("[/INST]")[-1]
                action_feedback["ex_result"].append(ex_result)
                ex_rating_a, ex_rating_b = self.rating_format(
                    ex_result, action_input["eval_model"]
                )
                action_feedback["ex_rating_a"].append(ex_rating_a)
                action_feedback["ex_rating_b"].append(ex_rating_b)
        if action_input["eval_model"] == "one_chat":
            self.delete_cache("pairwise_solving", dialogue_idx)
            self.logger.info("delete cache {}".format(filename))   
            for cache_key in cache_key_list:
                self.cache.pop(cache_key)
        return action_feedback
    
    def build_context(
        self,
        context,
    ):
        formatted_str = ""
        for i, text in enumerate(context):
            if i % 2 == 0:
                formatted_str += "[Query]: " + text
            else:
                formatted_str += "[Response]: " + text
            if i < len(context) - 1:
                formatted_str += "\n"
        return formatted_str
        
    def build_prompt(
        self,
        dialogue: PairwiseDialogue,
        eval_model: str,
        context: str,
        branch: str,
        scoring: str,
        exchange: bool,
    ):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        if "branch_message" not in template_json and "system" not in template_json:
            system_message = None
        else:
            system_message = template_json["system"]
        # system_message = None
        prompt = template_json["sovling"]
        ex_prompt = template_json["ex_sovling"] if "ex_sovling" in template_json else None
        
        if meta_info["turn"] == 1:
            if exchange:
                if "info" in meta_info and meta_info['info']['correction_1']:
                    prompt = ex_prompt.format(
                        query=dialogue.get_query_by_idx(0)["content"],
                        criteria=branch,
                        scoring=scoring,
                        response1=meta_info['info']['correction_1'],
                        response2=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                            "content"
                        ],
                        context=self.build_context(context) if context and context != "" else "None.",
                    )
                else:
                    prompt = ex_prompt.format(
                        query=dialogue.get_query_by_idx(0)["content"],
                        criteria=branch,
                        scoring=scoring,
                        response1=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                            "content"
                        ],
                        response2=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                            "content"
                        ],
                        context=self.build_context(context) if context and context != "" else "None.",
                    )
            else:
                if "info" in meta_info and meta_info['info']['correction_1']:
                    # pdb.set_trace()
                    prompt = prompt.format(
                        query=dialogue.get_query_by_idx(0)["content"],
                        criteria=branch,
                        scoring=scoring,
                        response1=meta_info['info']['correction_1'],
                        response2=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                            "content"
                        ],
                        context=self.build_context(context) if context and context != "" else "None.",
                    )
                else:
                    prompt = prompt.format(
                        query=dialogue.get_query_by_idx(0)["content"],
                        criteria=branch,
                        scoring=scoring,
                        response1=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                            "content"
                        ],
                        response2=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                            "content"
                        ],
                        context=self.build_context(context) if context and context != "" else "None.",
                    )
        else:
            raise Exception("Not Support")
        return system_message, prompt

    def rating_format(self, result, eval_model):
        result = result.replace("*", "")
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
        # if 'the final decision is response a' in result.lower():
        #     rating_a = 1
        #     rating_b = -1
        # elif 'the final decision is response b' in result.lower():
        #     rating_a = -1
        #     rating_b = 1
        # elif 'the final decision is tie' in result.lower() or 'the final decision is a tie' in result.lower():
        #     rating_a = 1
        #     rating_b = 1
        else:
            # import pdb;pdb.set_trace()
            print(result)
            raise Exception("Format Error")

        return rating_a, rating_b

    def get_result_from_server(
        self, dialogue, server, eval_model, context, branch, scoring, exchange=False
    ):
        system_message, prompt = self.build_prompt(
            dialogue, eval_model, context, branch, scoring, exchange
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
                prompt = prompt.format(
                    query=dialogue.get_query_by_idx(0)["content"],
                    criteria=branch,
                    scoring=scoring,
                    response=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                        "content"
                    ],
                )
            else:
                prompt = prompt.format(
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
        
        judge = action_input["judge"][0]
        # if judge == "model_a":
        #     judge = 0
        # elif judge == "model_b":
        #     judge = 1
        # elif "tie" in judge:
        #     judge = 2
                    
        if "result" in action_input["result"]:
            l = random.sample(range(10), 5)
            x_l = [i for i, (a, b) in enumerate(zip(action_input["result"]["rating_a"],action_input["result"]["rating_b"])) if a != b]
            if len(x_l) != 0:
                # x = random.choice(x_l)
                x = x_l[0]
            else:
                x = 0
            for model_a, model_b, ex_model_a, ex_model_b in zip(
                action_input["result"]["rating_a"],
                action_input["result"]["rating_b"],
                action_input["result"]["ex_rating_a"],
                action_input["result"]["ex_rating_b"],
            ):
                
                # if model_a <= 0 or model_b <= 0 or ex_model_a <= 0 or ex_model_b <= 0:
                #     idx += 1
                #     continue
                # import random
                
                # if 5 <= idx < 10:
                # if idx in l:
                if idx < 10:
                # if idx == x:
                # import pdb;pdb.set_trace()
                # if (judge == 0 and model_a > model_b) or (judge == 1 and model_a < model_b) or (judge == 2 and model_a == model_b):
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
        
@auto_register("fennec_branch_selection")
class FennecBranchSelectionAction(Action):
    action_name = "fennec_branch_selection"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server
        
    def extract_judge(self, text):
        if '[Decision]' in text:
            decision = text.split('[Decision]')[-1]
        elif 'Decision' in text:
            decision = text.split('Decision')[-1]
        else:
            return 'error'
        decision = decision.split('.')[0]
        a = False
        b = False
        if "assistant a" in decision.lower():
            a = True
        if "assistant b" in decision.lower():
            b = True
        if a and b:
            # raise Exception("format error: {}".format(text))
            print("error: cannot extract!")
            return 'error'
        else:
            if a:
                return 'A'
            elif b:
                return 'B'
            else:
                return 'error'
            
    def execute(self, **action_input):
        action_feedback = {}

        query = action_input["dialogue"].get_query_by_idx(0)["content"]
        response_a = action_input["dialogue"].get_pairwise_response_by_idx(0, "model_a")["content"]
        response_b = action_input["dialogue"].get_pairwise_response_by_idx(0, "model_b")["content"]

        branch_list = [b.replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "").replace("5.", "") for b in action_input["branch_list"]]
        scoring_list = action_input["scoring_list"]
        solving_list = action_input["solving_list"]["result"]

        rating_a_list = action_input["solving_list"]["rating_a"]
        rating_b_list = action_input["solving_list"]["rating_b"]

        action_feedback["judge"] = []
        action_feedback["result"] = []
        action_feedback["rating_a"] = []
        action_feedback["rating_b"] = []
        
        # for b, a_score, b_score, scoring, solving in zip(branch_list, rating_a_list, rating_b_list, scoring_list, solving_list):
        for idx in range(5):
            if idx + 5 > len(branch_list):
                print("error: out of range")
                break
            max_repeat = 10
            repeat = 0
            while repeat < max_repeat:
                result = self.get_result_from_server(
                    query,
                    response_a,
                    response_b,
                    [branch_list[idx], branch_list[idx + 5]],
                    [scoring_list[idx], scoring_list[idx + 5]],
                    [solving_list[idx], solving_list[idx + 5]],
                    action_input["server"],
                    action_input["eval_model"],
                )
                judge = self.extract_judge(result)
                if judge != 'error':
                    break
                else:
                    repeat += 1
            if judge not in ['A', 'B']:
                judge = random.choice(['A', 'B'])
                print("error: random choice {}".format(judge))
            action_feedback["judge"].append(judge)
            action_feedback["result"].append(result)
            if judge == 'A':
                action_feedback["rating_a"].append(rating_a_list[idx])
                action_feedback["rating_b"].append(rating_b_list[idx])
            else:
                action_feedback["rating_a"].append(rating_a_list[idx + 5])
                action_feedback["rating_b"].append(rating_b_list[idx + 5])
        return action_feedback

    def build_prompt(self, query, response_a, response_b, branch_pair, scoring_pair, solving_pair, eval_model):
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        system_message = None
        
        prompt = template_json["selection"].format(
            query=query,
            response_a=response_a,
            response_b=response_b,
            branch_a=branch_pair[0],
            branch_b=branch_pair[1],
            scoring_a=scoring_pair[0],
            scoring_b=scoring_pair[1],
            solving_a=solving_pair[0],
            solving_b=solving_pair[1]
        )
        return system_message, prompt

    def get_result_from_server(self, query, response_a, response_b, branch_pair, scoring_pair, solving_pair, server, eval_model):
        system_message, prompt = self.build_prompt(query, response_a, response_b, branch_pair, scoring_pair, solving_pair, eval_model)
        
        result = self.llm_server.chat_compeletion(
            eval_model,
            server,
            query=prompt,
            system=system_message,
            context=None,
            temperature=0.1,
            top_p=0.99,
        )
        # import pdb; pdb.set_trace()
        if isinstance(result, str):
            return result
        else:
            raise Exception("error")

@auto_register("fennec_branch_quick_sort")
class FennecBranchQuickSortAction(Action):
    action_name = "fennec_branch_quick_sort"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server
        self.query = ""
        self.response_a = ""
        self.response_b = ""
        self.branch_list = []
        self.scoring_list = []
        self.solving_list = []
        
    def extract_judge(self, text):
        if '[Decision]' in text:
            decision = text.split('[Decision]')[-1]
        elif 'Decision' in text:
            decision = text.split('Decision')[-1]
        else:
            return 'error'
        decision = decision.split('.')[0]
        a = False
        b = False
        if "assistant a" in decision.lower():
            a = True
        if "assistant b" in decision.lower():
            b = True
        if a and b:
            # raise Exception("format error: {}".format(text))
            print("error: cannot extract!")
            return 'error'
        else:
            if a:
                return 'A'
            elif b:
                return 'B'
            else:
                return 'error'
    
    def compare(self, idx1, idx2, server, eval_model):
        max_repeat = 10
        repeat = 0
        while repeat < max_repeat:
            result = self.get_result_from_server(
                self.query,
                self.response_a,
                self.response_b,
                [self.branch_list[idx1], self.branch_list[idx2]],
                [self.scoring_list[idx1], self.scoring_list[idx2]],
                [self.solving_list[idx1], self.solving_list[idx2]],
                server,
                eval_model,
            )
            judge = self.extract_judge(result)
            if judge != 'error':
                break
            else:
                repeat += 1
        if judge not in ['A', 'B']:
            judge = random.choice(['A', 'B'])
            print("error: random choice {}".format(judge))
        return judge
    
    def partition(self, arr, low: int, high: int, server, eval_model):
        pivot, j = arr[low], low
        for i in range(low + 1, high + 1):
            if self.compare(arr[i], pivot, server, eval_model) == 'A':
                j += 1
                arr[j], arr[i] = arr[i], arr[j]
        arr[low], arr[j] = arr[j], arr[low]
        return j 

    def quick_sort_between(self, arr, low: int, high: int, server, eval_model):
        if high-low <= 1: # 递归结束条件
            return
        m = self.partition(arr, low, high, server, eval_model)  # arr[m] 作为划分标准
        self.quick_sort_between(arr, low, m - 1, server, eval_model)
        self.quick_sort_between(arr, m + 1, high, server, eval_model)

    def quicksort(self, arr, server, eval_model):
        """
        快速排序(in-place)
        :param arr: 待排序的List
        :return: 快速排序是就地排序(in-place)
        """
        self.quick_sort_between(arr, 0, len(arr) - 1, server, eval_model)

    def execute(self, **action_input):
        action_feedback = {}

        self.query = action_input["dialogue"].get_query_by_idx(0)["content"]
        self.response_a = action_input["dialogue"].get_pairwise_response_by_idx(0, "model_a")["content"]
        self.response_b = action_input["dialogue"].get_pairwise_response_by_idx(0, "model_b")["content"]

        self.branch_list = [b.replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "").replace("5.", "") for b in action_input["branch_list"]]
        self.scoring_list = action_input["scoring_list"]
        self.solving_list = action_input["solving_list"]["result"]

        rating_a_list = action_input["solving_list"]["rating_a"]
        rating_b_list = action_input["solving_list"]["rating_b"]
        server = action_input["server"]
        eval_model = action_input["eval_model"]
        
        action_feedback["judge"] = []
        action_feedback["result"] = []
        action_feedback["rating_a"] = []
        action_feedback["rating_b"] = []
        arr = [idx for idx, (a, b) in enumerate(zip(rating_a_list, rating_b_list)) if a != b]
        
        self.quicksort(arr, server, eval_model)
        action_feedback["result"] = arr
        action_feedback["rating_a"] = [rating_a_list[x] for x in action_feedback["result"]]
        action_feedback["rating_b"] = [rating_b_list[x] for x in action_feedback["result"]]
        
        return action_feedback

    def build_prompt(self, query, response_a, response_b, branch_pair, scoring_pair, solving_pair, eval_model):
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        system_message = None
        
        prompt = template_json["selection"].format(
            query=query,
            response_a=response_a,
            response_b=response_b,
            branch_a=branch_pair[0],
            branch_b=branch_pair[1],
            scoring_a=scoring_pair[0],
            scoring_b=scoring_pair[1],
            solving_a=solving_pair[0],
            solving_b=solving_pair[1]
        )
        return system_message, prompt

    def get_result_from_server(self, query, response_a, response_b, branch_pair, scoring_pair, solving_pair, server, eval_model):
        system_message, prompt = self.build_prompt(query, response_a, response_b, branch_pair, scoring_pair, solving_pair, eval_model)
        
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
            raise Exception("error")
