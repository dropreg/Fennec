from data.dialogue import Dialogue, PairwiseDialogue
from server.llm_server import LLMServer
from .registry import auto_register
from .meta import Action
import pdb


@auto_register("bsm_branch")
class BSMBranchAction(Action):
    action_name = "bsm_branch"

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
        action_feedback["branch_list"] = self.extract_branch_list(result)
        return action_feedback

    def extract_branch_list(self, result):
        if "\n<|assistant|>\n" in result:
            result = result.replace("\n<|assistant|>\n", "")
        branch_list = []
        for branch in result.split("\n"):
            if branch:
                branch_list.append(branch)
        return branch_list[:5]

    def build_prompt(self, dialogue, eval_model):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()]
        system_message = template_json["system"]

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
            return result[0]["generated_text"]


@auto_register("bsm_pairwise_solving")
class BSMPairwiseSolvingAction(Action):
    action_name = "bsm_pairwise_solving"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {
            "solving_list": [],
            "solving_score": [],
            "ex_solving_list": [],
            "ex_solving_score": [],
        }

        for idx, branch in enumerate(action_input["branch_list"]):
            result = self.get_result_from_server(
                branch,
                action_input["dialogue"],
                action_input["server"],
                action_input["eval_model"],
            )
            action_feedback["solving_list"].append(result)
            action_feedback["solving_score"].append(self.extract_solving_score(result))

            ex_result = self.get_result_from_server(
                branch,
                action_input["dialogue"],
                action_input["server"],
                action_input["eval_model"],
                exchange=True,
            )
            action_feedback["ex_solving_list"].append(ex_result)
            action_feedback["ex_solving_score"].append(
                self.extract_solving_score(ex_result)
            )
        return action_feedback

    def extract_solving_score(self, result):
        if "\n<|assistant|>\n" in result:
            result = result.replace("\n<|assistant|>\n", "")
        if "\n***\nEvaluation:\n" in result:
            pos = result.find("\n***\nEvaluation:\n")
            result = result[pos:]

        score_1 = 0
        score_2 = 0
        for result_span in result.split("\n"):
            try:
                pos = result_span.rfind("Response A: ")
                if pos != -1 and score_1 == 0:
                    pred_rest = result_span[pos + len("Response A: ") :].strip().lower()
                    score_1 = int(pred_rest[:1].strip())
            except:
                pass
            try:
                pos = result_span.rfind("Response B: ")
                if pos != -1 and score_2 == 0:
                    pred_rest = result_span[pos + len("Response B: ") :].strip().lower()
                    score_2 = int(pred_rest[:1].strip())
            except:
                pass
        return {
            "score_1": score_1,
            "score_2": score_2,
        }

    def build_prompt(self, branch, dialogue, eval_model, exchange):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()]
        system_message = template_json["system"]

        pdb.set_trace()
        if meta_info["turn"] == 1:
            if exchange:
                prompt = template_json["solve"].format(
                    query=dialogue.get_query_by_idx(0)["content"],
                    response1=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                        "content"
                    ],
                    response2=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                        "content"
                    ],
                    eval_criterion=branch,
                )
            else:
                prompt = template_json["solve"].format(
                    query=dialogue.get_query_by_idx(0)["content"],
                    response1=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                        "content"
                    ],
                    response2=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                        "content"
                    ],
                    eval_criterion=branch,
                )
        else:
            raise Exception("Not Support")
        return system_message, prompt

    def get_result_from_server(
        self, branch, dialogue, server, eval_model, exchange=False
    ):
        system_message, prompt = self.build_prompt(
            branch, dialogue, eval_model, exchange
        )
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


@auto_register("bsm_pairwise_merge")
class BSMPairwiseMergeAction(Action):
    action_name = "bsm_pairwise_merge"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {}

        model_a = 0
        model_b = 0

        ex_model_a = 0
        ex_model_b = 0
        for score, ex_score in zip(
            action_input["solving_score"], action_input["ex_solving_score"]
        ):
            if (
                score["score_1"] > 0
                and score["score_2"] > 0
                and ex_score["score_1"] > 0
                and ex_score["score_2"] > 0
            ):
                model_a += score["score_1"]
                model_b += score["score_2"]
                ex_model_a += ex_score["score_1"]
                ex_model_b += ex_score["score_2"]

        action_feedback["model_a"] = model_a
        action_feedback["model_b"] = model_b
        action_feedback["ex_model_a"] = ex_model_a
        action_feedback["ex_model_b"] = ex_model_b

        return action_feedback


@auto_register("bsm_single_branch")
class BSMSingleBranchAction(Action):
    action_name = "bsm_single_branch"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server
        self.max_branch = 5

    def execute(self, **action_input):
        action_feedback = {"branch_list": []}
        for b_idx in range(self.max_branch):
            result = self.get_result_from_server(
                action_input["dialogue"],
                action_input["server"],
                action_input["eval_model"],
            )
            action_feedback["branch_list"].append(self.extract(result))
        return action_feedback

    def extract(self, result):
        if "\n<|assistant|>\n" in result:
            result = result.replace("\n<|assistant|>\n", "")
        return result

    def build_prompt(self, dialogue, eval_model):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        system_message = template_json["system"]

        if meta_info["turn"] == 1:
            prompt = template_json["single_branch"].format(
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
            return result[0]["generated_text"]


@auto_register("bsm_single_solving")
class BSMSingleSolvingAction(Action):
    action_name = "bsm_single_solving"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {
            "solving_list": [],
            "solving_score": [],
            "ex_solving_list": [],
            "ex_solving_score": [],
            "solving_list2": [],
            "solving_score2": [],
            "ex_solving_list2": [],
            "ex_solving_score2": [],
        }

        for idx in range(2):
            for b_idx, branch in enumerate(action_input["branch_list"]):
                result = self.get_result_from_server(
                    branch,
                    action_input["dialogue"],
                    action_input["server"],
                    action_input["eval_model"],
                )
                if idx:
                    action_feedback["solving_list2"].append(result)
                    action_feedback["solving_score2"].append(
                        self.extract_solving_score(result)
                    )
                else:
                    action_feedback["solving_list"].append(result)
                    action_feedback["solving_score"].append(
                        self.extract_solving_score(result)
                    )

                ex_result = self.get_result_from_server(
                    branch,
                    action_input["dialogue"],
                    action_input["server"],
                    action_input["eval_model"],
                    exchange=True,
                )
                if idx:
                    action_feedback["ex_solving_list2"].append(ex_result)
                    action_feedback["ex_solving_score2"].append(
                        self.extract_solving_score(ex_result)
                    )
                else:
                    action_feedback["ex_solving_list"].append(ex_result)
                    action_feedback["ex_solving_score"].append(
                        self.extract_solving_score(ex_result)
                    )
        return action_feedback

    def extract_solving_score(self, result):
        if "\n<|assistant|>\n" in result:
            result = result.replace("\n<|assistant|>\n", "")

        x = result.split("[RESULT]")
        try:
            if len(x) >= 2:
                rating = int(x[1].strip())
            else:
                rating = -1
        except:
            rating = 0
        return rating

    def build_prompt(self, branch, dialogue, eval_model, exchange):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        system_message = template_json["system"]

        if meta_info["turn"] == 1:
            if exchange:
                prompt = template_json["single_solve"].format(
                    query=dialogue.get_query_by_idx(0)["content"],
                    response=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                        "content"
                    ],
                    criterion=branch,
                )
            else:
                prompt = template_json["single_solve"].format(
                    query=dialogue.get_query_by_idx(0)["content"],
                    response=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                        "content"
                    ],
                    criterion=branch,
                )
        else:
            raise Exception("Not Support")
        return system_message, prompt

    def get_result_from_server(
        self, branch, dialogue, server, eval_model, exchange=False
    ):
        system_message, prompt = self.build_prompt(
            branch, dialogue, eval_model, exchange
        )
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


@auto_register("bsm_single_merge")
class BSMSingleMergeAction(Action):
    action_name = "bsm_single_merge"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {}

        model_a = 0
        model_b = 0

        ex_model_a = 0
        ex_model_b = 0

        idx = 0
        for score, ex_score, score2, ex_score2 in zip(
            action_input["solving_score"],
            action_input["ex_solving_score"],
            action_input["solving_score2"],
            action_input["ex_solving_score2"],
        ):
            if score > 0 and ex_score > 0 and score2 > 0 and ex_score2 > 0:
                model_a += score
                model_b += ex_score

                ex_model_a += score2
                ex_model_b += ex_score2
            idx += 1

        action_feedback["model_a"] = model_a
        action_feedback["model_b"] = model_b
        action_feedback["ex_model_a"] = ex_model_a
        action_feedback["ex_model_b"] = ex_model_b

        return action_feedback
