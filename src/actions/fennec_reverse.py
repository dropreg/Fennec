import pdb
from data.dialogue import PairwiseDialogue
from server.llm_server import LLMServer
from .registry import auto_register
from .meta import Action


@auto_register("fennec_reverse_branch")
class FennecReverseBranchAction(Action):
    action_name = "fennec_reverse_branch"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {}
        branch_list = []
        for branch_prompt in action_input["candi_branch"]:
            result = self.get_result_from_server(
                branch_prompt,
                action_input["server"],
                action_input["eval_model"],
            )
            branch_list.append(result.strip())
        action_feedback["branch_list"] = branch_list
        return action_feedback

    def get_result_from_server(self, branch_prompt, server, eval_model):
        result = self.llm_server.chat_compeletion(
            eval_model,
            server,
            query=branch_prompt,
            system="",
            context=None,
        )
        if isinstance(result, str):
            if "\n<|assistant|>\n" in result:
                pos = result.find("\n<|assistant|>\n")
                result = result[pos + len("\n<|assistant|>\n") :]
            return result
        else:
            raise Exception("error")


@auto_register("fennec_reverse_scoring")
class FennecReverseScoringAction(Action):
    action_name = "fennec_reverse_scoring"

    def __init__(self, config, llm_server: LLMServer) -> None:
        super().__init__(config)

        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {}
        scoring_list = []
        for scoring_prompt in action_input["candi_scoring"]:
            result = self.get_result_from_server(
                scoring_prompt,
                action_input["server"],
                action_input["eval_model"],
            )
            scoring_list.append(result.strip())
        action_feedback["scoring_list"] = scoring_list
        return action_feedback

    def get_result_from_server(self, scoring_prompt, server, eval_model):
        result = self.llm_server.chat_compeletion(
            eval_model,
            server,
            query=scoring_prompt,
            system="",
            context=None,
        )
        if isinstance(result, str):
            if "\n<|assistant|>\n" in result:
                pos = result.find("\n<|assistant|>\n")
                result = result[pos + len("\n<|assistant|>\n") :]
            return result
        else:
            raise Exception("error")
