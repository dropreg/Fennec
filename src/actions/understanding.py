from .registry import auto_register
from .meta import Action


@auto_register("evaluate_understanding")
class EvaluateUnderstandingAction(Action):
    action_name = "evaluate_understanding"

    def __init__(self, config, llm_server) -> None:
        super().__init__(config)

        self.llm_server = llm_server
        template_json = self.get_template_json()[self.get_lang()]
        self.subcategories = template_json["subcategories"]
        self.subcategories2categories = {}
        for categories, sub_list in template_json["categories"].items():
            for sub in sub_list:
                self.subcategories2categories[sub] = categories

    def execute(self, **action_input):
        action_feedback = {}
        eval_model = (
            action_input["eval_model"]
            if "eval_model" in action_input
            else self.eval_model
        )

        result = self.get_result_from_server(action_input["query"], eval_model)
        action_feedback["prediction"] = result["prediction"]
        action_feedback["probs"] = result["probs"]

        subctg = self.subcategories[action_input["meta_info"]["subject"]][0]
        action_feedback["subcategories"] = subctg
        action_feedback["categories"] = self.subcategories2categories[subctg]
        return action_feedback

    def get_result_from_server(self, query, eval_model):
        result = self.llm_server.chat_compeletion(
            eval_model, query=query, system=None, context=None
        )
        return result
