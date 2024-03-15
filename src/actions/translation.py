from data.dialogue import PairwiseDialogue
from server.llm_server import LLMServer
from .registry import auto_register
from .meta import Action
import pdb


@auto_register("translation_comparing")
class TranslationComparingAction(Action):
    action_name = "translation_comparing"

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
        action_feedback["prediction"] = self.pariwise_format(result)
        ex_result = self.get_result_from_server(
            action_input["dialogue"],
            action_input["server"],
            action_input["eval_model"],
            exchange=True,
        )
        action_feedback["ex_result"] = ex_result
        action_feedback["ex_prediction"] = self.pariwise_format(ex_result)
        return action_feedback

    def pariwise_format(self, raw_output):
        raw_output = raw_output.strip()
        pos = raw_output.rfind("final decision is ")
        pred_label = -1
        if pos != -1:
            pred_rest = raw_output[pos + len("final decision is ") :].strip().lower()
            if pred_rest.startswith("candidate a"):
                pred_label = 0
            elif pred_rest.startswith("candidate b"):
                pred_label = 1
            elif pred_rest.startswith("tie"):
                pred_label = 2
        return pred_label

    def build_prompt(self, dialogue: PairwiseDialogue, eval_model, exchange):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        system_message = template_json["system"]

        if meta_info["turn"] == 1:
            if exchange:
                prompt = template_json["instruction"].format(
                    source=dialogue.get_query_by_idx(0)["content"],
                    target1=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                        "content"
                    ],
                    target2=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                        "content"
                    ],
                )
            else:
                prompt = template_json["instruction"].format(
                    source=dialogue.get_query_by_idx(0)["content"],
                    target1=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                        "content"
                    ],
                    target2=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                        "content"
                    ],
                )
        else:
            raise Exception("Not Support")
        return system_message, prompt

    def get_result_from_server(
        self,
        dialogue,
        server,
        eval_model,
        exchange=False,
    ):
        system_message, prompt = self.build_prompt(dialogue, eval_model, exchange)
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
