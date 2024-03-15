from data.dialogue import Dialogue, PairwiseDialogue
from server.llm_server import LLMServer
from .registry import auto_register
from .meta import Action
import pdb


@auto_register("pairwise_comparing")
class PairwiseComparingAction(Action):
    action_name = "pairwise_comparing"

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
        action_feedback["prediction"] = self.pariwise_format(
            result, action_input["eval_model"]
        )
        ex_result = self.get_result_from_server(
            action_input["dialogue"],
            action_input["server"],
            action_input["eval_model"],
            exchange=True,
        )
        action_feedback["ex_result"] = ex_result
        action_feedback["ex_prediction"] = self.pariwise_format(
            ex_result, action_input["eval_model"]
        )
        pdb.set_trace()
        return action_feedback

    def pariwise_format(self, raw_output, eval_model):
        raw_output = raw_output.strip()
        if eval_model == "judgelm" or eval_model == "one_chat":
            pred_label = -1
            pos_a = raw_output.find("Assistant 1: ")
            pos_b = raw_output.find("Assistant 2: ")
            if pos_a != -1 and pos_b != -1:
                
                x = raw_output[pos_a + len("Assistant 1: ")]
                try:
                    rating_a = int(x.strip())
                except:
                    raise Exception("parse error")

                x = raw_output[pos_b + len("Assistant 2: ")]
                try:
                    rating_b = int(x.strip())  
                except:
                    raise Exception("parse error")
                
                if rating_a > rating_b:
                    pred_label = 0
                elif rating_a < rating_b:
                    pred_label = 1
                else:
                    pred_label = 2

        elif eval_model == "zephyr_judgelm":
            pos = raw_output.rfind("<|assistant|>\n")
            pred_label = -1
            if pos != -1:
                pred_rest = raw_output[pos + len("<|assistant|>\n") :].split("\n")[0]
                pred_a, pred_b = pred_rest.split(" ")
                pred_a, pred_b = float(pred_a), float(pred_b)
                if pred_a > pred_b:
                    pred_label = 0
                elif pred_a < pred_b:
                    pred_label = 1
                elif pred_a == pred_b:
                    pred_label = 2
                else:
                    pred_label = -1
        elif "final decision is " in raw_output:
            pos = raw_output.rfind("final decision is ")
            pred_label = -1
            if pos != -1:
                pred_rest = (
                    raw_output[pos + len("final decision is ") :].strip().lower()
                )
                if pred_rest.startswith("response 1"):
                    pred_label = 0
                elif pred_rest.startswith("response 2"):
                    pred_label = 1
                elif pred_rest.startswith("tie"):
                    pred_label = 2
        elif "Assistant 1's Response Score: " in raw_output:
            pos = raw_output.find("Assistant 1's Response Score: ")
            pred_label = -1
            if pos != -1:
                x = raw_output[pos + len("Assistant 1's Response Score: ")]
                rating_a = int(x.strip())

                pos = raw_output.rfind("Assistant 2's Response Score: ")
                x = raw_output[pos + len("Assistant 2's Response Score: ")]
                rating_b = int(x.strip())
                
                if rating_a > rating_b:
                    pred_label = 0
                elif rating_a < rating_b:
                    pred_label = 1
                else:
                    pred_label = 2
        else:
            pred_label = -1
        return pred_label

    def build_prompt(self, dialogue: PairwiseDialogue, eval_model, exchange):
        meta_info = dialogue.get_meta_info()
        template_json = self.get_template_json()[self.get_lang()][eval_model]
        system_message = template_json["system"]
        
        if meta_info["turn"] == 1:
            if eval_model == "judgelm":
                prompt = system_message
                system_message = ""
                prompt += template_json["query"].format(
                    query=dialogue.get_query_by_idx(0)["content"]
                )
                if exchange:
                    prompt += template_json["response_a"].format(
                        response=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                            "content"
                        ]
                    )
                    prompt += template_json["response_b"].format(
                        response=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                            "content"
                        ]
                    )
                else:
                    prompt += template_json["response_a"].format(
                        response=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                            "content"
                        ]
                    )
                    prompt += template_json["response_b"].format(
                        response=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                            "content"
                        ]
                    )
                prompt += template_json["instruction"]
                prompt += template_json["format"]
            else:
                prompt = template_json["instruction"]
                prompt += template_json["query"].format(
                    query=dialogue.get_query_by_idx(0)["content"]
                )
                if exchange:
                    prompt += template_json["response_b"].format(
                        response=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                            "content"
                        ]
                    )
                    prompt += template_json["response_a"].format(
                        response=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                            "content"
                        ]
                    )
                else:
                    prompt += template_json["response_a"].format(
                        response=dialogue.get_pairwise_response_by_idx(0, "model_a")[
                            "content"
                        ]
                    )
                    prompt += template_json["response_b"].format(
                        response=dialogue.get_pairwise_response_by_idx(0, "model_b")[
                            "content"
                        ]
                    )
                prompt += template_json["format"]
        elif meta_info["turn"] == 2:
            prompt = template_json["instruction"]
            prompt += template_json["query"].format(
                query=dialogue.get_query_by_idx(1)["content"]
            )
            if exchange:
                prompt += template_json["response_a"].format(
                    response=dialogue.get_pairwise_response_by_idx(1, "model_b")[
                        "content"
                    ]
                )
                prompt += template_json["response_b"].format(
                    response=dialogue.get_pairwise_response_by_idx(1, "model_a")[
                        "content"
                    ]
                )
            else:
                prompt += template_json["response_a"].format(
                    response=dialogue.get_pairwise_response_by_idx(1, "model_a")[
                        "content"
                    ]
                )
                prompt += template_json["response_b"].format(
                    response=dialogue.get_pairwise_response_by_idx(1, "model_b")[
                        "content"
                    ]
                )
            prompt += template_json["format"]
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
            if "you've provided. [/INST]" in result:
                pos = result.find("you've provided. [/INST]")
                result = result[pos:]
            elif "<|assistant|>\n" in result:
                pos = result.find("<|assistant|>\n")
                result = result[pos:]

            return result
        else:
            return result[0]["generated_text"]
