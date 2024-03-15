from .meta import Server
from .registry import auto_register
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np


@auto_register("logits")
class LogitsServer(Server):
    eval_model = "logits"

    def __init__(self, config):
        super().__init__(config)

        self.model_id = self.config["model_id"]
        self.path = self.config["path"]
        self.tokenizer = None
        self.model = None

    def lazy_load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path,
            use_fast=False,
            add_bos_token=False,
            model_max_length=4096,
            padding_side="right",
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    def message_warpper(self, query, system, context):
        messages_ids = self.tokenizer(query, return_tensors="pt").input_ids.cuda()
        return messages_ids

    def chat_compeletion(self, query, system, context, temperature, max_tokens):
        if self.model is None:
            self.lazy_load()

        messages_ids = self.message_warpper(query, system, context)
        logits = self.model(input_ids=messages_ids).logits[:, -1].flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[self.tokenizer("A").input_ids[-1]],
                        logits[self.tokenizer("B").input_ids[-1]],
                        logits[self.tokenizer("C").input_ids[-1]],
                        logits[self.tokenizer("D").input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        outputs = {}
        outputs["prediction"] = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        outputs["probs"] = probs.tolist()
        return outputs
