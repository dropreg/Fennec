from transformers import AutoTokenizer
from .registry import auto_register
from .meta import Server
import requests
import json
import random


@auto_register("vllm")
class VllmServer(Server):
    eval_model = "vllm"

    def __init__(self, config):
        super().__init__(config)

        self.model_id = self.config["model_id"]

        if self.config["api_server"] == "openai":
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.config["api_key"] if "api_key" in self.config.keys() else "EMPTY",
                base_url=self.config["api_url"],
            )
        else:
            self.url = self.config["api_url"]
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def message_warpper(self, query, system, context):
        if system is None and context is None:
            return [
                {"role": "user", "content": query},
            ]
        elif system is None and not context is None:
            dialog = []
            for i, string in enumerate(context):
                role = "user" if i % 2 == 0 else "assistant"
                dialog.append({"role": role, "content": string})
            dialog.append({"role": "user", "content": query})
            return dialog
        else:
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ]

    def chat_compeletion(self, query, system, context, temperature, top_p, max_tokens):
        if self.config["api_server"] == "openai":
            return self.chat_openai(query, system, context)
        else:
            return self.chat(query, system, context, temperature, top_p, max_tokens)

    def chat(self, query, system, context, temperature, top_p, max_tokens):
        messages = self.message_warpper(query, system, context)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens if max_tokens else self.config["max_tokens"],
            "temperature": temperature if temperature else self.config["temperature"],
            "top_p": top_p if top_p else self.config["top_p"],
        }
        
        header = {"Content-Type": "application/json"}
        port = random.choice(["8000", "8001", "8002", "8003", "8004", "8005", "8006", "8007"])
        url = self.url.replace("8000", port)
        # url = self.url
        response = requests.post(url, headers=header, data=json.dumps(payload))
        
        if response.ok:
            response = json.loads(response.text)["text"][0]
            if "\n<|assistant|>\n" in response:
                pos = response.rfind("\n<|assistant|>\n")
            elif "###Feedback: [/INST]" in response:
                pos = response.rfind("###Feedback: [/INST]")
            else:
                pos = 0
            return response[pos:]
        else:
            raise Exception("Server Error!")

    def chat_openai(self, query, system, context):
        from openai import OpenAI
        if self.client.api_key == "EMPTY":
            port = random.choice(["8000"])
            url = self.config["api_url"].replace("8000", port)
            client = OpenAI(
                api_key="EMPTY",
                base_url=url,
            )
        else:
            client = self.client
        messages = self.message_warpper(query, system, context)
        output = client.chat.completions.create(
            model=self.model_id,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            max_tokens=self.config["max_tokens"],
        )
        return output.choices[0].message.content
