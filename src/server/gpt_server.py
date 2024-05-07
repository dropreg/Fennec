from .registry import auto_register
from .meta import Server
import openai


@auto_register("gpt")
class GPTServer(Server):
    eval_model = "gpt"

    def __init__(self, config):
        super().__init__(config)

        self.model_id = self.config["model_id"]
        self.api_url = self.config["api_url"]
        self.api_key = self.config["api_key"]

    def message_warpper(self, system, query):
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

    def chat_compeletion(
        self,
        query,
        system,
        context,
        temperature,
        top_p,
        max_tokens,
    ):
        openai.api_base = self.api_url
        openai.api_key = self.api_key

        messages = self.message_warpper(system, query)
        
        # try:
        response = openai.ChatCompletion.create(
            model=self.model_id,
            messages=messages,
            # n=1,
        )
        if "choices" in response and response["choices"] is None:
            output = ""
            print(response)
            raise Exception('Choices is None')
        else:
            output = response["choices"][0]["message"]["content"]
        if output.strip().startswith("Requests to the ChatCompletions_Create Operation under Azure OpenAI API version"):
            raise Exception('Azure OpenAI API failed')
        if output.strip().startswith("Too many requests. Please try again later."):
            raise Exception('Too many requests. Please try again later.')
        # except:
            # output = "server error"

        return output
