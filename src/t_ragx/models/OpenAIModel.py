from openai import OpenAI

from .API_Model import APIModel
from ._utils import DummyTokenizer as BaseDummyTokenizer


class DummyTokenizer(BaseDummyTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_chat_template(self, conversation, *args, **kwargs):
        return conversation


class OpenAIModel(APIModel):
    tokenizer = DummyTokenizer()
    model = None

    def __init__(self, host='localhost', port=11434, endpoint='/v1', model="t_ragx_mistral",
                 protocol="http", api_key='ollama'):
        super().__init__(host=host, port=port, endpoint=endpoint, model=model,
                         protocol=protocol)

        self.openai_client = OpenAI(
            base_url=f"{self.protocol}://{self.host}:{self.port}{self.endpoint}",
            api_key=api_key,
        )

    def generate(self, input_chat_list, generation_config={}):
        out_text = []
        for chat in input_chat_list:
            chat_completion = self.openai_client.chat.completions.create(
                messages=chat,
                model=self.model,
                **generation_config
            )
            out_text.append(chat_completion.choices[0].message.content.strip())

        return out_text
