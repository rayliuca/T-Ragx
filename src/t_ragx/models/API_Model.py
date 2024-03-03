import requests

from .BaseModel import BaseModel
from ._utils import DummyTokenizer


class APIModel(BaseModel):
    tokenizer = DummyTokenizer()
    model = None

    def __init__(self, host='localhost', port='11434', endpoint='/api/generate', model="t_ragx_mistral",
                 protocol="http"):
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.protocol = protocol
        self.model = model
        super().__init__(model_id="Dummy", tokenizer=self.tokenizer, model=model)

    def generate(self, input_text_list, generation_config={}):
        if isinstance(input_text_list, str):
            input_text_list = [input_text_list]

        out_text = []
        for t in input_text_list:
            data = {
                "model": self.model,
                "prompt": t,
                'stream': False,
                **generation_config
            }

            r = requests.post(f"{self.protocol}://{self.host}:{self.port}{self.endpoint}", json=data)
            assert r.status_code == 200, f"Failed to generate {t}"
            out_text.append(r.json()['response'])

        return out_text

    def tokenize(self,
                 text_list=None,
                 *args, **kwargs
                 ):
        return text_list

    @staticmethod
    def clean_output(text):
        return text.strip()

    def process_output(self, model_output, tokenized_input):
        return [self.clean_output(t) for t in model_output]
