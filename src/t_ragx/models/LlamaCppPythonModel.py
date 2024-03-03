import logging

from llama_cpp import Llama

from .BaseModel import BaseModel
from ._utils import DummyTokenizer

logging.getLogger("llama-cpp-python").setLevel(logging.WARNING)


class LlamaCppPythonModel(BaseModel):
    tokenizer = DummyTokenizer()
    model = None

    def __init__(self,
                 repo_id="rayliuca/TRagx-GGUF-NeuralOmniBeagle-7B",
                 filename="*Q4_K_M*",
                 # see https://huggingface.co/rayliuca/TRagx-GGUF-NeuralOmniBeagle-7B/tree/main
                 # for filename formats
                 model=None,
                 chat_format="mistral-instruct",
                 reset_model=True,
                 model_config={'n_ctx': 2048},
                 ):

        self.reset_model = reset_model
        if model is None:
            model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                chat_format=chat_format,
                **model_config
            )
        self.model = model
        super().__init__(model_id="Dummy", tokenizer=self.tokenizer, model=model)

    def generate(self, input_text_list, generation_config={}):
        if isinstance(input_text_list, str):
            input_text_list = [input_text_list]

        default_generation_config = {
            'max_tokens': 100,  # this is the max new token equivalent
            'stop': ["[INST]", "[/INST]", "<s>", "</s>"]
        }

        for k in default_generation_config:
            if k not in generation_config:
                generation_config[k] = default_generation_config[k]

        out_text = []
        for t in input_text_list:
            messages = [{
                "role": "user",
                "content": t
            }]

            if self.reset_model:
                self.model.reset()
            output = self.model.create_chat_completion(
                messages,
                **generation_config
            )

            out_text.append(output['choices'][0]['message']['content'])

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
