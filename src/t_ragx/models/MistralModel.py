import re

from .BaseModel import BaseModel


class MistralModel(BaseModel):
    tokenizer = None
    model = None

    def __init__(self, model_id="rayliuca/TRagx-Mistral-7B-Instruct-v0.2", adapter=None, tokenizer=None, model=None):
        super().__init__(model_id=model_id, adapter=adapter, tokenizer=tokenizer, model=model)

    def tokenize(self,
                 text_list=None,
                 tokenize_config=None
                 ):

        if text_list is None:
            text_list = []
        if tokenize_config is None:
            tokenize_config = {}

        default_tokenize_config = {
            'pad_to_multiple_of': 8,
            'padding': True,
            'truncation': True,
            'max_length': 2000,
            'return_tensors': 'pt',
            'add_special_tokens': False
        }
        for k in default_tokenize_config:
            if k not in tokenize_config:
                tokenize_config[k] = default_tokenize_config[k]

        return self.tokenizer.batch_encode_plus(text_list, **tokenize_config).to(self.model.device)

    @staticmethod
    def clean_output(text):
        special_tok_q = " ?/[//INST/] ?"
        return re.sub(special_tok_q, "", text.replace("//", "")).strip()

    def process_output(self, model_output, tokenized_input):
        translation_outputs = [
            o[len(i):]
            for o, i in zip(model_output.cpu().numpy(), tokenized_input['input_ids'].cpu().numpy())
        ]

        decoded_outputs = self.tokenizer.batch_decode(translation_outputs, skip_special_tokens=True)
        decoded_outputs = [
            self.clean_output(o) for o in decoded_outputs
        ]
        return decoded_outputs
