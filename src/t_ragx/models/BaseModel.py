import abc

from transformers import AutoTokenizer, AutoModelForCausalLM

from .constants import LANG_BY_LANG_CODE


def pretext_to_text(pretext_list, max_sent=5):
    if pretext_list is None or len(pretext_list) < 1:
        return ""

    out_text = "Preceding text:\n"
    for source_text in pretext_list[:max_sent]:
        out_text += f"  {source_text}\n"
    return out_text + "\n"


def glossary_to_text(glossary):
    out_text = "Relevant Dictionary records:\n"
    for source_text in glossary:
        out_text += f"  {source_text}: {', '.join(glossary[source_text])}\n"
    return out_text


def trans_mem_to_text(trans_mem: list, source_lang_code='ja', target_lang_code='en'):
    if len(trans_mem) < 1:
        return ""
    out_text = "Examples translations:\n"
    count = 1
    for row in trans_mem:
        out_text += f""" {count}. \n   {row[source_lang_code]}\n   {row[target_lang_code]}\n"""
        count += 1
    return out_text


class BaseModel(metaclass=abc.ABCMeta):
    tokenizer = None
    model = None

    def __init__(self, model_id, adapter=None, tokenizer=None, model=None):

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                padding_side='left',
                truncation_side='left',
            )

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
                tokenizer.pad_token = tokenizer.unk_token

        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
            if adapter is not None:
                if isinstance(adapter, list):
                    for a in adapter:
                        model.load_adapter(a)
                elif isinstance(adapter, str):
                    model.load_adapter(adapter)
                else:
                    ValueError("the adapter parameter must be either string or a list of strings")

            model = model.eval()

        self.model = model
        self.tokenizer = tokenizer

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

    def generate(self, tokenized_input, generation_config=None):
        if generation_config is None:
            generation_config = {
                'max_new_tokens': 100,
                'early_stopping': True,
                'eos_token_id': [self.tokenizer.eos_token_id],
                'pad_token_id': self.tokenizer.eos_token_id
            }

        for k in tokenized_input:
            tokenized_input[k] = tokenized_input[k].to(self.model.device)

        return self.model.generate(**tokenized_input, **generation_config)

    @staticmethod
    def clean_output(text):
        raise NotImplementedError

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

    def batch_translate(self, batch_text: list,
                        source_lang_code="ja",
                        target_lang_code="en",
                        batch_search_result: list = None,
                        batch_pre_text: list = None,
                        tokenize_config=None,
                        generation_config=None
                        ):

        query_prompts = self.batch_build_prompt(
            text=batch_text,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            pre_text_list=batch_pre_text,
            search_result=batch_search_result
        )

        token_data = self.tokenize(query_prompts, tokenize_config)
        generation_output = self.generate(token_data, generation_config)
        translated_output = self.process_output(generation_output, token_data)
        return translated_output

    def translate(self, text: str,
                  source_lang_code="ja",
                  target_lang_code="en",
                  search_result: list = None,
                  pre_text: list = None,
                  tokenize_config=None,
                  generation_config=None
                  ):
        batch_text = [text]
        batch_pre_text = [pre_text]
        batch_search_result = [search_result]

        return self.batch_translate(
            batch_text,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            batch_search_result=batch_search_result,
            batch_pre_text=batch_pre_text,
            tokenize_config=tokenize_config,
            generation_config=generation_config
        )[0]

    def batch_build_prompt(self,
                           text: list,
                           source_lang_code="Japanese",
                           target_lang_code="English",
                           search_result: list = None,
                           pre_text_list: list = None,
                           ):

        if pre_text_list is not None:
            assert len(pre_text_list) == len(text)
        else:
            pre_text_list = [None] * len(text)

        if search_result is not None:
            assert len(search_result) == len(text)
        else:
            search_result = [None] * len(text)

        return [
            self.build_prompt(
                t,
                source_lang_code=source_lang_code,
                target_lang_code=target_lang_code,
                search_result=sr,
                pre_text=pt
            )
            for t, sr, pt in zip(text, search_result, pre_text_list)
        ]

    def build_prompt(self,
                     text,
                     source_lang_code="ja",
                     target_lang_code="en",
                     search_result=None,
                     pre_text: list = None
                     ):
        source_lang = LANG_BY_LANG_CODE[source_lang_code]
        target_lang = LANG_BY_LANG_CODE[target_lang_code]
        if search_result is None:
            search_result = {'glossary': [], 'memory': []}

        chat = [
            {"role": "user", "content": (
                "As a large language model, you are a trained expert in multiple languages. "
                "These are some references that might help you translating passages:\n"
                f"{glossary_to_text(search_result['glossary'])}{pretext_to_text(pre_text)}{trans_mem_to_text(search_result['memory'], source_lang_code=source_lang_code, target_lang_code=target_lang_code)}"
                f"Translate this {source_lang} passage to {target_lang} "
                "without additional questions, disclaimer, or explanations, but accurately and completely:"
                f"{text}"
            )},
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, )
