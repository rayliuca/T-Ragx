import abc
from typing import Union

import fasttext
import regex
from huggingface_hub import hf_hub_download


class BaseLangDetectModel(metaclass=abc.ABCMeta):
    """
    Heuristic lang detect model by counting the occurrence of Japanese and Chinese characters
    Only supports ja, zh, en outputs
    """

    def __init__(
            self,
            *arg,
            **kwargs
    ):
        pass

    @staticmethod
    def lang_detect(text, filter=False, filter_thrush=0.5) -> Union[str, None]:
        if text is None:
            return None
        ja_text = r"[\p{Katakana}\p{Hiragana}]"
        en_text = r"[a-zA-Z]"
        zh_text = r"\p{han}"

        text_len_dict = {
            'ja': len(regex.findall(ja_text, text)),
            'en': len(regex.findall(en_text, text)),
            'zh': len(regex.findall(zh_text, text))
        }

        max_key = max(text_len_dict, key=text_len_dict.get)

        if filter:
            total_len = len(text)
            if text_len_dict[max_key] / total_len < filter_thrush:
                return None
        return max_key


class FastTextLangDetectModel(BaseLangDetectModel):
    """
    Detect the language model via the Facebook language identification model
    """

    def __init__(self, repo_id="facebook/fasttext-language-identification", filename="model.bin", hf_hub_args={}, *arg,
                 **kwargs):
        super().__init__(*arg, **kwargs)
        fasttext_langid_model_path = hf_hub_download(repo_id=repo_id,
                                                     filename=filename, **hf_hub_args)
        self.model = fasttext.load_model(fasttext_langid_model_path)

    def get_lang(self, text):
        for lang in self.model.predict(text.replace("\n", " "), k=-1)[0]:
            if "_Han" in lang or "_Tibt" in lang:
                return "zh"
            elif "_Latn" in lang:
                return "en"
            elif "_Jpan" in lang:
                return "ja"
        return self.model.predict(text, k=-1)[0].split("__")[-1]
