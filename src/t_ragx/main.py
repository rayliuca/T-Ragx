from typing import List, Union

import numpy as np
from tqdm.auto import tqdm

from .processors import ElasticInputProcessor
from .processors import BaseInputProcessor
from .models.AggregationModel import CometAggregationModel
from .models.BaseModel import BaseModel


class TRagx:
    """
    🦖🦖🦖
    Translation using LLM with Retrieval Augmented Generation (RAG)

    """
    aggregate_model = None

    def __init__(self,
                 generation_models: Union[BaseModel, List[BaseModel]],
                 aggregate_model=None,
                 input_processor: BaseInputProcessor = None,
                 ):
        """

        Args:
            generation_models: a list of the T-Ragx models
            aggregate_model: a model that would choose the best translation without references. Currently only support
                                t_ragx.models.AggregationModel.CometAggregationModel
            input_processor: a T-Ragx input processor
        """

        self.input_processor = input_processor
        if input_processor is None:
            self.input_processor = ElasticInputProcessor()

        self.generation_models = generation_models
        if not isinstance(generation_models, list):
            self.generation_models = [generation_models]

        if aggregate_model is not None:
            self.aggregate_model = aggregate_model
        else:
            if len(self.generation_models) > 1:
                self.aggregate_model = CometAggregationModel()

    def __call__(self, *args, **kwargs):
        return self.translate(*args, **kwargs)

    def translate(self, text, pre_text: list = None,
                  search_glossary=True, search_memory=True,
                  memory_search_args: dict = None,
                  glossary_search_args: dict = None,
                  prompt_args: List[dict] = None,
                  generation_args: List[dict] = None):
        pass

    def batch_translate(self,
                        text_list,
                        pre_text_list: list = None,
                        batch_size=1,
                        source_lang_code='ja',
                        target_lang_code='en',
                        search_glossary=True,
                        search_memory=True,
                        memory_search_args: dict = None,
                        glossary_search_args: dict = None,
                        tokenize_args: List[dict] = None,
                        prompt_args: List[dict] = None,
                        generation_args: List[dict] = None
                        ):

        if pre_text_list is None:
            pre_text_list = [None] * len(text_list)

        if memory_search_args is None:
            memory_search_args = {}
        if glossary_search_args is None:
            glossary_search_args = {}

        if prompt_args is None:
            prompt_args = [{}] * len(self.generation_models)

        if generation_args is None:
            generation_args = [{}] * len(self.generation_models)

        if tokenize_args is None:
            tokenize_args = [{}] * len(text_list)

        if generation_args is None:
            generation_args = [{}] * len(text_list)

        memory_results = [[]] * len(text_list)
        if search_memory:
            memory_results = self.input_processor.search_memory(text_list, **memory_search_args)

        glossary_results = [[]] * len(text_list)
        if search_glossary:
            glossary_results = self.input_processor.batch_search_glossary(text_list, **glossary_search_args)

        generation_output_dict = {}
        for model_idx, generation_model, p_args, tok_args, gen_args in zip(
                range(len(self.generation_models)),
                self.generation_models,
                prompt_args,
                tokenize_args,
                generation_args
        ):
            translated_text_list = []
            for batch_idx in tqdm(
                    np.array_split(list(range(len(text_list))), int(max(len(text_list) / batch_size, 1)))
            ):
                batch_text = [text_list[i] for i in batch_idx]
                batch_pre_text = [pre_text_list[i] for i in batch_idx]

                batch_search_result = [
                    {
                        'memory': memory_results[i],
                        'glossary': glossary_results[i],
                    }
                    for i in range(len(batch_idx))
                ]

                translated_text_list += generation_model.batch_translate(
                    batch_text,
                    source_lang_code=source_lang_code,
                    target_lang_code=target_lang_code,
                    batch_search_result=batch_search_result,
                    batch_pre_text=batch_pre_text,
                    tokenize_config=tok_args,
                    generation_config=gen_args
                )
            generation_output_dict[model_idx] = translated_text_list

        generation_output = generation_output_dict[0]
        if len(generation_output_dict) > 1:
            generation_output = self.aggregate_model.combine_preds(
                generation_output_dict, batch_text, target_lang_code=target_lang_code
            )
        return generation_output
