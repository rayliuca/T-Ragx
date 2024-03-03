import abc
import typing

import datasets
import pandas as pd
import torch
from elasticsearch import Elasticsearch
from elasticsearch import client as elastic_client
from jinja2 import Template as JinjaTemplate
from tqdm.autonotebook import tqdm

from ._utils import get_glossary, file_cacher
from .constants import DEFAULT_GLOSSARY_PARQUET_FOLDER
from ..utils.heuristic import clean_text


class BaseInputProcessor(metaclass=abc.ABCMeta):
    """
    The base input processor, use  huggingface datasets with elastic search for translation memories

    See the ElasticInputProcessor for using with Elasticsearch directly
    """

    def __init__(self,
                 device=None,
                 prompt_template: typing.Optional[typing.Union[str, JinjaTemplate]] = None
                 ):
        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.es_client = None
        self.general_memory_elastic_index = None
        self.general_memory: typing.Optional[datasets.Dataset] = None
        self.task_memory: dict = {}

        self.general_glossary_dict: dict = {}
        self.task_glossary: dict = {}
        self.glossary_parquet_folder = DEFAULT_GLOSSARY_PARQUET_FOLDER

    def load_general_translation(self,
                                 parquet_path,
                                 index_key='ja',
                                 elasticsearch_host: str = "localhost",
                                 elasticsearch_port: int = 9200,
                                 es_client: elastic_client = None,
                                 dataset_args={},
                                 elastic_args={},
                                 elastic_client_args={}
                                 ):
        """
        Load the general translation examples
        """
        self.general_memory = datasets.Dataset.from_pandas(pd.read_parquet(parquet_path), **dataset_args)
        hasher = datasets.fingerprint.Hasher()
        memory_fingerprint = hasher.hash(self.general_memory)

        es_index_name = f"hf_{index_key}_{memory_fingerprint}"

        # initiate the elastic index
        if es_client is None:
            es_client = Elasticsearch(
                elasticsearch_host,  # Elasticsearch endpoint
                port=elasticsearch_port,
                **elastic_client_args
            )

        self.es_client = es_client

        if es_client.indices.exists(index=es_index_name):
            self.general_memory.load_elasticsearch_index(index_key, es_index_name=es_index_name)
        else:
            self.general_memory.add_elasticsearch_index(index_key, es_index_name=es_index_name,
                                                        es_client=es_client, **elastic_args)

        self.general_memory_elastic_index = index_key

        return

    def load_task_translation(self):
        """
        Load the general translation examples
        """
        raise NotImplementedError()
        pass

    def search_general_memory(self, text, search_index: str = None, k=4, max_item_len=500, **search_kwargs):
        """
        search general translation examples using elasticsearch
        """
        if search_index is None:
            search_index = self.general_memory_elastic_index

        mem_scores, mem_indices = self.general_memory.search_batch(search_index, text, k=k, **search_kwargs)

        ref_trans_data = [self.general_memory[midx] for midx in mem_indices]

        # truncate in case the example translations are too long
        processed_output = []
        for rtd, score_list in zip(ref_trans_data, mem_scores):
            wide_output = []
            key_list = list(rtd.keys())
            for i in range(len(rtd[key_list[0]])):
                wide_output.append({
                    k: rtd[k][i][:max_item_len] for k in key_list
                })
                wide_output[-1]['score'] = score_list[i]
            processed_output.append(wide_output)

        return processed_output

    def search_task_memory(self, client=None):
        """
        search in-task translation examples using elasticsearch
        """
        raise NotImplementedError()

    def load_general_glossary(self, glossary_parquet_folder=None, source_lang='ja', target_lang='en', encoding="utf8"):
        """
        Load the general glossary (i.e. wikidata title pair/ dictionary entries )
        format: {
            "original text" : ["translation 1", "translation 2"],
            "original text2" : ["translation 3", "translation 4"],
        }
        """
        if glossary_parquet_folder is not None:
            self.glossary_parquet_folder = glossary_parquet_folder
        self.general_glossary_dict[f"{source_lang}_{target_lang}"] = pd.read_parquet(
            file_cacher(f"{self.glossary_parquet_folder}/{source_lang}_{target_lang}.parquet")).to_dict("index")

        return

    def load_task_glossary(self, glossary_parquet_path, glossary_index):
        # raise NotImplementedError()
        task_glossary_df = pd.read_parquet(file_cacher(glossary_parquet_path))
        clean_index_dict = {k: clean_text(k) for k in task_glossary_df.index}
        task_glossary_df.rename(index=clean_index_dict, inplace=True)

        glossary_dict = task_glossary_df.to_dict("index")

        self.task_glossary[glossary_index] = glossary_dict

        return

    def batch_search_glossary(self, text_list, max_k=10, task_index=None, search_general_glossary=True, max_workers=8,
                              chunksize=1, k=None, source_lang='ja', target_lang='en', pbar=False):
        def _temp_search_glossary(text, max_k=max_k, task_index=task_index,
                                  search_general_glossary=search_general_glossary):
            return self.search_glossary(text, max_k=max_k, task_index=task_index,
                                        search_general_glossary=search_general_glossary, source_lang=source_lang,
                                        target_lang=target_lang)
            pass

        return [_temp_search_glossary(t) for t in tqdm(text_list, disable=(not pbar))]

    def search_glossary(self, text, max_k=10, task_index=None, search_general_glossary=True, k=None, source_lang='ja',
                        target_lang='en'):
        if k is not None:
            max_k = k
        text = clean_text(text)

        found_glossary = {}

        if task_index is not None:
            found_glossary.update(self.search_task_glossary(text, task_index, max_k=max_k, target_lang=target_lang))

        if len(found_glossary) < max_k:
            general_glossary = self.search_general_glossary(text, max_k=max_k - len(found_glossary),
                                                            source_lang=source_lang, target_lang=target_lang)
            for k in general_glossary:
                skip_flag = False
                for existing_key in found_glossary:
                    if k in existing_key:
                        # ignore glossary words being a component of a longer glossary word
                        skip_flag = True
                        break
                if skip_flag:
                    continue

                if k in found_glossary:
                    found_glossary[k] = list(set(found_glossary[k] + general_glossary[k]))
                else:
                    found_glossary[k] = general_glossary[k]

        return found_glossary

    def search_general_glossary(self, text, max_k=10, source_lang='ja', target_lang='en'):
        if f"{source_lang}_{target_lang}" not in self.general_glossary_dict:
            self.load_general_glossary(self.glossary_parquet_folder, source_lang=source_lang, target_lang=target_lang)
        return get_glossary(clean_text(text), self.general_glossary_dict[f"{source_lang}_{target_lang}"], max_k=max_k,
                            lang_code=target_lang, source_lang=source_lang)
        # raise NotImplementedError()

    def search_task_glossary(self, text, task_index, max_k=10, source_lang='ja', target_lang='en'):
        return get_glossary(clean_text(text), self.task_glossary[task_index], max_k=max_k, lang_code=target_lang,
                            source_lang=source_lang)
