import logging
from operator import itemgetter
from typing import List, Union

import torch
from Levenshtein import distance
from elasticsearch import Elasticsearch
from elasticsearch import client as elastic_client
from tqdm.autonotebook import tqdm

from .BaseInputProcessor import BaseInputProcessor
from ..utils.heuristic import clean_text

logger = logging.getLogger("t_ragx")


def rerank_elastic_result(elastic_result, source_lang, search_term, top_k=5):
    if len(elastic_result) < 1:
        return []
    top_score = None
    result_list = []
    for r in elastic_result['hits']['hits']:
        if top_score is None:
            top_score = r['_score']
        if r['_score'] < top_score and len(result_list) > top_k:
            break

        if source_lang not in r['_source']:
            continue
        r['distance'] = distance(r['_source'][source_lang], search_term)
        result_list.append(r)

    result_list = sorted(result_list, key=itemgetter('distance'))
    return result_list[:top_k]


def search_single_elastic(es_client, index, search_term, source_lang, target_lang, top_k=10, request_timeout=50,
                          task_index=None, task_boost=1.2):
    index_list = [index]
    indices_boost = []
    if task_index is not None:
        index_list.append(task_index)
        indices_boost.append({
            task_index: task_boost
        })

    resp = es_client.search(
        index=index_list,
        body={
            "size": top_k,
            "indices_boost": indices_boost,
            "_source": {
                "includes": [source_lang, target_lang, 'source']
            },
            "query": {
                "bool": {
                    "must": [
                        {
                            "query_string": {
                                "query": search_term,
                                "fields": [
                                    source_lang
                                ],
                                "escape": True
                            }
                        },
                    ],
                    "filter": [
                        {"exists": {"field": target_lang}},
                    ]
                }
            }
        },
        request_timeout=request_timeout
    )
    return resp


def search_elastic_with_retry(es_client, index, search_term, source_lang, target_lang, top_k=10, retry=3,
                              task_index=None, task_boost=1.2):
    for i in range(retry):
        try:
            return search_single_elastic(es_client, index, search_term, source_lang, target_lang, top_k=top_k,
                                         task_index=task_index, task_boost=task_boost)
        except:
            pass
    logger.warning("elastic time out")
    return []


def batch_search_elastic(es_client, index, search_term_list, source_lang, target_lang, top_k=10, rerank_top_k=5,
                         pbar=False, task_index=None, task_boost=1.2, max_item_len=-1):
    bulk_result = []
    for search_term in tqdm(search_term_list, disable=(not pbar)):
        search_result = search_elastic_with_retry(es_client, index, search_term, source_lang, target_lang, top_k=top_k,
                                                  task_index=task_index, task_boost=task_boost)

        # truncate if the max_item_len variable is set
        for r in search_result['hits']['hits']:
            r['_source'][source_lang] = r['_source'][source_lang][:max_item_len]
            r['_source'][target_lang] = r['_source'][target_lang][:max_item_len]

        bulk_result.append(search_result)

    return [
        rerank_elastic_result(resp, source_lang, search_term, top_k=rerank_top_k) for search_term, resp in
        zip(search_term_list, bulk_result)
    ]


class ElasticInputProcessor(BaseInputProcessor):
    """
    The default input processor, rely on an elastic search database and pre-computed indexes
    """

    def __init__(self,
                 device=None,
                 ):
        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__()

    def load_general_translation(self, elastic_index="translation_memory", elasticsearch_host: str = "localhost",
                                 elasticsearch_port: int = 9200, es_client: elastic_client = None, elastic_args={},
                                 elastic_client_args={}, **kwargs):
        """
        Load the general translation examples
        """

        # initiate the elastic index
        if es_client is None:
            es_client = Elasticsearch(
                elasticsearch_host,  # Elasticsearch endpoint
                port=elasticsearch_port,
                **elastic_client_args
            )

        self.es_client = es_client

        assert es_client.indices.exists(index=elastic_index)

        self.general_memory_elastic_index = elastic_index

        return

    def search_general_memory(self, *args, **kwargs):
        return self.search_memory(*args, **kwargs)

    def search_memory(self, text_list: Union[List[str], str], search_index: str = None, source_lang='ja',
                      target_lang='en', top_k=10,
                      rerank_top_k=None, max_item_len=500, pbar=False, task_index=None, task_boost=1.2,
                      **search_kwargs):
        """
        search general translation examples using elasticsearch
        """
        if isinstance(text_list, str):
            text_list = [text_list]

        text_list = [clean_text(t) for t in text_list]
        if search_index is None:
            search_index = self.general_memory_elastic_index

        if rerank_top_k is None:
            rerank_top_k = top_k

        search_result_list = batch_search_elastic(self.es_client, search_index, text_list, source_lang, target_lang,
                                                  top_k=top_k, rerank_top_k=rerank_top_k, pbar=pbar,
                                                  task_index=task_index, task_boost=task_boost,
                                                  max_item_len=max_item_len)

        processed_output = [[{'score': r['_score'], 'distance': r['distance']} | r['_source'] for r in search_result]
                            for search_result in search_result_list]

        # "normed_distance" is the Levenshtein
        for t, l in zip(text_list, processed_output):
            for d in l:
                d['normed_distance'] = d['distance'] / len(t)

        return processed_output
