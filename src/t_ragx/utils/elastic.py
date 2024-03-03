import json
import logging
from hashlib import sha1

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from tqdm.notebook import tqdm

from .heuristic import clean_text, is_noise
from .heuristic import lang_detect as heuristic_lang_detect
from ..processors.constants import DEFAULT_MEMORY_INDEX
from ..models.constants import LANG_BY_LANG_CODE

logger = logging.getLogger("t_ragx")


def index_doc(df, index="translation_memory_demo"):
    """
    Formatted index action generator helper to help upload records to Elasticsearch

    Args:
        df:
        index:

    Returns:

    """
    for record in df.to_dict(orient="records"):
        # pop none
        for k in record:
            if record[k] is None:
                record.pop(k)
        yield ('{ "index" : { "_index" : "%s", "_id": "%s"}}' % (
            index, sha1(record[record['id_key']].encode('utf8')).hexdigest()))
        yield json.dumps(record, default=int)


def upsert_doc(df: pd.DataFrame, index: str = None):
    """
    Formatted upsert action generator helper to help upload records to Elasticsearch

    Args:
        df:
        index:

    Returns:

    """
    if index is None:
        index = DEFAULT_MEMORY_INDEX

    for record in df.to_dict(orient="records"):
        # pop none
        pop_list = []
        for k in record:
            if record[k] is None:
                pop_list.append(k)

        for k in pop_list:
            record.pop(k)
        yield ('{ "update" : {"_index" : "%s", "_id" : "%s", "retry_on_conflict" : 3} }' % (
            index, sha1(record[record['id_key']].encode('utf8')).hexdigest()))
        yield '{ "doc" : %s, "doc_as_upsert" : true }' % json.dumps(record, default=int)


def filter_df(df: pd.DataFrame, source_lang: str = 'ja', lang_cols: list = None):
    if lang_cols is None:
        lang_cols = list(LANG_BY_LANG_CODE.keys())

    lang_cols = list(set(lang_cols).intersection(df.columns))

    df.dropna(subset=lang_cols, how='all', inplace=True)
    df.drop_duplicates(subset=[source_lang], inplace=True)
    df[source_lang] = df[source_lang].apply(clean_text)
    df = df[~df[source_lang].map(is_noise)]
    df.reset_index(drop=True, inplace=True)

    for c in lang_cols:
        df = df[~df[c].str.contains("\n", na=False)]

    for c in lang_cols:
        if c in ['ja', 'zh']:
            str_len = df[c].str.len()
            df = df[
                ((350 > str_len) & (str_len > 4)) | (str_len.isna())
                ]
        elif c in ['en']:
            word_count = df[c].str.split(" ").str.len()
            df = df[
                ((100 > word_count) & (word_count > 3)) | (word_count.isna())
                ]

    for c in lang_cols:
        detected_lang = df[c].apply(heuristic_lang_detect)
        df = df[(c == detected_lang) | (detected_lang.isna())]

    df.reset_index(drop=True, inplace=True)

    return df


def upload_df(df: pd.DataFrame, es_client: Elasticsearch(), id_key: str = 'ja', batch_size: int = 10000,
              index: str = None) -> None:
    """
    upload_df

    Args:
        df:
        es_client:
        id_key: The language column to hash (sha1) as ID. Duplicate records with common id will be merged.
                        id_key should be in df.columns
        batch_size:
        index: Defaulted to be "translation_memory". Should be explicitly set for in-task memories

    Returns:

    """
    df = filter_df(df, source_lang=id_key)
    df['id_key'] = id_key
    if len(df) < 1:
        print("Empty dataset")
        return
    batch_idx = np.array_split(range(len(df)), max(int(len(df) / batch_size), 1))
    for select_idx in tqdm(batch_idx):
        try:
            r = es_client.bulk(upsert_doc(df.iloc[select_idx]), index)  # return a dict
        except:
            raise r


def csv_to_elastic(file_path,
                   id_key='ja',
                   elasticsearch_host: str = "localhost",
                   elasticsearch_port: int = 9200,
                   es_client: Elasticsearch() = None,
                   batch_size=10000,
                   read_csv_config: dict = {},
                   index=None,
                   elastic_client_args: dict = {}):
    """
    Upload a CSV file to Elasticsearch
    The input csv should be parallel texts with the language code as their header
    For example:
        | ja  | en        | zh    |
        |-----|-----------|-------|
        | 例1 | example 1 | 範例1 |
        |     |           |       |
        |     |           |       |


    Args:

        file_path:
        id_key: The language column to hash (sha1) as ID. Duplicate records with common id will be merged.
                        id_key should be in df.columns
        elasticsearch_host:
        elasticsearch_port:
        es_client:
        batch_size:
        read_csv_config:
        index: Defaulted to be "translation_memory". Should be explicitly set for in-task memories
        elastic_client_args:

    Returns:

    """

    if es_client is None:
        es_client = Elasticsearch(
            elasticsearch_host,  # Elasticsearch endpoint
            port=elasticsearch_port,
            **elastic_client_args
        )

    df = pd.read_csv(file_path, **read_csv_config)
    assert len(df.columns) > 1, "The CSV file has only one column"

    if len(set(df.columns).intersection(LANG_BY_LANG_CODE.keys())) < 2:
        logger.warning(f"The columns of the CSV are {df.columns}")

    upload_df(df, es_client, id_key=id_key, batch_size=batch_size, index=index)
