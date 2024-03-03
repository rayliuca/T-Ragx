import json
import urllib

import pandas as pd
from elasticsearch import Elasticsearch

from ..utils.elastic import upload_df

INDEX = "translation_memory_demo"
DEMO_MEMORY_FOLDER_URL = "https://l8u0.c18.e2-1.dev/t-ragx-public/memory/demo"

es_client = Elasticsearch()

data_index_url = f"{DEMO_MEMORY_FOLDER_URL}/index.json"
response = urllib.request.urlopen(data_index_url)
assert response.status == 200, "Cannot read the DEMO memory"

data_index = json.loads(response.read())

for data_file in data_index:
    file_url = f"{DEMO_MEMORY_FOLDER_URL}/{data_file['file_name']}"
    df = pd.read_parquet(file_url)
    upload_df(df, es_client, id_key=data_file['source_lang'])
