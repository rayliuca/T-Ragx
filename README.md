🦖 T-Ragx
==============================

<p align="center">
  <picture>
    <img alt="T-Ragx Featured Image" src="assets/featured_repo.png" height="300" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>


<h3 align="center">
    <p>Enhancing Translation with RAG-Powered Large Language Models</p>
</h3>


🚧 T-Ragx Demo [colab demo]🚧 

🚧 T-Ragx Colab Tool [colab tool]🚧

## TL;DR

### Overview

- Democratize high-quality machine translations
- Open-soured system-level translation framework
- Fluent/ natural translations using LLM
- Private and secure local translations
- Zero-shot in-task translations

### Methods

- QLoRA fine-tuned models
- General + in-task translation memory/ glossary
- Include preceding text for document-level translations for additional context

### Results

- In-task translation memory and glossary achieved a significant (~45%) increase in aggregated translation scores on the QLoRA Mistral 7b model 
- Great recall for valid translation memory/ glossary (i.e. previous translations/ character names)
- Outperforms native TowerInstruct on all 6 language directions finetuned (Ja x Zh x En)
- Outperforms DeepL in translating Japanese web novel (That Time I Got Reincarnated as a Slime) to Chinese with in-task
  memories
    - Japanese -> Chinese
        - +29% by sacrebleu
        - +0.4% by comet22

[//]: # (### 🚧 [***See the write-up for more details***]&#40;reports/README.md&#41; 🚧)

## Getting Started

### Install
#### Environment
##### Conda / Mamba (Recommended)
```python

```

##### pip
Use your favourite virtual environment, and run:

`pip install -r requirment.txt`


### Examples

Initiate the input processor:

```Python
import t_ragx

# Initiate the input processor which will retrieve the memory and glossary results for us
input_processor = t_ragx.Processors.ElasticInputProcessor()

# Load/ point to the demo resources
input_processor.load_general_glossary("https://l8u0.c18.e2-1.dev/t-ragx-public/glossary")
input_processor.load_general_translation(elasticsearch_host="t-ragx-fossil.rayliu.ca", elasticsearch_port=80)
```

Using the `llama-cpp-python` backend:

```python
import t_ragx

# T-Ragx currently support 
# Huggingface transformers: MistralModel, InternLM2Model
# Ollama API: OllamaModel
# OpenAI API: OpenAIModel
# Llama-cpp-python backend: LlamaCppPythonModel
mistral_model = t_ragx.models.LlamaCppPythonModel(
    repo_id="rayliuca/TRagx-GGUF-Mistral-7B-Instruct-v0.2",
    filename="*Q4_K_M*",
    # see https://huggingface.co/rayliuca/TRagx-GGUF-Mistral-7B-Instruct-v0.2
    # for other files
    chat_format="mistral-instruct",
    model_config={'n_ctx':2048}, # increase the context window
)

t_ragx_translator = t_ragx.TRagx([mistral_model], input_processor=input_processor)
```

Translate!

```python
t_ragx_translator.batch_translate(
    source_text_list,  # the input text list to translate
    pre_text_list=pre_text_list,  # optional, including the preceding context to translate the document level
    # Can generate via:
    # pre_text_list = t_ragx.utils.helper.get_preceding_text(source_text_list, max_sent=3)
    source_lang_code='ja',
    target_lang_code='en',
    memory_search_args={'top_k': 3}  # optional, pass additional arguments to input_processor.search_memory
)
```

## Data Sources

|                           Dataset                           | Translation Memory |  Glossary  | Training | Testing |                                            License                                             |
|:-----------------------------------------------------------:|:------------------:|:----------:|:--------:|:-------:|:----------------------------------------------------------------------------------------------:|
|                         OpenMantra                          |         ✅          |            |    ✅     |         | [CC BY-NC 4.0](https://github.com/mantra-inc/open-mantra-dataset?tab=License-1-ov-file#readme) |
|                         WMT < 2023                          |         ✅          |            |    ✅     |         |           [for research](https://www2.statmt.org/wmt23/translation-task.html#_data)            |
|                           ParaMed                           |         ✅          |            |    ✅     |         |                  [cc-by-4.0](https://huggingface.co/datasets/bigbio/paramed)                   |
|                       ted_talks_iwslt                       |         ✅          |            |    ✅     |         |                   [cc-by-nc-nd-4.0](https://nlp.stanford.edu/projects/jesc/)                   |
|                            JESC                             |         ✅          |            |    ✅     |         |                    [CC BY-SA 4.0](https://nlp.stanford.edu/projects/jesc/)                     |
|                            MTNT                             |         ✅          |            |          |         |           [Custom/ Reddit API](https://pmichel31415.github.io/mtnt/index.html#licen)           |
|                           WCC-JC                            |         ✅          |            |    ✅     |         |   [for research](https://github.com/zhang-jinyi/Web-Crawled-Corpus-for-Japanese-Chinese-NMT)   |
|                            ASPEC                            |                    |            |    ✅     |         |              [custom, for research](https://jipsti.jst.go.jp/aspec/terms_en.html)              |
|               All other ja-en/zh-en OPUS data               |         ✅          |            |          |         |                       mix of open licenses: check https://opus.nlpl.eu/                        |
|                          Wikidata                           |                    |     ✅      |          |         |                    [CC0](https://www.wikidata.org/wiki/Wikidata:Copyright)                     |
|             Tensei Shitara Slime Datta Ken Wiki             |                    | ☑️ in task |          |         |                          [CC BY-SA](https://www.fandom.com/licensing)                          |
|                          WMT 2023                           |                    |            |          |    ✅    |           [for research](https://www2.statmt.org/wmt23/translation-task.html#_data)            |
| Tensei Shitara Slime Datta Ken Web Novel & web translations |     ☑️ in task     |            |          |    ✅    |                                Not included translation memory                                 |

## Elasticsearch

Note: you can access a read-only preview T-Ragx Elasticsearch service at `http://t-ragx-fossil.rayliu.ca:80`
(But you will need a personal Elasticsearch service to add your in-task memories)

### Install using Docker

See the [T-Rex-Fossil](https://github.com/rayliuca/T-Ragx-Fossil) repo

### Install Locally

Note: this project was built with Elasticsearch 7

1. Download the Elasticsearch binary
2. Unzip
3. Enter into the unzipped folder
4. Install the plugins

```bash
bin/elasticsearch-plugin install repository-s3
bin/elasticsearch-plugin install analysis-icu
bin/elasticsearch-plugin install analysis-kuromoji
bin/elasticsearch-plugin install analysis-smartcn
```

5. Add the S3 keys

   (The snapshot is stored on IDrive e2 which is apparently not compatible with S3 enough for read-only Elastic S3 repo
   to work)

   This read-only key will help you connect to the snapshot

```bash
bin/elasticsearch-keystore add s3.client.default.access_key
CG4KwcrNPefWdJcsBIUp

bin/elasticsearch-keystore add s3.client.default.secret_key
Cau5uITwZ7Ke9YHKvWE9cXuTy5chdapBLhqVaI3C
```

6. Add the snapshot

```bash
curl -X PUT "http://localhost:9200/_snapshot/public_t_ragx_translation_memory" -H "Content-Type: application/json" -d "{\"type\":\"s3\",\"settings\":{\"bucket\":\"t-ragx-public\",\"base_path\":\"elastic\",\"endpoint\":\"o3t0.or.idrivee2-37.com\"}}"
```

Note: this is the JSON body:

```json
{
  "type": "s3",
  "settings": {
    "bucket": "t-ragx-public",
    "base_path": "elastic",
    "endpoint": "o3t0.or.idrivee2-37.com"
  }
}
```

7. Restore the Snapshot

   If you use any GUI client i.e. elasticvue, you likely could do this via their interface



