🦖 T-Ragx
==============================

<p align="center">
  <picture>
    <img alt="T-Ragx Featured Image" src="https://l8u0.c18.e2-1.dev/t-ragx-public/assets/featured_repo.png" height="300" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>


<h3 align="center">
    <p>Enhancing Translation with RAG-Powered Large Language Models</p>
</h3>

<br>

T-Ragx Demo: <a target="_blank" href="https://colab.research.google.com/github/rayliuca/T-Ragx/blob/main/examples/T_Ragx_Demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

🚧 T-Ragx Colab Tool [colab tool]🚧

## TL;DR

### Overview

- Democratize high-quality machine translations
- Open-soured system-level translation framework
- Fluent/ natural translations using LLMs
- Private and secure local translations
- Zero-shot in-task translations

### Methods

- QLoRA fine-tuned models
- General + in-task translation memory/ glossary
- Include preceding text for document-level translations for additional context

### Results

- QLoRA + in-task translation memory and glossary achieved a significant (~45%) increase in aggregated WMT23 translation scores on the Mistral 7b Instruct model 
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
Simply run:

`pip install t-ragx`

or if you are feeling lucky:

`pip install git+https://github.com/rayliuca/T-Ragx.git`


### Elasticsearch

See the [wiki page instructions](https://github.com/rayliuca/T-Ragx/wiki/Getting-Started#install-elasticsearch)


Note: you can access read-only preview T-Ragx Elasticsearch services at `https://t-ragx-fossil.rayliu.ca` and `https://t-ragx-fossil2.rayliu.ca`
(But you will need a personal Elasticsearch service to add your in-task memories)



#### Environment
##### (Recommended) Conda / Mamba
Download the conda [`environment.yml` file](environment.yml) and run:

```bash
conda env create -f environment.yml

## or with mamba
# mamba env create -f environment.yml
```

Which will crate a `t_ragx` environment that's compatible with this project

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
input_processor.load_general_translation(elasticsearch_host=["https://t-ragx-fossil.rayliu.ca", "https://t-ragx-fossil2.rayliu.ca"])
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



