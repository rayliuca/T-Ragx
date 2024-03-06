Project Write Up
======  

Loose write-up for some experiments, observations, and thoughts

## Intro


Large language models (LLMs) have significantly transformed machine translation (MT) for the general population, producing outputs that are more fluent compared to the traditionally awkward phrasing associated with earlier MT services like Google Translate. Notable advancements have previously been achieved with neural-network-based machine translation models and services such as T5, Opus, NLLB, and especially, DeepL. DeepL has gained substantial popularity among web novel enthusiasts due to its ability to deliver more natural translations. The advent of ChatGPT and similar LLM-based services/models has further enhanced the accessibility and naturalness of machine translation.

However, a significant limitation of these models is their lack of persistent memory concerning the literature they translate. This deficiency can lead to sub-optimal translations, such as inconsistent naming for the same character across different sentences. Traditionally, human translators addressed this issue by establishing a set of translation glossaries, ensuring uniformity in the translation of all proper nouns. Additionally, the creation of a translation memory for each project can facilitate time efficiency and help maintain a consistent voice throughout the translated material.

## Experiments


### Datasets

Caveat: all of the Chinese texts (input text, prediction, and reference text) were converted to Traditional Chinese  using [OpenCC](https://github.com/BYVoid/OpenCC) with the `s2tw.json` configuration

#### Translation Memory & Glossary

[Several open datasets](../README.md#data-sources) were combined to build the general translation memory.

##### General

Translation Memory

- A mix of open-source parallel corpus (see the table above)

Glossary

- Wiki-data (https://huggingface.co/datasets/rayliuca/WikidataLabels)

##### In-task (Reincarnated Slime)

Translation Memory

- Chapters not in the test set

Glossary

- Japanese -> English

    - Scraped from [Fandom wiki](https://tensura.fandom.com/wiki/That_Time_I_Got_Reincarnated_as_a_Slime)

- Japanese -> Chinese

    - Scraped from Wikipedia

#### Training

The quality of the training data has proven to be crucial in experimental outcomes. While valuable, datasets such as JESC and MTNT were not aligned as accurately as the WMT test set. Consequently, the final training dataset primarily consists of human-curated datasets such as WMT, OpenMantra and ASPEC. These are supplemented with machine-aligned datasets including JESC, ted_talks_iwslt, and WCC-JC, which have slightly higher levels of noise or mismatch.

The processed training set incorporated the strategy of including up to three preceding sentences where available (notably in the WMT dataset), introducing a balanced mix of translation examples. This mix entails an equal probability (25%) of presenting no translation example, one translation example, three translation examples, or the correct translation output (an intentional data leak to mimic succesful TM retrivel). Furthermore, the training set facilitates bi-directional translation across three languages, yielding six directions of translation coverage. In total, the dataset encompasses approximately 80,000 sentences, aiming to optimize the model's understanding and generation capabilities across various context availabilities.

#### Evaluation

To evaluate against possible use cases and promote comparisons with existing models, T-Ragx was evaluated on the

- That Time I Got Reincarnated as a Slime (Tensei Shitara Slime Datta Ken)

    - ja -> en

    - ja -> zh

- WMT23 test sets with language pair

    - ja <-> en
    - zh <-> en

The Reincarnated Slime was selected for its comprehensive and detailed fandom wiki page, which proved to be invaluable in constructing a glossary. Its popularity and status as a completed series further justified its choice for analysis.

The test set was triply aligned among the Japanese source + Chinese, and English translations utilizing [sentence-transformer](https://www.sbert.net/) with the `paraphrase-multilingual-MiniLM-L12-v2` model. This alignment was refined by filtering out entries with scores below `0.4`. The chapters selected for the test set, `[224, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 247]`, were determined based on the availability of translations. The preference for later chapters aims to mirror the practical scenario of employing T-Ragx for computer-assisted translation tasks.

Additionally, the WMT23 test set is comprised of human-translated sentences, ensuring a high standard of quality and reliability for comparative analysis.

### Method

- Elasticsearch for translation memory \(TM\), with the classical Levenshtein distance for reranking
    - Experimented with vector search (Weaviate, HuggingFace Dataset + Faiss)
        - Results were more relevant to semantic meaning rather than text surface
        - Not very effective as translation memories
    - Elasticsearch is fast and scalable
        - Capable of searching datasets larger than available RAM
        - Much, much  faster than MySQL with BM25

- General and task-specific glossaries using heuristics
    - For example, regex or simple substring search
    - Experimented with entity extraction using `hanlp`
        - Slow and prone to high false negative rates

- Incorporating preceding text in context
    - May increase accuracy or improve flow in some cases

- Q-LoRA fine-tuned for translation
    - Optional inputs for TM, glossary, and preceding text
    - Loss calculated only on the translation output
        - Batched short inputs without truncation
        - Utilized the unknown token as the padding token, if not set for the particular model
        - Batching did not significantly affect model quality
        - Training was significantly faster with batching
    - Trained for 2 epochs
        - Note: InterLM2 experienced catastrophic forgetting after approximately 1 epoch, hence trained only for 1 epoch

- Ensemble approach
    - Incorporated the closest TM result
    - Employed multiple models
    - Used `Unbabel/wmt22-cometkiwi-da` (reference-free) for scoring
    - Marginal improvement

## Results
*Note: The detailed evaluation tables for each task are also available in this directory. Check them out if you only care about one of the language directions*

Without fine-tuning, LLMs may produce additional text. For instance, the native Mistral instruct v0.2 model, in the absence of Retrieval-Augmented Generation (RAG), generated the following translation for a sentence from the WMT23 dataset:

> If you're a member of Furumaru (a Japanese term that could refer to various companies or services, and "Furumaru-kaiin" means "Furumaru member"), shipping costs are usually waived.
>  
>  Ja -> En #35

While providing additional context might be beneficial for chatbots, it poses a challenge for MT models by introducing unrelated content. Integrating translation examples as part of the RAG framework has been shown to mitigate this issue. This aligns with the practices of one-shot or few-shot learning discussed in the original [ALMA](https://github.com/fe1ixxu/ALMA) [paper](https://arxiv.org/abs/2309.11674). Moreover, LLMs that are specifically tailored for translation tasks, such as TowerInstruct, demonstrate significantly improved performance in avoiding such behavior. For examples of TowerInstruct's translations, refer to this [folder](supplemental/preds/official_tower).

Table 1: Mistral 7B model, average sacrebleu and comet22 scores on the 4 WMT23 (Ja <-> EN, Zh <-> En) test sets

<table>  
<thead>  
<tr>  
<th>Model</th>  
<th colspan="2">sacrebleu</th>  
<th colspan="2">comet22</th>  
</tr>  
</thead>  
<tbody>  
<tr>  
<td>Mistral 7B Inst.</td>  
<td align="center" colspan="2">11.168 </td>  
<td align="center" colspan="2">0.394 </td>  
</tr>  
<tr>  
<td>+RAG</td>  
<td>11.029 (-0.139) </td>  
<td> </td>  
<td>0.394 (+0)</td>  
<td> </td>  
</tr>  
<tr>  
<td>+Preceding Text </td>  
<td>11.050 (+0.021)</td>  
<td> </td>  
<td>0.395 (+0.001)</td>  
<td> </td>  
</tr>  
<tr>  
<td>QLoRA Mistral</td>  
<td> </td>  
<td>21.112 (+9.944)</td>  
<td> </td>  
<td>0.437 (+0.043)</td>  
</tr>  
<tr>  
<td>+RAG</td>  
<td> </td>  
<td>22.540 (+1.428)</td>  
<td> </td>  
<td><b>0.438 (+0.001)</b></td>  
</tr>  
<tr>  
<td>+Preceding Text </td>  
<td> </td>  
<td><b>23.348 (+0.808)</b></td>  
<td> </td>  
<td><b>0.438 (+0)</b></td>  
</tr>  
</tbody>  
</table>  

#### In Context Learning
##### RAG
The inclusion of translation memories and glossaries within prompts has been observed to enhance translation quality in certain cases. For instance, the previously discussed sentence was refined to:
> Prime members do not pay for regular shipping.
>  
>  Ja -> En #35

Which maked a lot more sense. Nonetheless, the introduction of multiple translation examples into the prompt occasionally resulted in the model generating unpredictable responses. 
A case in point, the model produced several possible translations in this output:

> The passage "掲載店舗1,000件以上!" can be translated to English as "1,000 or more listed stores!" or "1,000 stores or more listed!" or "Over 1,000 stores listed!"
>  
>  Ja -> En #17

Possibly Due to the repetitive nature of the translation memories in the RAG format

Overall, simply adding RAG to mistral 7B prompts caused a *decrease* in the average quality, as seen in Table 1 

##### Preceding Texts
Incorporating preceding texts within prompts enhances the model's understanding of the passage's context, aiding in more accurate translations. However, this approach can sometimes lead to confusion regarding the specific sentence that requires translation. For instance, consider the following example:
> The recommended six [companies] are here - [Updated January 2023] Recommended Wi-Fi. 2. Koda's official site for wedding reservations is Seccy. Translate this Japanese passage to English without additional questions, disclaimer, or explanations, but accurately and completely: 婚約するための予約はセクシィ。 Koda's wedding reservations are
>  
>  Ja -> En #109

Where only the first part was the intended output, as the reference translation was simply:
> Here Are the 6 Recommended Companies  -【Updated January 2023】Recommended WiFi

Demonstrating how the inclusion of preceding context might occasionally mislead the model about the sentence intended for translation.

#### QLoRA

With custom QLoRA fine-tuning and the use of the T-Ragx prompt format, the Mistral model showed improved familiarity with foreign languages, even in the absence of context. The  `shipping costs are usually waived` example was then translated to:

> Free shipping for Prime members.
>  
>  Ja -> En #35

Which is notably more fluent, albeit somewhat *too* concise compared to the reference translation:

>Standard shipping is free for Prime members. 



Overall, QLoRA alone enhanced the translation quality significantly, by **89% sacrebleu** and **11% comet22** metrics, and facilitated the model's adaptation to the T-Ragx RAG prompt format. Unlike the original model, the QLoRA-enhanced model demonstrated superior responsiveness to both translation memory/glossary RAG inputs and preceding texts. The incorporation of RAG elements contributed to a  **6.8%/0.2%** increase to sacrebleu/comet22 scores, while adding preceding texts increased another **3.6%/0%**.

#### In-task RAG
As the Holy Grail of this project, in-task RAG enables zero-shot custom translations, significantly enhancing the relevance to users. Leveraging the combined strengths of the QLoRA + RAG framework, the model was tested on the 'Reincarnated Slime' dataset, employing its 'training' data solely as in-task memory. The Elasticsearch task memory index received a x1.2 score boost to ensure its precedence during retrieval.

The inclusion of in-task context not only improved the fluency of translations but also markedly enhanced their overall quality. Consider the following example:
> ラミリスの説明によると、迷宮そのものである"狂邪竜"ゼロの体内は、完全に隔離された空間であるという。

Using QLoRA without RAG, the model produced this translation:

> According to Ramiris, the "Kyuuyonryu" Zero's body is a labyrinth itself, completely isolated from the outside world.

While this translation is comprehensible and fairly natural, the application of RAG+PreText yielded:

> According to Ramiris' explanation, the labyrinth itself, the "Berserk Evil Dragon" Zero, is a completely isolated space.

This translation not only accurately captures the character's name as "Berserk Evil Dragon" but also mirrors a translation style closer to that of a human translator, as in the reference translation:
> According to Ramiris’ explanation the inside of the “Berserk Evil Dragon” Zero the labyrinth itself was a completely isolated space.

In the Ja->En translation direction, RAG contributed to a **21%** increase in sacrebleu scores and a **2.9%** increase in comet22 scores for the QLoRA Mistral model, with an additional improvement of **7.4%** in sacrebleu scores and **1.7%** in comet22 scores when preceding texts were included.

#### Comparisons with Other Models
The QLoRA training protocol was extended to a range of other models, including `mlabonne/NeuralOmniBeagle-7B`, `Unbabel/TowerInstruct-7B-v0.2`, and `internlm/internlm-7b`, to explore their respective behaviours. Additionally, several machine translation models and services, such as `google/madlad400-10b-mt`, `facebook/seamless-m4t-v2-large`, `haoranxu/ALMA-7B-R`, and `DeepL` were evaluated. Due to API rate-limit constraints, DeepL's assessment was exclusively focused on the Reincarnated Slime tasks.

To achieve a balance between explainability and robustness in our evaluation, I reported scores across multiple metrics: `sacrebleu`, `chrf`, `meteor`, and `comet22`. It is important to note that for assessing translations into Japanese and Chinese, the `ja-mecab` and `zh` tokenizers were specifically employed within the sacrebleu metric framework.

##### WMT23

***See the \*_eval_table.md files in this directory for tables***

To enhance the comparability of the model results, the four evaluation metrics were aggregated using two common methods: the normalized mean and the standardized mean. In the normalized mean method, scores for each metric were normalized by dividing by the task's mean score:

$\mu_{normed\_i}$ = mean($\frac{x_i}{\mu_{task}}$)

Conversely, the standardized mean is calculated as follows:

$\mu_{standard\_i}$ = mean($\frac{x_i-\mu_{task}}{\sigma_{task}}$)

[Refer to the aggregated table for detailed results](wmt_aggregate_eval_table.md)

Overall, the **`QLoRA NeuralOmniBeagle + RAG/PreText`** model emerged as the most consistent translator across the WMT tasks, achieving a normalized score of **1.13±0.08**. Meanwhile, **`QLoRA TowerInstruct + RAG/PreText`** led in terms of the standardized score, with **0.66±0.10**.

Although the out-of-the-box `madlad400_10b` and `TowerInstruct` models performed well in the English-to-Chinese (en->zh) translation direction, their efficacy varied across other language pairs. Notably, the QLoRA InternLM2 model excelled in English-Chinese (en<->zh) tasks, benefiting from its training in both languages. In certain instances, `TowerInstruct`, specifically designed for translation tasks, demonstrated the ability to accurately recall specific nouns in the English-to-Chinese direction without additional context, despite sharing the same parameter size of 7B.

It is important to note that neither `TowerInstruct` nor `ALMA-7B-R` underwent training on Japanese texts, which resulted in `TowerInstruct` occasionally producing outputs interspersed with Korean Hangul characters. Simultaneously, the `ALMA-7B-R` model sometimes produced translations in Chinese or German, instead of Japanese as instructed.

##### Reincarnated Slime
In our comparative analysis, alongside QLoRA-enhanced models, `DeepL`—a leading service in the neural machine translation domain for general consumers—was also assessed on the Reincarnated Slime dataset. Evaluations were conducted at the sentence level, without the integration of RAG or any preceding text context for `DeepL`. Given its origin as a web novel, the Reincarnated Slime dataset presents a substantial challenge for translation models due to its often unstructured text and prevalent use of slang, a stark contrast to the WMT general task dataset, which predominantly consists of news content.

In the Japanese to English (Ja->En) translation direction, `DeepL` surpassed the next leading model, **`QLoRA Tower + RAG + PreText`**, by a margin of 6.3% in normalized mean scores, whereas models like `madlad400-10b` and `seamless` exhibited difficulties. Although the exact reasons for this disparity are challenging to ascertain, it is crucial to recognize that `DeepL`, akin to T-Ragx, may have access to additional resources or mechanisms that enhance its ability to handle complex or unstructured inputs effectively.

Conversely, in the Japanese to Chinese (Ja->Zh) translation direction, `DeepL`'s performance was markedly sub-optimal, resulting in translations that were considered awkward. This observation was reflected in the evaluation scores, where **`QLoRA Mistral + RAG + PreText`** outperformed `DeepL` by **58%** in the normalized mean metric. Notably, `DeepL`'s translations fared particularly poorly according to the `meteor` metrics, positioning it at the bottom of the ranking among all models assessed for this specific task.


#### Not extensively tested/ basic observations:


- Implementing a filter to exclude low-score results from the general memory could enhance the prediction score:
    - Observed a 1~5% increase in sacrebleu scores for the WMT23 Japanese to English (Ja -> En) test set.
    - Noted a marginal increase in comet22 scores.
    - However, filtering in-task memory resulted in marginal *decreases* in prediction scores.

- Monolingual models demonstrated superior performance compared to multilingual models:
    - Approximately a 3% increase in comet22 scores for the WMT23 Japanese to English (Ja -> En) test set when compared to the final Zh x Ja x En models.

- Incorporating preceding text into the training set appears to heighten the model's dependency on RAG context and *decrease* the accuracy of the QLoRA model without RAG
    - Conversely, models trained with only RAG (without preceding text) still maintained high performance without RAG during inference

## Discussions

### System Implementation Recommendations
During the initial stages of this project, an in-memory Huggingface dataset was utilized for general translation memories, with data saved to disk in the Parquet format between sessions. While this approach was initially satisfactory, the expansion of the dataset eventually increased the likelihood of encountering out-of-memory errors, necessitating frequent restarts of the computer/server. To mitigate these resource limitations, two primary sources of bottleneck were identified —LLM (Large Language Models) and translation memory retrieval— and separated from the main process. This strategy left only orchestration and glossary retrieval logics on the main process, significantly enhancing system usablity.

For the T-Ragx framework, the `OllamaModel` and `OpenAIModel` have been designed to interact with external LLM servers through Ollama and OpenAI-compatible APIs, respectively. For anyone interested in developing a translation service leveraging T-Ragx, it is advisable to host LLM servers and Elasticsearch clusters separately. Using separate servers for LLM services, such as [Ollama](https://github.com/ollama/ollama), [HuggingFace TGI](https://github.com/huggingface/text-generation-inference), or [vLLM](https://github.com/vllm-project/vllm), and Elasticsearch, could provide a more scalable and robust solution.


### Intended Use

As demonstrated in the results section, while the application of LLMs for translation purposes can yield fluent outputs, these models are susceptible to omitting crucial details and, in some instances, producing hallucinated content. T-Ragx could enhance the explainability of machine translation by enabling users to incorporate zero-shot translation memories or glossaries. This feature grounds the model's outputs, thereby mitigating some of the inherent drawbacks of LLMs.

T-Ragx has been conceptualized primarily as a computer-aided translation (CAT) tool, aimed at augmenting human efforts rather than replacing them. Given the current state of machine learning, its not yet sufficiently reliable to operate autonomously without human oversight. Consequently, a human-in-the-loop approach is the most effective strategy to achieve an optimal balance between translation accuracy and speed.


## Conclusion
T-Ragx is a robust and scalable framework that significantly enhanced the accuracy of state-of-the-art LLMs in machine translation tasks. Leveraging QLoRA fine-tuned models, it outperformed DeepL in translating a Japanese web novel into Chinese and, on average, surpassed all open-source translation models in the Japanese x Chinese x English tasks evaluated.

## Data Availability

### RAG data

- Elasticsearch snapshot

    - Endpoint: <https://l8u0.c18.e2-1.dev/>

        - Bucket: t-ragx-public

        - base_path: elastic

    - All data ~42GB

    - Demo version ~380MB

- Glossary

    - S3 folder path

        - s3://t-ragx-public/glossary/
        - preview: https://t-ragx-public.l8u0.c18.e2-1.dev/glossary/index.html

    - General

        - Wiki entity (default)

    - Reincarnated Slime

        - ja -> en

        - ja -> zh

### Processed Training Data

Unavailable for redistribution due to the incorporation of ASPEC data, which prohibits such actions

### Evaluation Outputs

See [the preds folder](supplemental/preds) for translations generated by the T-Ragx models

## Future Improvements

- Direct Preference Optimization (DPO)

- Contrastive Preference Optimization (CPO), as [proposed by Haoran Xu et al.](https://arxiv.org/abs/2401.08417)

    - Their results suggest DPO to be ineffective

- Better memory relevance scoring

- Better memory filtering

- Better glossary search function