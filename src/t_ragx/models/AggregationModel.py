import pandas as pd
from comet import download_model, load_from_checkpoint

from .LangDetectModel import FastTextLangDetectModel


class CometAggregationModel:
    model = None

    def __init__(
            self,
            model_id="Unbabel/wmt22-cometkiwi-da",
            get_lang_func=None,

    ):
        self.get_lang = get_lang_func
        if get_lang_func is None:
            fast_text_lang_detect_model = FastTextLangDetectModel()
            self.get_lang = fast_text_lang_detect_model.get_lang

        model_path = download_model(model_id)
        self.model = load_from_checkpoint(model_path)

    def get_blind_score(self, out_text_list, source_text, target_lang_code='en'):
        comet_data = [{
            "src": source_text[0],
            "mt": out_text_list[i],
            "ref": ""
        } for i in range(len(out_text_list))
        ]
        scores = self.model.predict(comet_data, batch_size=8, gpus=1)
        out_score = scores.scores
        for i in range(len(out_score)):
            if self.get_lang(out_text_list[i]) != target_lang_code:
                out_score[i] = 0
        return out_score

    def combine_preds(self, pred_dict, source_text, target_lang_code='en'):
        blind_results = {k: self.get_blind_score(pred_dict[k], source_text, target_lang_code=target_lang_code) for k in
                         pred_dict}
        score_df = pd.DataFrame.from_dict({k: blind_results[k] for k in blind_results}, orient="columns")
        best_pred_key = score_df.apply(lambda row: row.index[row.argmax()], axis=1).to_list()

        combined_pred = []
        for i in range(len(best_pred_key)):
            key = best_pred_key[i]
            combined_pred.append(pred_dict[key][i])

        return combined_pred
