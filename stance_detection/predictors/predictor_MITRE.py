from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('predictor_MITRE_classifier')
class MITREPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        s = json_dict['s']
        target = json_dict['target']
        instance = self._dataset_reader.text_to_instance(s=s, target=target)

        # label_dict will be like {0: "Favor", 1: "Against", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')

        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance, {"all_labels": all_labels}
