from typing import Dict, Optional
import logging

import numpy as np
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary, Instance
from allennlp.modules import Seq2VecEncoder, FeedForward, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


logger = logging.getLogger(__name__)


@Model.register('model_mitre_classifier')
class MITREClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 s_encoder: Seq2VecEncoder,
                 target_encoder: Seq2VecEncoder,
                 feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(MITREClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size('labels')
        self.s_encoder = s_encoder
        self.target_encoder = target_encoder
        self.feedforward = feedforward
        self.metrics = {'accuracy': CategoricalAccuracy()}
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,
                s: Dict[str, torch.LongTensor],
                target: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        embedded_target = self.text_field_embedder(target)
        mask_target = get_text_field_mask(target)
        encoded_target = self.target_encoder(embedded_target, mask_target)

        embedded_s = self.text_field_embedder(s)
        mask_s = get_text_field_mask(s)
        encoded_s = self.s_encoder(embedded_s, mask_s)

        logits = self.feedforward(torch.cat([encoded_s, encoded_target], dim=-1))

        output_dict = {'logits': logits}

        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict['loss'] = loss

        return output_dict

    @overrides
    def decode(self,
               output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_prob = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_prob'] = class_prob

        predictions = class_prob.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace='labels')
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @classmethod
    def from_params(cls,
                    vocab: Vocabulary,
                    params: Params) -> 'MITREClassifier':

        text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop('text_field_embedder'))
        s_encoder = Seq2VecEncoder.from_params(params.pop('s_encoder'))
        target_encoder = Seq2VecEncoder.from_params(params.pop('target_encoder'))
        feedforward = FeedForward.from_params(params.pop('feedforward'))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   s_encoder=s_encoder,
                   target_encoder=target_encoder,
                   feedforward=feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
