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


@Model.register('model_seq2vec_classifier')
class Seq2VecClassifier(Model):
    """
    This ``SequenceClassifier`` simply encodes a sequence of text with a ``Seq2VecEncoder``, then
    predicts a label for the sequence.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2VecEncoder``
        The encoder that we will use in between embedding tokens
        and predicting output tags.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 s_encoder: Seq2VecEncoder,
                 feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Seq2VecClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size('labels')
        self.s_encoder = s_encoder
        self.feedforward = feedforward
        self.metrics = {'accuracy': CategoricalAccuracy(), 'f1_favor': F1Measure(positive_label=0)}
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,
                s: Dict[str, torch.LongTensor],
                target: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_s = self.text_field_embedder(s)
        mask_s = get_text_field_mask(s)
        encoded_s = self.s_encoder(embedded_s, mask_s)

        logits = self.feedforward(encoded_s)

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
                    params: Params) -> 'Seq2VecClassifier':

        text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop('text_field_embedder'))
        s_encoder = Seq2VecEncoder.from_params(params.pop('s_encoder'))
        feedforward = FeedForward.from_params(params.pop('feedforward'))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   s_encoder=s_encoder,
                   feedforward=feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
