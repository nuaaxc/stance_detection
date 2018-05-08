from typing import Dict, List
import logging

from overrides import overrides

import csv

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_filter import WordFilter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token

logger = logging.getLogger(__name__)


@WordFilter.register('stance_stopwords')
class StanceStopwordFilter(WordFilter):
    """
    Uses a list of stopwords to filter.
    """

    def __init__(self):
        self.stopwords = {"\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                          ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!", "?",
                          "rt", "semst", "...", "thats", "im", "'s", "via",
                          'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                          "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                          'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                          'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                          'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                          'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                          'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                          'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                          'up', 'down', 'in', 'on', 'under', 'then', 'once',
                          'here', 'there', 'when', 'where', 'why', 'how', 'both', 'each', 'other', 'some', 'own',
                          'same', 'than', 's', 't', 'will', 'just', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ma'}
        # self.stopwords = {"\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
        #                   ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!", "?",
        #                   "rt", "semst", "...", "thats", "im", "'s", "via"}

    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        return [word for word in words if word.text.lower() not in self.stopwords]


@DatasetReader.register('stance_reader')
class CrossNetTSVReader(DatasetReader):
    """
    Reads a , and creates a dataset suitable for

    Input: Expected format for each input line:

    Output: The output of ``_read`` is a list of ``Instance`` s with the fields:

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstract into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 pos_s: int,
                 pos_target: int,
                 pos_label: int,
                 lazy: bool = False,
                 skip_header: bool = True,
                 delimiter: str = '\t',
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._pos_s = pos_s
        self._pos_target = pos_target
        self._pos_label = pos_label
        self._skip_header = skip_header
        self._delimiter = delimiter
        self._tokenizer = tokenizer or WordTokenizer(word_filter=StanceStopwordFilter())
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer}

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'r') as data_file:
            logger.info('Reading instances from lines in file at: %s', file_path)
            reader = csv.reader(data_file, delimiter=self._delimiter)
            if self._skip_header:
                next(data_file)
            for example in reader:
                s = example[self._pos_s]
                target = example[self._pos_target]
                label = example[self._pos_label]
                yield self.text_to_instance(s, target, label)

    @overrides
    def text_to_instance(self,
                         s: str,
                         target: str,
                         label: str = None) -> Instance:
        tokenized_s = self._tokenizer.tokenize(s)
        tokenized_target = self._tokenizer.tokenize(target)

        s_field = TextField(tokenized_s, self._token_indexers)
        target_field = TextField(tokenized_target, self._token_indexers)

        fields = {'s': s_field, 'target': target_field}

        if label:
            fields['label'] = LabelField(label)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'CrossNetTSVReader':
        pos_s = params.pop('pos_s', None)
        pos_target = params.pop('pos_target', None)
        pos_label = params.pop('pos_label', None)
        lazy = params.pop('lazy', False)
        skip_header = params.pop('skip_header', None)
        delimiter = params.pop('delimiter', None)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(pos_s=pos_s,
                   pos_target=pos_target,
                   pos_label=pos_label,
                   skip_header=skip_header,
                   delimiter=delimiter,
                   lazy=lazy,
                   tokenizer=tokenizer,
                   token_indexers=token_indexers)
