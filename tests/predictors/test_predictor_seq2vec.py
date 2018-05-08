from unittest import TestCase

from pytest import approx

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from sklearn.metrics import f1_score

import stance_detection


class TestSeq2VecPredictor(TestCase):
    def test_uses_named_inputs(self):

        target = 'a'
        TEST_FILE = 'C:/Users/nuaax/Dropbox/data61/project/stance_classification/dataset/semeval/' \
                    'SemEval2016-Task6-subtaskA-test-%s.txt' % target
        archive = load_archive('C:/Users/nuaax/Dropbox/data61/project/stance_classification/dataset/semeval/models'
                               '/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'predictor_seq2vec_classifier')
        goldens = []
        predictions = []
        with open(TEST_FILE) as test_file:
            for line in test_file:
                if line.startswith('ID\tTarget\tTweet\tStance'):
                    continue
                _, t, s, label = line.strip().split('\t')
                inputs = {
                    's': s,
                    'target': t
                }

                result = predictor.predict_json(inputs)
                pred_label = result.get("label")
                # class_prob = result.get("class_prob")
                # assert class_prob is not None
                # assert all(cp > 0 for cp in class_prob)
                # assert sum(class_prob) == approx(1.0)
                assert pred_label in ['FAVOR', 'AGAINST', 'NONE']

                predictions.append(pred_label)
                goldens.append(label)

        macro = f1_score(goldens, predictions, average='macro')
        micro = f1_score(goldens, predictions, average='micro')
        mean = (macro + micro) / 2.
        print('macro:', macro)
        print('micro:', micro)
        print('mean:', mean)
