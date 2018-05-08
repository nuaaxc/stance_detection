from allennlp.common.testing import ModelTestCase
import stance_detection


class Seq2VecClassifierTest(ModelTestCase):
    def setUp(self):
        super(Seq2VecClassifierTest, self).setUp()
        self.set_up_model(
            'C:/Users/nuaax/PycharmProjects/consensus_prediction_allan/experiments/config_BiLSTM.json',
            'C:/Users/nuaax/PycharmProjects/consensus_prediction_allan/tests/fixtures/test_crossnet_reader.txt'
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
