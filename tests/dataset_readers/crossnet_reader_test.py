from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from stance_detection.readers import CrossNetTSVReader


class TestConsNetDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = CrossNetTSVReader(pos_s=2, pos_target=1, pos_label=3)
        instances = ensure_list(reader.read(
            'C:/Users/nuaax/PycharmProjects/stance_detection_allan/tests/fixtures/test_crossnet_reader.txt'))
        print('# instances:', len(instances))
        for instance in instances:
            print(instance)



