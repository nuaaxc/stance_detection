
from typing import Dict
import os

DATA_DIR_PATH = 'C:/Users/nuaax/Dropbox/data61/project/consensus_detection/dataset/createdebate_saidul/'


def load(target_name: str = None):
    text_dict: Dict = {}
    stance_dict: Dict = {}
    for filename in os.listdir(DATA_DIR_PATH + target_name):
        print(filename)
        doc_id, _type = filename.split('.')
        if _type == 'data':
            text = open(os.path.join(DATA_DIR_PATH + target_name, filename), encoding='utf-8').read().strip()
            text_dict[doc_id] = text
        elif _type == 'meta':
            stance = open(os.path.join(DATA_DIR_PATH + target_name, filename), encoding='utf-8').read().strip()
            stance = stance.split('\n')[2].split('=')[1]
            if stance == '-1':
                stance = 'AGAINST'
            elif stance == '+1':
                stance = 'FAVOR'
            else:
                raise ValueError('Invalid stance type:', stance, 'in', filename)
            stance_dict[doc_id] = stance
        else:
            raise ValueError('Invalid file type:', _type)

    stances = stance_dict.values()
    print('# posts:', len(stances))
    print('# favor:', len([stance for stance in stances if stance == 'FAVOR']))
    print('# against:', len([stance for stance in stances if stance == 'AGAINST']))

    return text_dict, stance_dict


def create_stance_corpus(target_name: str = None):
    """
    abortion (+1066, -849)
    gayRights (+877, -499)
    marijuana (+444, -182)
    obama (+526, -459)
    """
    text_dict, stance_dict = load(target_name)
    output_file = os.path.join(DATA_DIR_PATH, target_name + '.txt')
    print('saving stance corpus to %s ...' % output_file)
    with open(output_file, mode='w', encoding='utf-8') as corpus:
        corpus.write('ID\tTarget\tTweet\tStance\n')
        for _id in text_dict.keys():
            text = text_dict[_id]
            stance = stance_dict[_id]
            corpus.write('\t'.join([_id, target_name, text, stance]) + '\n')
    print('saved.')


def create_consensus_corpus(target_name: str = None,
                            seed: int = 2018):
    text_dict, stance_dict = load(target_name)
    output_file = os.path.join(DATA_DIR_PATH, target_name + '_consensus_seed_%d.txt' % seed)
    favor_ids = [_id for _id, stance in stance_dict.items() if stance == 'FAVOR']
    against_ids = [_id for _id, stance in stance_dict.items() if stance == 'AGAINST']

    assert len(favor_ids) + len(against_ids) == len(stance_dict)

    agree =


if __name__ == '__main__':
    # create_stance_corpus('abortion')
    # create_stance_corpus('gayRights')
    # create_stance_corpus('marijuana')
    # create_stance_corpus('obama')

    create_consensus_corpus('abortion')
