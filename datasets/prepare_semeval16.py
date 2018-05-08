from sklearn.model_selection import train_test_split


DATA_DIR = 'C:/Users/nuaax/Dropbox/data61/project/stance_classification/dataset/semeval/'
TRAIN_DEV = DATA_DIR + 'semeval2016-task6-subtaskA-train-dev-%s.txt'
TRAIN = DATA_DIR + 'semeval2016-task6-subtaskA-train-%s.txt'
DEV = DATA_DIR + 'semeval2016-task6-subtaskA-dev-%s.txt'


data_size_tr = {
    'a': 513,
    'cc': 395,
    'fm': 664,
    'hc': 689,
    'la': 653,
}

target_name = {
    'a': 'Atheism',
    'cc': 'Climate Change is a Real Concern',
    'fm': 'Feminist Movement',
    'hc': 'Hillary Clinton',
    'la': 'Legalization of Abortion',
}


def create_dev_set(t, ignore_header=True):
    """
    Create validation set from training set based on stratified split
    """
    with open(TRAIN_DEV % t, encoding='windows-1252') as train_dev_file:
        X = []
        y = []
        if ignore_header:
            next(train_dev_file)
        for line in train_dev_file:
            _id, target, text, label = line.strip().split('\t')
            assert target == target_name[t]
            X.append((_id, target, text))
            y.append(label)
        assert len(X) == len(y) == data_size_tr[t]
        X_train, X_dev, y_train, y_dev = train_test_split(X, y,
                                                          test_size=0.1,
                                                          random_state=42,
                                                          stratify=y)
        print('X_train:', len(X_train))
        print('y_train:', len(y_train))
        print('X_dev:', len(X_dev))
        print('y_dev:', len(y_dev))

        assert len(X_train) + len(X_dev) == data_size_tr[t]
        assert len(y_train) + len(y_dev) == data_size_tr[t]

        n_f_train = len([label for label in y_train if label == 'FAVOR'])
        n_a_train = len([label for label in y_train if label == 'AGAINST'])
        n_n_train = len([label for label in y_train if label == 'NONE'])

        n_f_dev = len([label for label in y_dev if label == 'FAVOR'])
        n_a_dev = len([label for label in y_dev if label == 'AGAINST'])
        n_n_dev = len([label for label in y_dev if label == 'NONE'])

        print('training set\n\t#f: %d, #a: %d, #n: %d, f/a: %f, a/n: %f'
              % (n_f_train, n_a_train, n_n_train, n_f_train/n_a_train, n_a_train/n_n_train))
        print('development set\n\t#f: %d, #a: %d, #n: %d, f/a: %f, a/n: %f'
              % (n_f_dev, n_a_dev, n_n_dev, n_f_dev/n_a_dev, n_a_dev/n_n_dev))

        # write to train
        print('saving to %s ...' % (TRAIN % t))
        with open(TRAIN % t, 'w') as train_file:
            if ignore_header:
                train_file.write('ID\tTarget\tTweet\tStance\n')
            for i in range(len(X_train)):
                train_file.write('\t'.join(X_train[i]) + '\t' + y_train[i] + '\n')
        print('saved.')

        # write to dev
        print('saving to %s ...' % (DEV % t))
        with open(DEV % t, 'w') as dev_file:
            if ignore_header:
                dev_file.write('ID\tTarget\tTweet\tStance\n')
            for i in range(len(X_dev)):
                dev_file.write('\t'.join(X_dev[i]) + '\t' + y_dev[i] + '\n')
        print('saved.')


if __name__ == '__main__':
    # create_dev_set('a')
    # create_dev_set('cc')
    # create_dev_set('fm')
    # create_dev_set('hc')
    # create_dev_set('la')
    pass

