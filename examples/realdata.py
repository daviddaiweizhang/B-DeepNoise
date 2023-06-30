import numpy as np


def get_data(dataset, split):
    if dataset in ['abide', 'abcd', 'abcd-noimg']:
        _DATA_DIRECTORY_PATH = "./data/neuroimaging/" + dataset + "/"
    else:
        _DATA_DIRECTORY_PATH = "./data/uci/" + dataset + "/data/"
    # Adopted from github.com/yaringal/DropoutUncertaintyExps
    _DATA_FILE = _DATA_DIRECTORY_PATH + "data.txt"
    _INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "index_features.txt"
    _INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "index_target.txt"
    _INDEX_TRAIN_FILE = (
            _DATA_DIRECTORY_PATH
            + 'index_train_' + str(split) + '.txt')
    _INDEX_TEST_FILE = (
            _DATA_DIRECTORY_PATH
            + 'index_test_' + str(split) + '.txt')

    data = np.loadtxt(_DATA_FILE)
    index_features = np.loadtxt(_INDEX_FEATURES_FILE)
    index_target = np.loadtxt(_INDEX_TARGET_FILE)

    features = data[:, [int(i) for i in index_features.tolist()]]
    targets = data[:, [index_target.astype(int)]]

    index_train = np.loadtxt(_INDEX_TRAIN_FILE, dtype=int)
    index_test = np.loadtxt(_INDEX_TEST_FILE, dtype=int)

    features_train = features[index_train]
    targets_train = targets[index_train]
    features_test = features[index_test]
    targets_test = targets[index_test]

    return (features_train, targets_train), (features_test, targets_test)
