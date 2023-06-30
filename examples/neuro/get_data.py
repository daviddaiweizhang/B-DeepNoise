import pandas as pd
import numpy as np


def get_data(dataset, split):
    data_file = f'data/neuroimaging/{dataset}/features.tsv'
    indexes_file = f'data/neuroimaging/{dataset}/splits.tsv'
    data = pd.read_csv(data_file, sep='\t', index_col=0)
    indexes = pd.read_csv(indexes_file, sep='\t')
    is_feature = data.columns.str.startswith('feature_')
    idx_train = indexes[f'split_{split}']
    n_observations = data.shape[0]
    idx_test = set(range(n_observations)).difference(idx_train)
    idx_test = list(idx_test)
    data_train = data.iloc[idx_train]
    data_test = data.iloc[idx_test]
    targets_train = data_train['target'].to_numpy()[:, np.newaxis]
    features_train = data_train.loc[:, is_feature].to_numpy()
    targets_test = data_test['target'].to_numpy()[:, np.newaxis]
    features_test = data_test.loc[:, is_feature].to_numpy()
    return (features_train, targets_train), (features_test, targets_test)
