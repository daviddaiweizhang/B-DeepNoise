import numpy as np


def save_data(x, y, prefix):
    assert x.ndim == 2
    assert y.ndim == 2
    n_observations = x.shape[0]
    assert y.shape[0] == n_observations
    xy = np.concatenate([x, y], axis=1)
    idx_x = np.arange(x.shape[1])
    idx_y = np.arange(y.shape[1]) + x.shape[1]
    outfile = f'{prefix}data.txt'
    np.savetxt(
            outfile, xy,
            fmt='%.3f', delimiter='\t')
    print(outfile)
    outfile = f'{prefix}index_features.txt'
    np.savetxt(
            outfile, idx_x.reshape(-1, 1),
            fmt='%d', delimiter='\t')
    print(outfile)
    outfile = f'{prefix}index_target.txt'
    np.savetxt(
            outfile, idx_y.reshape(-1, 1),
            fmt='%d', delimiter='\t')
    print(outfile)


def save_split_index(n_observations, prop_test, n_splits, prefix):
    n_test = int(n_observations * prop_test)
    for i in range(n_splits):
        idx_test = np.random.choice(
                n_observations, n_test, replace=False)
        idx_test.sort()
        idx_train = np.array([
            i for i in range(n_observations)
            if i not in idx_test])
        outfile = f'{prefix}index_train_{i}.txt'
        np.savetxt(
                outfile,
                idx_train.reshape(-1, 1),
                fmt='%d', delimiter='\t')
        print(outfile)
        outfile = f'{prefix}index_test_{i}.txt'
        np.savetxt(
                outfile,
                idx_test.reshape(-1, 1),
                fmt='%d', delimiter='\t')
        print(outfile)
