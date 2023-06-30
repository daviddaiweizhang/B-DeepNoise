from os import makedirs
from os.path import dirname
import numpy as np
from utils.utils import gen_splits, split_train_test
import sys


def cond_dist_sampler(x, y_median, y_deviation, n_samples):

    assert np.ndim(x) == 1
    n_observations = x.size

    if y_median == 'constant':
        y = x * 0
    if y_median == 'identity':
        y = x
    elif y_median == 'spline':
        y = (
                (x <= -0.5) * (x + 1)
                + (x > -0.5) * (x <= 0) * (-x)
                + (x > 0) * x)

    unif_quantiles = np.linspace(0, 1, n_samples+2)[1:-1]
    unif_quantiles = np.tile(
            unif_quantiles.reshape(-1, 1),
            [1, n_observations])
    if y_deviation == 'zero':
        noise = 0
        noise_scale = 0
        noise_sign = 0
    elif y_deviation == 'gaussian':
        noise = np.random.randn(n_observations)
        noise_scale = 1.0 / 3
        noise_sign = 1
    elif y_deviation == 'uniform':
        noise = unif_quantiles * 2 - 1
        noise_scale = 0.1
        noise_sign = 1
    elif y_deviation == 'heteroscedastic':
        noise = unif_quantiles * 2 - 1
        noise_scale = (
                (x > -np.inf) * (x < -0.85) * 0.1
                + (x > -0.85) * (x < -0.65) * 0.5
                + (x > -0.65) * (x < -0.35) * 0.1
                + (x > -0.35) * (x < -0.15) * 0.5
                + (x > -0.15) * (x < 0.30) * 0.1
                + (x > 0.30) * (x < 0.7) * 0.5
                + (x > 0.7) * (x < np.inf) * 0.1
                )
        noise_sign = 1
    elif y_deviation == 'skewed':
        noise = unif_quantiles * 2 - 1
        noise[noise > 0] = noise[noise > 0] * 8
        noise_scale = 0.1
        noise_sign = ((x > -0.5) * (x < 0)) * 2 - 1
    elif y_deviation == 'multimodal':
        noise = unif_quantiles - 0.5
        noise[(np.abs(x) < 0.5) * (noise >= 0.2)] += 1.7
        noise[(np.abs(x) < 0.5) * (noise <= -0.2)] -= 1.7
        noise[(np.abs(x) < 0.5) * (np.abs(noise) < 0.2)] *= 8.5
        noise_scale = 0.25
        noise_sign = 1
    elif y_deviation == 'mixed':
        noise = unif_quantiles - 0.5
        noise[(np.abs(x) < 0.5) * (noise >= 0.2)] += 1.7
        noise[(np.abs(x) < 0.5) * (noise <= -0.2)] -= 1.7
        noise[(np.abs(x) < 0.5) * (np.abs(noise) < 0.2)] *= 8.5
        noise[(np.abs(x) > 0.75) * (noise > 0)] *= 4
        noise_sign = -np.sign(x)
        noise_scale = 0.25

    y = y + noise * noise_scale * noise_sign
    return y


def get_data_1d(
        x_distribution, y_median, y_deviation,
        n_inputs, n_outputs_per_input):

    if x_distribution == 'uniform':
        x = np.linspace(-1, 1, n_inputs)
    if x_distribution == 'beta':
        x = np.random.beta(5, 5, n_inputs) * 2 - 1
    elif x_distribution == 'interval':
        n_pos = n_inputs // 10
        n_neg = n_inputs - n_pos
        x = np.concatenate([
            np.linspace(-1, 0, n_neg, endpoint=False),
            np.linspace(0, 1, n_pos, endpoint=True)])
    x.sort()

    y = cond_dist_sampler(x, y_median, y_deviation, n_outputs_per_input)

    return x, y


def get_data_3d(
        y_median, y_deviation,
        n_inputs, n_outputs_per_input):
    start = -1
    stop = 1
    n = np.ceil(n_inputs**(1/3)).astype(int)
    step = (stop - start) / (n - 1)
    stop += step / 2
    x = np.mgrid[start:stop:step, start:stop:step, start:stop:step]
    x = x.T
    x = x.reshape(np.prod(x.shape[:-1]), x.shape[-1])
    idx = np.random.choice(x.shape[0], n_inputs, replace=False)
    x = x[idx]
    u = 0.5 * x[:, 0] * x[:, 1] + 0.5 * x[:, 2]**3
    # u = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * x[:, 2]
    y = cond_dist_sampler(u, y_median, y_deviation, n_outputs_per_input)
    return x, y


def get_data_nd(
        x_distribution, y_median, y_deviation,
        n_features, n_observations):

    assert n_features >= 4

    if x_distribution == 'uniform':
        x = np.random.rand(n_observations, n_features) * 2 - 1
    elif x_distribution == 'gaussian':
        x = np.random.randn(n_observations, n_features) * 0.5

    u = (
            np.sin((x[:, 0] + x[:, 1]) * 2*np.pi)
            + np.sqrt(x[:, 2]**2 + x[:, 3]**2) * np.sqrt(2) - 1)
    y = cond_dist_sampler(u, y_median, y_deviation)
    y = y.reshape(-1, 1)
    return x, y


def get_data_synthetic(
        x_distribution, y_median, y_deviation, n_features,
        n_inputs, n_outputs_per_input, n_observed):
    if n_features == 1:
        x_quantiles, y_quantiles = get_data_1d(
                x_distribution=x_distribution,
                y_median=y_median,
                y_deviation=y_deviation,
                n_inputs=n_inputs,
                n_outputs_per_input=n_outputs_per_input)
    else:
        x_quantiles, y_quantiles = get_data_nd(
                x_distribution=x_distribution,
                y_median=y_median,
                y_deviation=y_deviation,
                n_features=n_features,
                n_inputs=n_inputs,
                n_outputs_per_input=n_outputs_per_input)
    x_tiled = np.tile(
            x_quantiles[np.newaxis, ...],
            [n_outputs_per_input, 1])
    (x_observed, y_observed), (__, __) = split_train_test(
            x=x_tiled.reshape(-1, x_tiled.shape[-1]),
            y=y_quantiles.reshape(-1, y_quantiles.shape[-1]),
            n_train=n_observed)
    return (x_observed, y_observed), (x_quantiles, y_quantiles)


def gen_data_synthetic(
        x_distribution, y_median, y_deviation, n_features,
        n_inputs, n_outputs_per_input, prefix):
    if n_features == 1:
        x, y = get_data_1d(
                x_distribution=x_distribution,
                y_median=y_median,
                y_deviation=y_deviation,
                n_inputs=n_inputs,
                n_outputs_per_input=n_outputs_per_input)
        x = x[..., np.newaxis]
    elif n_features == 3:
        x, y = get_data_3d(
                y_median=y_median,
                y_deviation=y_deviation,
                n_inputs=n_inputs,
                n_outputs_per_input=n_outputs_per_input)
    else:
        x, y = get_data_nd(
                x_distribution=x_distribution,
                y_median=y_median,
                y_deviation=y_deviation,
                n_features=n_features,
                n_inputs=n_inputs,
                n_outputs_per_input=n_outputs_per_input)
    makedirs(dirname(prefix), exist_ok=True)
    outfile = f'{prefix}features.tsv'
    np.savetxt(outfile, x, fmt='%.3f', delimiter='\t')
    print(outfile)
    outfile = f'{prefix}targets.tsv'
    np.savetxt(outfile, y.T, fmt='%.3f', delimiter='\t')
    print(outfile)


def synthesize_datasets(
        n_features, x_distribution, y_median, y_deviation, n_inputs,
        n_outputs_per_input, n_splits, n_train, prefix):

    gen_data_synthetic(
        x_distribution=x_distribution,
        y_median=y_median,
        y_deviation=y_deviation,
        n_features=n_features,
        n_inputs=n_inputs,
        n_outputs_per_input=n_outputs_per_input,
        prefix=prefix)

    gen_splits(
            n_observations=n_inputs * n_outputs_per_input,
            n_train=n_train,
            n_splits=n_splits,
            outfile=f'{prefix}splits.tsv')


def get_data(dataset, split):
    prefix = f'data/synthetic/{dataset}/'
    x = np.loadtxt(f'{prefix}features.tsv', ndmin=2, dtype=np.float32)
    y = np.loadtxt(f'{prefix}targets.tsv', ndmin=2, dtype=np.float32)
    indices = np.loadtxt(f'{prefix}splits.tsv', dtype=int)
    assert indices.ndim == 2
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[0] == y.shape[0]
    y = y.T[..., np.newaxis]
    x_tiled = np.tile(x[np.newaxis, ...], [y.shape[0], 1, 1])
    split_train = split
    n_splits = indices.shape[1]
    split_test = (split + 1) % n_splits
    indices_train = indices[:, split_train]
    indices_test = indices[:, split_test]
    x_train = x_tiled.reshape(-1, x_tiled.shape[-1])[indices_train]
    y_train = y.reshape(-1, y.shape[-1])[indices_train]
    x_test = x_tiled.reshape(-1, x_tiled.shape[-1])[indices_test]
    y_test = y.reshape(-1, y.shape[-1])[indices_test]
    return (x, y), (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':

    x_distribution = 'uniform'
    n_train = 2000
    n_features = 1
    for y_deviation in ['heteroscedastic', 'skewed', 'multimodal']:
        dataset = (
                f'features{n_features}-'
                'identity-'
                f'{y_deviation}-'
                f'ntrain{n_train:04d}')
        synthesize_datasets(
            n_features=n_features,
            x_distribution=x_distribution,
            y_median='identity',
            y_deviation=y_deviation,
            n_inputs=1000,
            n_outputs_per_input=500,
            n_splits=20,
            n_train=n_train,
            prefix=f'data/synthetic/{dataset}/')

    n_train = int(sys.argv[1])
    for y_deviation in ['heteroscedastic', 'skewed', 'multimodal', 'mixed']:
        for n_features in [1]:
            for x_distribution in ['uniform', 'interval']:
                dataset = (
                        f'features{n_features}-'
                        f'{x_distribution}-'
                        f'{y_deviation}-'
                        f'ntrain{n_train:07d}')
                synthesize_datasets(
                    n_features=n_features,
                    x_distribution=x_distribution,
                    y_median='spline',
                    y_deviation=y_deviation,
                    n_inputs=1000,
                    n_outputs_per_input=500,
                    n_splits=20,
                    n_train=n_train,
                    prefix=f'data/synthetic/{dataset}/')
                data = get_data(dataset, split=0)
                print('x quantiles:', data[0][0].shape)
                print('y quantiles:', data[0][1].shape)
                print('x train:', data[1][0].shape)
                print('y train:', data[1][1].shape)
                print('x test:', data[2][0].shape)
                print('y test:', data[2][1].shape)
