from time import time
import numpy as np
from dalea.samplers_model import (
        sample_kernel_conjugate,
        sample_kernel_heteroscedastic)
from utils import fit_slope


def get_samples(func, n_samples, **kwargs):
    sample_list = []
    for _ in range(n_samples):
        sample = func(**kwargs)
        sample_list.append(sample)
    return np.concatenate(sample_list)


def get_data(
        n_features, n_targets, n_observations, n_replicates, noise_scale):
    kernel = np.random.randn(n_replicates, n_features, n_targets)*2 + 3
    features = np.random.randn(n_replicates, n_observations, n_features)
    noise = np.random.randn(n_replicates, n_observations, n_targets)
    targets = features @ kernel + noise * noise_scale
    return features, targets


def main():
    n_samples = 100
    n_observations = 200
    n_features = 64
    n_targets = 64
    n_replicates = 5
    noise_scale = 0.5
    kernel_mean = 0.0
    kernel_scale = 10

    noise_scale = noise_scale * np.ones((1, n_targets))
    kernel_mean = kernel_mean + np.zeros((n_features, n_targets))
    kernel_scale = kernel_scale * np.ones((n_features, n_targets))
    features, targets = get_data(
            n_features, n_targets, n_observations,
            n_replicates, noise_scale)

    kwargs = dict(
            n_samples=n_samples,
            features=features,
            targets=targets,
            noise_scale=noise_scale,
            kernel_mean=kernel_mean,
            kernel_scale=kernel_scale)

    t0 = time()
    beta_states_conjugate = get_samples(
            func=sample_kernel_conjugate, **kwargs)
    print((time() - t0) / n_samples * 1000)

    t0 = time()
    beta_states_heteroscedastic = get_samples(
            func=sample_kernel_heteroscedastic, **kwargs)
    print((time() - t0) / n_samples * 1000)

    slope_mean = fit_slope(
            beta_states_conjugate.mean(0).flatten(),
            beta_states_heteroscedastic.mean(0).flatten())
    assert 0.95 < slope_mean < 1.05

    slope_scale = fit_slope(
            np.log(beta_states_conjugate.std(0)).flatten(),
            np.log(beta_states_heteroscedastic.std(0)).flatten())
    assert 0.95 < slope_scale < 1.05


if __name__ == '__main__':
    main()
