import pytest
import numpy as np
from scipy import stats

from dalea.samplers_model import (
        param_mean_known_scale,
        sample_kernel,
        sample_pre_activation,
        sample_pre_categorization,
        hard_sigmoid_slope)
from dalea.model import hardmax


def test_param_mean_known_scale_shape():
    shape = (2, 3)
    prior_mean = np.zeros(shape)
    prior_scale = np.ones(shape)
    observation = np.random.randn(*shape)
    observation_scale = np.ones(shape)
    posterior_mean, posterior_scale = param_mean_known_scale(
            prior_mean, prior_scale,
            observation, observation_scale)
    assert posterior_mean.shape == shape
    assert posterior_scale.shape == shape


def test_param_mean_known_scale_value_strong_prior():
    prior_mean = 0
    large = 1e3
    prior_scale = 1.0 / large
    observation = 100
    observation_scale = 1.0 * large
    posterior_mean, posterior_scale = param_mean_known_scale(
            prior_mean, prior_scale,
            observation, observation_scale)
    assert np.isclose(posterior_scale, prior_scale)
    assert np.isclose(posterior_mean, prior_mean)


def test_param_mean_known_scale_value_weak_prior():
    prior_mean = 0
    large = 1e3
    prior_scale = 1.0 * large
    observation = 100
    observation_scale = 1.0 / large
    posterior_mean, posterior_scale = param_mean_known_scale(
            prior_mean, prior_scale,
            observation, observation_scale)
    assert np.isclose(posterior_scale, observation_scale)
    assert np.isclose(posterior_mean, observation)


def test_sample_kernel_shape():
    n_targets = 4
    n_features = 5
    n_observations = 6
    batch_shape = (2, 3)

    features = np.random.randn(*batch_shape, n_observations, n_features)
    targets = np.random.randn(*batch_shape, n_observations, n_targets)
    noise_scale = np.ones(batch_shape + (1, n_targets))
    kernel_scale = np.ones(batch_shape + (n_features, n_targets))
    kernel_mean = np.zeros(batch_shape + (n_features, n_targets))

    sample = sample_kernel(
            features=features,
            targets=targets,
            noise_scale=noise_scale,
            kernel_mean=kernel_mean,
            kernel_scale=kernel_scale)
    assert sample.shape == batch_shape + (n_features, n_targets)


def test_sample_kernel_distribution():
    n = 2
    x = 2
    y = 11
    sigma = 1 / 3
    tau = 1 / 5
    n_samples = 1000
    alpha = 0.01

    n_features = n
    n_targets = 1
    features = np.eye(n_features) * x
    targets = np.ones([n, n_targets]) * y
    noise_scale = np.ones((1, n_targets)) * sigma
    kernel_scale = np.ones((n_features, n_targets)) * tau
    kernel_mean = np.zeros((n_features, n_targets))

    # true posterior parameters
    lambda_0 = (tau / sigma)**(-2)
    lambda_n = lambda_0 + x**2
    kappa_n = 1 / lambda_n
    beta_hat = y / x
    mu_n = x**2 / lambda_n * beta_hat

    # true posterior confidence intervals
    beta_mean_conf_int = (
            mu_n
            + kappa_n**(0.5)
            / n_samples**0.5
            * stats.norm.ppf([alpha/2, 1 - alpha/2]))
    beta_var_conf_int = (
            stats.chi2.ppf([alpha/2, 1 - alpha/2], df=n_samples)
            / n_samples
            * kappa_n * sigma**2)

    # draw samples of beta
    sample_list = []
    for _ in range(n_samples):
        sample = sample_kernel(
                features=features,
                targets=targets,
                noise_scale=noise_scale,
                kernel_mean=kernel_mean,
                kernel_scale=kernel_scale)
        sample_list.append(sample)
    beta_states = np.array(sample_list)

    # check MCMC sample mean and variance are in confidence intervals
    assert (beta_states.mean(0) > beta_mean_conf_int[0]).all()
    assert (beta_states.mean(0) < beta_mean_conf_int[1]).all()
    assert (beta_states.var(0) > beta_var_conf_int[0]).all()
    assert (beta_states.var(0) < beta_var_conf_int[1]).all()


def test_sample_pre_activation_shape():

    n_observations = 10
    n_features = 2
    batch_shape = (3, 4)
    border = np.array([0, 1])
    slope = np.array([0, 1/40.0, 0])
    intercept = np.array([0, 0.5, 1])
    pre_act_mean = np.zeros(batch_shape + (n_observations, n_features))
    pre_act_scale = np.ones(batch_shape + (n_observations, n_features))
    post_act_observation = np.random.randn(
            *batch_shape, n_observations, n_features)
    post_act_scale = np.ones(batch_shape + (n_observations, n_features))

    samples = sample_pre_activation(
            pre_act_mean,
            pre_act_scale,
            post_act_observation,
            post_act_scale,
            border=border, slope=slope, intercept=intercept)
    assert samples.shape == batch_shape + (n_observations, n_features)


def test_sample_pre_activation_distribution():

    n_samples = int(1e4)
    n_features = 1
    border = np.array([-2, 2])
    slope = np.array([1, 1, 1])
    intercept = np.array([0, 0, 0])
    pre_act_mean = np.ones([n_samples, n_features]) * (-5.0)
    pre_act_scale = np.ones([n_samples, n_features]) * np.sqrt(1 / 0.1)
    post_act_observation = np.ones([n_samples, n_features]) * 5.0
    post_act_scale = np.ones([n_samples, n_features]) * np.sqrt(1 / 0.9)

    pvalue_threshold = 1e-3

    sample_mean_correct = ((
        pre_act_mean * pre_act_scale**(-2)
        + post_act_observation * post_act_scale**(-2))
        / (pre_act_scale**(-2) + post_act_scale**(-2)))[0, 0]
    sample_scale_correct = (pre_act_scale**(-2) + post_act_scale**(-2))**(-1/2)

    samples = sample_pre_activation(
            pre_act_mean,
            pre_act_scale,
            post_act_observation,
            post_act_scale,
            border=border, slope=slope, intercept=intercept)
    assert samples.shape == (n_samples, n_features)
    rv_correct = stats.norm(sample_mean_correct, sample_scale_correct)
    pvalue = stats.kstest(samples.flatten(), rv_correct.cdf).pvalue

    assert pvalue > pvalue_threshold


def test_sample_pre_categorization_shape():

    n_observations = 10
    n_features = 7
    n_targets = 5
    batch_shape = (2, 3)

    pre_cat_mean = np.zeros(batch_shape + (n_observations, n_features))
    pre_cat_scale = np.ones(batch_shape)
    pre_cat_observations_old = np.random.randn(
            *batch_shape, n_observations, n_features)
    post_cat_observations = np.random.choice(
            n_targets, size=batch_shape + (n_observations,))

    samples = sample_pre_categorization(
            pre_cat_mean,
            pre_cat_scale,
            pre_cat_observations_old,
            post_cat_observations)

    assert samples.shape == batch_shape + (n_observations, n_features)


@pytest.mark.parametrize('cat_correct', [0, 1, 2])
def test_sample_pre_categorization_weak_prior_correctness(cat_correct):

    pvalue_threshold = 1e-3
    n_samples = 1000
    n_features = 2
    pre_cat_mean = np.zeros((n_samples, n_features))
    pre_cat_scale = 10.0
    pre_cat_observations_old = np.zeros((n_samples, n_features))
    post_cat_observations = np.zeros(n_samples).astype(int) + cat_correct

    samples = sample_pre_categorization(
            pre_cat_mean,
            pre_cat_scale,
            pre_cat_observations_old,
            post_cat_observations)

    assert samples.shape == (n_samples, n_features)
    is_correct = hardmax(samples) == cat_correct
    pvalue = stats.binom_test(
            x=is_correct.sum(),
            n=is_correct.size,
            p=1/(n_features + 1),
            alternative='greater')
    assert pvalue < pvalue_threshold


@pytest.mark.parametrize('cat_correct', [0, 1])
def test_sample_pre_categorization_weak_prior_distribution(cat_correct):

    n_features = 1
    n_samples = 100000
    pvalue_threshold = 1e-3
    pre_cat_scale = 10.0
    large = pre_cat_scale * 3
    pre_cat_mean = np.zeros((n_samples, n_features))
    pre_cat_observations_old = np.zeros((n_samples, n_features))
    post_cat_observations = np.zeros(n_samples).astype(int) + cat_correct

    samples_all = sample_pre_categorization(
            pre_cat_mean,
            pre_cat_scale,
            pre_cat_observations_old,
            post_cat_observations)

    correct_all = {}

    # interval and correct distribution of the slope section
    sign_slope = cat_correct * 2 - 1
    correct_all['slope'] = {
            'interval': np.sort(
                np.array([0.5 / hard_sigmoid_slope, large])
                * sign_slope),
            'dist': np.random.laplace(size=n_samples)
            }

    # interval and correct distribution of the flat section
    correct_all['flat'] = {
            'interval': -correct_all['slope']['interval'][::-1],
            'dist': (
                np.random.randn(n_samples, n_features)
                * pre_cat_scale
                + pre_cat_mean).flatten()
            }

    # interval and correct distribution of the middle section
    correct_all['middle'] = {
            'interval': np.array([-0.5, 0.5]) / hard_sigmoid_slope,
            'dist': (
                np.random.randn(n_samples)
                * hard_sigmoid_slope**(-0.5)
                - 0.5 / hard_sigmoid_slope * sign_slope)
            }

    for sect, correct in correct_all.items():
        samples = samples_all[np.logical_and(
            samples_all > correct['interval'][0],
            samples_all < correct['interval'][1])]

        samples_correct = correct['dist'][np.logical_and(
            correct['dist'] > correct['interval'][0],
            correct['dist'] < correct['interval'][1])]

        assert samples.size > 0
        pvalue = stats.ks_2samp(
                samples,
                samples_correct).pvalue
        assert pvalue > pvalue_threshold


@pytest.mark.parametrize('cat_correct', [0, 1, 2])
def test_sample_pre_categorization_strong_prior(cat_correct):

    pvalue_threshold = 1e-3
    n_samples = 1000
    n_features = 2
    if cat_correct < n_features:
        pre_cat_mean = np.zeros((n_samples, n_features))
        pre_cat_mean[:, cat_correct] = 10
    else:
        pre_cat_mean = np.zeros((n_samples, n_features)) - 10
    pre_cat_scale = 1.0
    pre_cat_observations_old = (
            pre_cat_mean
            + pre_cat_scale * np.random.randn(n_samples, n_features))
    post_cat_observations = np.random.choice(n_features+1, n_samples)

    samples = sample_pre_categorization(
            pre_cat_mean,
            pre_cat_scale,
            pre_cat_observations_old,
            post_cat_observations)
    assert samples.shape == (n_samples, n_features)
    is_correct = hardmax(samples) == cat_correct
    pvalue = stats.binom_test(
            x=is_correct.sum(),
            n=is_correct.size,
            p=1/(n_features + 1),
            alternative='greater')
    assert pvalue < pvalue_threshold


def test_sample_scale_known_mean():
    # TODO: write test to ensure n_observations is correct
    pass


def test_sample_features():
    # TODO: write test to ensure precision/covariance is used correctly
    pass
