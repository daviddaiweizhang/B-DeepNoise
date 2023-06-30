import pytest
import numpy as np
from scipy import stats

from dalea.samplers_basic import (
        sample_heterogeneous_normal,
        fit_bayesian_linear_regression,
        sample_bayesian_linear_regression,
        fit_heteroscedastic_linear_regression,
        sample_heteroscedastic_linear_regression,
        sample_multivariate_normal
        )
import dalea.numeric as numeric
from dalea.dist_grid import normal_log_prob_grid, normal_inv_cdf_grid
from utils import fit_slope


def test_sample_heterogeneous_normal_broadcasting():
    border = np.array([-np.inf, 0.0, 1.0, np.inf])
    mu = np.zeros([5, 4, 3])
    sigma = np.ones([4, 3])
    log_denrat = np.array([0.3, 0.7])

    x = sample_heterogeneous_normal(
            border, mu, sigma, log_denrat)

    assert x.shape == (5, 4)


def test_sample_heterogeneous_normal_length_one_batch():
    border = np.array([-np.inf, 0.0, 1.0, np.inf])
    mu = np.zeros([10, 1, 3])
    sigma = np.ones([3])
    log_denrat = np.array([0.3, 0.7])

    x = sample_heterogeneous_normal(
            border, mu, sigma, log_denrat)

    assert x.shape == (10, 1)


def test_sample_heterogeneous_normal_scalar_border():
    border = np.array([-np.inf, 1, np.inf])
    mu = np.zeros([3, 2])
    sigma = np.ones([3, 2])
    log_denrat = np.array([-0.3])

    x = sample_heterogeneous_normal(
            border, mu, sigma, log_denrat)

    assert x.shape == (3,)


def test_sample_heterogeneous_normal_sample_shape():
    border = np.tile(
            np.array([-np.inf, 0.0, 1.0, np.inf]),
            [5, 4, 1])
    mu = np.zeros([5, 4, 3])
    sigma = np.ones([5, 4, 3])
    log_denrat = np.array([-0.3, 0.5])
    sample_shape = (7, 6)

    x = sample_heterogeneous_normal(
            border, mu, sigma, log_denrat, sample_shape)

    assert x.shape == (7, 6, 5, 4)


def test_sample_heterogeneous_normal_wrong_shapes():
    with pytest.raises(ValueError):
        sample_heterogeneous_normal(
                border=np.array([-np.inf, 2.0, 4.0, np.inf]),
                mean=np.array([0.0, 0.0]),
                std=np.array([1.0, 1.0, 1.0]),
                log_denrat=np.array([0.3, 0.8]))

    with pytest.raises(ValueError):
        sample_heterogeneous_normal(
                border=np.array([-np.inf, 2.0, 4.0, np.inf]),
                mean=np.array([0.0, 0.0, 0.0]),
                std=np.array([1.0, 1.0]),
                log_denrat=np.array([0.3, 0.8]))

    with pytest.raises(ValueError):
        sample_heterogeneous_normal(
                border=np.array([-np.inf, 2.0, 4.0, np.inf]),
                mean=np.array([0.0, 0.0, 0.0]),
                std=np.array([1.0, 1.0, 1.0]),
                log_denrat=np.array([0.3]))


def test_sample_heterogeneous_normal_distribution():
    border = np.array([-np.inf, 0.0, np.inf])
    mu = np.array([-1.0, 3])
    sigma = np.array([1.0, 2.0])
    log_denrat = np.array([np.log(1.5)])
    sample_shape = (1000000,)
    qq_threshold = 0.01
    pvalue_threshold = 0.001
    binwidth = sigma.min() * 0.1

    n_components = len(border) - 1
    x = sample_heterogeneous_normal(
            border, mu, sigma, log_denrat, sample_shape)

    x_segment = [
            x[np.logical_and(
                x >= border[i],
                x < border[i+1])]
            for i in range(n_components)]
    for x_s in x_segment:
        assert x_s.size > 0

    x_pool = [
            np.random.randn(*sample_shape) * s + m
            for m, s in zip(mu, sigma)]
    x_correct = [
            x_pool[i][np.logical_and(
                x_pool[i] >= border[i],
                x_pool[i] < border[i+1])]
            for i in range(n_components)]
    for x_c in x_correct:
        assert x_c.size > 0

    q = np.linspace(0, 1, 10000)[1:-1]
    for x_s, x_c in zip(x_segment, x_correct):

        z_s = np.quantile(x_s, q)
        z_c = np.quantile(x_c, q)
        qq_diff = ((z_s - z_c)**2).mean()**0.5 / x_c.std()
        assert qq_diff < qq_threshold
        assert stats.ks_2samp(x_s, x_c).pvalue > pvalue_threshold

    for bd, dlp, x_l, x_r in zip(
            border[1:-1], log_denrat,
            x_segment[:-1], x_segment[1:]):
        bin_l = x_l[(x_l > bd - binwidth) * (x_l < bd)]
        bin_r = x_r[(x_r > bd) * (x_r < bd + binwidth)]
        cnt_l = bin_l.shape[-1]
        cnt_r = bin_r.shape[-1]
        cnt_logratio = np.log(cnt_r) - np.log(cnt_l)
        assert np.isclose(cnt_logratio, dlp, rtol=0.1, atol=0.1)


# @pytest.mark.skip(reason=(
#             'Histogram approximation can only handle'
#             'values in (-8, 8)'))
@pytest.mark.parametrize('border, mu', [
    ([0.0], 10.0),
    ([-10.0], 0.0),
    ([0.0], -10.0),
    ([10.0], 0.0),
    ])
def test_sample_heterogeneous_normal_extreme(border, mu):
    sigmasq = 1.0
    log_denrat = [0.0]
    n_samples = 1000
    pvalue_threshold = 1e-3

    n_components = len(border) + 1
    mean_components = mu + np.random.randn(n_components) * 1e-3
    variance_components = sigmasq * np.exp(
            np.random.randn(n_components) * 1e-3)

    x = sample_heterogeneous_normal(
            np.array([-np.inf] + border + [np.inf]),
            mean_components,
            variance_components,
            np.array(log_denrat),
            (n_samples,))
    x_standardized = (x - mu) / np.sqrt(sigmasq)
    pvalue = stats.kstest(x_standardized, 'norm').pvalue
    assert pvalue > pvalue_threshold


def test_sample_bayesian_linear_regression():
    n_observations = 100
    n_features = 3
    n_targets = 2
    n_replicates = 80
    n_samples = 1000
    lambda_s_prior = 0.01
    a_prior = 0.01
    b_prior = 0.01
    ci_alpha = 0.05
    pval_alpha = 0.05
    beta = np.random.randn(n_replicates, n_features, n_targets) * 4
    x = np.random.randn(n_replicates, n_observations, n_features)
    noise = np.random.randn(n_replicates, n_observations, n_targets)
    y = x @ beta + noise
    mu_prior = np.zeros((n_features, n_targets))

    # cov_mat with shape (..., n_targets, n_features, n_features)
    lambda_mat_prior = np.tile(
        np.expand_dims(np.eye(n_features), 0),
        [n_targets, 1, 1]) * lambda_s_prior
    samples = sample_bayesian_linear_regression(
            y=y, x=x,
            mu_prior=mu_prior,
            lambda_mat_prior=lambda_mat_prior,
            a_prior=np.array([a_prior] * n_targets),
            b_prior=np.array([b_prior] * n_targets),
            sample_shape=(n_samples,))
    ci = np.quantile(samples, (ci_alpha/2, 1-ci_alpha/2), axis=0)
    is_in = (beta > ci[0]) * (beta < ci[1])
    assert 1 - ci_alpha*2 < is_in.mean() < 1 - ci_alpha*0.5
    sample_mean = samples.mean(0)
    pval = stats.pearsonr(sample_mean.flatten(), beta.flatten())[1]
    assert pval < pval_alpha


def test_fit_bayesian_linear_regression():
    n_observations = 100
    n_features = 3
    n_targets = 2
    n_replicates = 80
    lambda_s_prior = 0.01
    a_prior = 0.01
    b_prior = 0.01
    pval_alpha = 0.05
    beta = np.random.randn(n_replicates, n_features, n_targets)*2 + 3
    x = np.random.randn(n_replicates, n_observations, n_features)
    noise = np.random.randn(n_replicates, n_observations, n_targets)
    y = x @ beta + noise
    mu_prior = np.zeros((n_features, n_targets))

    # cov_mat with shape (..., n_targets, n_features, n_features)
    lambda_mat_prior = np.tile(
        np.expand_dims(np.eye(n_features), 0),
        [n_targets, 1, 1]) * lambda_s_prior
    beta_fit = fit_bayesian_linear_regression(
            y=y, x=x,
            mu_prior=mu_prior,
            lambda_mat_prior=lambda_mat_prior,
            a_prior=np.array([a_prior] * n_targets),
            b_prior=np.array([b_prior] * n_targets))[0]
    pval = stats.pearsonr(beta_fit.flatten(), beta.flatten())[1]
    assert pval < pval_alpha


def test_normal_log_den():
    n = 100
    x = np.random.randn(n)
    mean = np.random.randn(n)
    scale = np.exp(np.random.randn(n))
    assert np.allclose(
        numeric.normal_log_den(x, mean, scale),
        stats.norm.logpdf(x, mean, scale))


def test_normal_log_prob_safe():
    n = 100
    lower = np.random.randn(n)
    upper = lower + np.exp(np.random.randn(n))
    mean = np.random.randn(n)
    scale = np.exp(np.random.randn(n))
    logdiffcdf, logcdflower, logsfupper = numeric.normal_log_prob_safe(
            lower, upper, mean, scale,
            return_lower=True, return_upper=True)

    logdiffcdf_correct = np.log(
        stats.norm.cdf(upper, mean, scale)
        - stats.norm.cdf(lower, mean, scale))
    has_answerkey_logdiffcdf = np.isfinite(logdiffcdf_correct)
    assert np.allclose(
            logdiffcdf[has_answerkey_logdiffcdf],
            logdiffcdf_correct[has_answerkey_logdiffcdf],
            rtol=1e-3, atol=1e-3)

    logcdflower_correct = stats.norm.logcdf(lower, mean, scale)
    has_answerkey_logcdflower = np.isfinite(logcdflower_correct)
    assert np.allclose(
            logcdflower[has_answerkey_logcdflower],
            logcdflower_correct[has_answerkey_logcdflower],
            rtol=1e-3, atol=1e-3)

    logsfupper_correct = stats.norm.logsf(upper, mean, scale)
    has_answerkey_logsfupper = np.isfinite(logsfupper_correct)
    assert np.allclose(
            logsfupper[has_answerkey_logsfupper],
            logsfupper_correct[has_answerkey_logsfupper],
            rtol=1e-3, atol=1e-3)


def test_normal_log_prob_grid():
    n = 100
    u_lower = np.random.rand(n)
    u_upper = np.random.rand(n) * (1 - u_lower) + u_lower
    z_lower = stats.norm.ppf(u_lower)
    z_upper = stats.norm.ppf(u_upper)
    log_prob = normal_log_prob_grid(z_lower, z_upper)
    log_prob_correct = np.log(u_upper - u_lower)
    isfinite = log_prob != -np.inf
    assert np.allclose(
            log_prob[isfinite], log_prob_correct[isfinite],
            rtol=0.1, atol=0.01)


def test_normal_inv_cdf_grid():
    n = 100
    u = np.random.rand(n)
    z = normal_inv_cdf_grid(u)
    z_correct = stats.norm.ppf(u)
    assert np.allclose(
            z, z_correct,
            rtol=0.1, atol=0.01)


def test_fit_heteroscedastic_linear_regression_homoscedastic_noise():
    n_observations = 100
    n_features = 3
    n_targets = 2
    n_replicates = 80
    pval_threshold = 0.05
    sigmasq = 4.0  # noise precision
    mu = 0.0  # prior mean
    kappasq = 0.01  # prior precision

    sigmasq = sigmasq * np.ones((n_observations, n_targets))
    mu = mu + np.zeros((n_features, n_targets))
    kappasq = kappasq * np.ones((n_features, n_targets))
    beta = np.random.randn(n_replicates, n_features, n_targets)*2 + 3
    x = np.random.randn(n_replicates, n_observations, n_features)
    noise = np.random.randn(n_replicates, n_observations, n_targets)
    y = x @ beta + noise * np.sqrt(sigmasq)

    # cov_mat with shape (..., n_targets, n_features, n_features)
    beta_fit, lamda_fit = fit_heteroscedastic_linear_regression(
            y=y.swapaxes(-1, -2),
            x=np.expand_dims(x, -3),
            sigmasq=sigmasq.swapaxes(-1, -2),
            mu=mu.swapaxes(-1, -2),
            kappasq=kappasq.swapaxes(-1, -2))
    beta_fit = beta_fit.swapaxes(-1, -2)
    assert beta_fit.shape == beta.shape
    pval_beta = stats.pearsonr(beta_fit.flatten(), beta.flatten())[1]
    assert pval_beta < pval_threshold
    slope = fit_slope(beta.flatten(), beta_fit.flatten())
    assert 0.95 < slope < 1.05
    xtx = np.tile(
            np.expand_dims(x.swapaxes(-1, -2) @ x, -3),
            (n_targets, 1, 1))
    pval_lamda = stats.pearsonr(lamda_fit.flatten(), xtx.flatten())[1]
    assert pval_lamda < pval_threshold
    lamda_correct = xtx * np.expand_dims(sigmasq[0], (-1, -2))
    idx = list(range(n_features))
    lamda_correct[..., idx, idx] += kappasq.swapaxes(-1, -2)
    assert np.allclose(lamda_fit, lamda_correct)


def test_sample_heteroscedastic_linear_regression_homoscedastic_noise():
    n_observations = 100
    n_features = 7
    n_targets = 1
    n_replicates = 5
    n_samples = 1000
    pval_threshold = 0.05
    sigmasq = 4.0  # noise precision
    mu = 0.0  # prior mean
    kappasq = 0.01  # prior precision

    sigmasq = sigmasq * np.ones((n_observations, n_targets))
    mu = mu + np.zeros((n_features, n_targets))
    kappasq = kappasq * np.ones((n_features, n_targets))
    beta = np.random.randn(n_replicates, n_features, n_targets)*2 + 3
    x = np.random.randn(n_replicates, n_observations, n_features)
    noise = np.random.randn(n_replicates, n_observations, n_targets)
    y = x @ beta + noise * np.sqrt(sigmasq)

    beta_states = sample_heteroscedastic_linear_regression(
            y=y.swapaxes(-1, -2),
            x=np.expand_dims(x, -3),
            sigmasq=sigmasq.swapaxes(-1, -2),
            mu=mu.swapaxes(-1, -2),
            kappasq=kappasq.swapaxes(-1, -2),
            sample_shape=(n_samples,))
    beta_states = beta_states.swapaxes(-1, -2)
    beta_fit = beta_states.mean(0)
    lamda_fit = np.stack([
        np.linalg.inv(np.cov(a.T))
        for a in beta_states[..., 0].swapaxes(0, 1)])
    assert beta_fit.shape == beta.shape
    pval_beta = stats.pearsonr(beta_fit.flatten(), beta.flatten())[1]
    assert pval_beta < pval_threshold
    slope_beta = fit_slope(beta.flatten(), beta_fit.flatten())
    assert 0.95 < slope_beta < 1.05
    xtx = np.tile(
            np.expand_dims(x.swapaxes(-1, -2) @ x, -3),
            (n_targets, 1, 1))
    pval_lamda = stats.pearsonr(lamda_fit.flatten(), xtx.flatten())[1]
    assert pval_lamda < pval_threshold
    lamda_correct = xtx * np.expand_dims(sigmasq[0], (-1, -2))
    lamda_correct = lamda_correct[..., 0, :, :]
    idx = list(range(n_features))
    lamda_correct[..., idx, idx] += kappasq.swapaxes(-1, -2)
    slope_lamda = fit_slope(
            lamda_correct.flatten(),
            lamda_fit.flatten())
    assert 0.95 < slope_lamda < 1.05


def test_fit_heteroscedastic_linear_regression_heteroscedastic_noise():
    # TODO: write test to check posterior mean and covariance are
    # correct when noise has difference variance per observation
    pass


def test_sample_multivariate_normal():
    n_samples = 10000
    mean = np.array([0.5, -0.2])
    cov = np.array([[2.0, 0.3], [0.3, 0.5]])
    scale = np.linalg.cholesky(cov)
    assert np.allclose(scale @ scale.T, cov)
    samples = sample_multivariate_normal(
            loc=mean, scale=scale, sample_shape=(n_samples,))
    assert np.allclose(samples.mean(0), mean, 0.05)
    assert np.allclose(np.cov(samples.T), cov, 0.05)
