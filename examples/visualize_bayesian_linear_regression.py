#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import dalea.samplers as samplers


def fit_linear_regression(Y, X, alpha=None):
    n, q = X.shape[-2:]
    assert Y.shape[-2] == n
    df = n - q
    A = np.swapaxes(X, -1, -2) @ X
    if n < q:
        print('Warning: n < q. Adding a diagnal matrix')
        evals = np.linalg.eigh(A)[0]
        emin = evals[..., ~np.isclose(evals, 0)].min(-1)
        A += np.eye(q) * emin
        df = n
    A_inv = np.linalg.inv(A)
    assert (A_inv.diagonal(0, -2, -1) > 0).all()
    beta = A_inv @ np.swapaxes(X, -1, -2) @ Y
    sigmasq = ((Y - X @ beta)**2).sum(-2, keepdims=True) / df
    se = np.sqrt(sigmasq * A_inv.diagonal(0, -2, -1)[..., np.newaxis])
    pval = (1 - scipy.stats.t.cdf(np.abs(beta / se), df=df))*2
    if alpha is None:
        return beta, se, pval
    else:
        cutoff = se * scipy.stats.t.ppf(1 - alpha / 2, df=df)
        return beta, se, pval, cutoff


def test_sample_bayesian_linear_regression():

    np.random.seed(101)

    x = np.random.randn(3, 50, 5)
    y = x @ np.random.randn(3, 5, 16)
    y += np.random.randn(3, 50, 16) * 10
    beta_ls, se_ls = fit_linear_regression(y, x)[:2]

    beta_priocov = np.tile(np.eye(5), (3, 16, 1, 1)) * 0.1
    beta_priomea = np.zeros((3, 5, 16))
    a_prio = np.ones((3, 16))
    b_prio = np.ones((3, 16))

    # TODO: compare with results by
    # sklearn.linear_model.BayesianRidge or
    # stan_glm from library(rstanarm)

    for i in range(2):
        if i == 0:
            beta_postmea = samplers.fit_bayesian_linear_regression(
                    y, x, beta_priomea, beta_priocov, a_prio, b_prio)[0]
            beta_postsamp = samplers.sample_bayesian_linear_regression(
                    y, x, beta_priomea, beta_priocov,
                    a_prior=a_prio, b_prior=b_prio,
                    sample_shape=1000)
            plt.subplot(2, 1, 1)
            plt.title('unknown sigsq')
        elif i == 1:
            sigsq = ((y - x @ beta_ls)**2).mean(-2)
            sigsq = sigsq / 2**2  # for debugging
            beta_postmea = samplers.fit_bayesian_linear_regression(
                    y, x, beta_priomea, beta_priocov)[0]
            beta_postsamp = samplers.sample_bayesian_linear_regression(
                    y, x, beta_priomea, beta_priocov,
                    sigmasq=sigsq, sample_shape=1000)
            plt.subplot(2, 1, 2)
            plt.title('known sigsq')

        beta_ci = np.quantile(beta_postsamp, [0.025, 0.975], axis=0)
        # TODO: replace visual checks with two-sample Kolmogorov-Smirnov tests
        plt.plot(
                beta_ls.flatten(), beta_ls.flatten(),
                color='C0', label='ls mean')
        plt.plot(
                beta_ls.flatten(), (beta_ls + se_ls * 1.96).flatten(),
                'o', color='C1', label='ls ci')
        plt.plot(
                beta_ls.flatten(), (beta_ls - se_ls * 1.96).flatten(),
                'o', color='C1')
        plt.plot(
                beta_ls.flatten(), beta_postmea.flatten(),
                'o', color='C2', label='bay mean')
        plt.plot(
                beta_ls.flatten(), beta_ci.reshape(2, -1)[0],
                'o', color='C3', label='bay ci')
        plt.plot(
                beta_ls.flatten(), beta_ci.reshape(2, -1)[1],
                'o', color='C3')
        plt.axvline(0, color='C4')
        plt.axhline(0, color='C4')
        plt.legend()
    plt.show()


def main():
    # TODO: use a unit test runner
    test_sample_bayesian_linear_regression()


if __name__ == '__main__':
    main()
