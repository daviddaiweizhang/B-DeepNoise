import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.special import logsumexp


def fit_ols_manual(x, y, fit_intercept):
    if fit_intercept:
        ones = np.ones_like(x[..., [0]])
        x = np.concatenate([x, ones], -1)
    else:
        intercept = np.zeros((1, y.shape[-1]))
    slope = (
            np.linalg.inv(x.swapaxes(-1, -2) @ x)
            @ (x.swapaxes(-1, -2) @ y))
    if fit_intercept:
        intercept = slope[..., -1:, :]
        slope = slope[..., :-1, :]
    return slope, intercept


def fit_ols_single(x, y, fit_intercept):
    model = LinearRegression(fit_intercept=fit_intercept).fit(X=x, y=y)
    slope = model.coef_.reshape(-1, 1)
    intercept = np.array(model.intercept_).reshape(-1, 1)
    return slope, intercept


def fit_ols(x, y, fit_intercept):
    batch_shape = np.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    x = np.broadcast_to(x, batch_shape + x.shape[-2:])
    y = np.broadcast_to(y, batch_shape + y.shape[-2:])
    x = x.reshape(-1, *x.shape[-2:])
    y = y.reshape(-1, *y.shape[-2:])
    out = [
            fit_ols_single(xi, yi, fit_intercept=fit_intercept)
            for xi, yi in zip(x, y)]
    slope = np.stack([ou[0] for ou in out])
    intercept = np.stack([ou[1] for ou in out])

    # slop, inte = fit_ols_manual(x, y, fit_intercept=fit_intercept)
    # assert np.allclose(slop, slope)
    # assert np.allclose(inte, intercept)

    slope = slope.reshape(*batch_shape, *slope.shape[-2:])
    intercept = intercept.reshape(*batch_shape, *intercept.shape[-2:])
    return slope, intercept


def standardize(x, return_info=False):
    assert x.ndim == 2
    mean = x.mean(0)
    std = x.std(0)
    n_unique = np.array([np.unique(e).size for e in x.T])
    is_invariant = (n_unique == 1)
    std[is_invariant] = 1.0
    mean[is_invariant] = x[0, is_invariant]
    x = x - mean
    x = x / std
    x[:, is_invariant] = 0.0
    if return_info:
        out = x, (mean, std)
    else:
        out = x
    return out


def split_train_test(x, y, n_train):
    assert x.shape[0] == y.shape[0]
    n_observations = x.shape[0]
    assert 0 < n_train < n_observations
    idx = np.random.choice(
            n_observations, n_observations, replace=False)
    idx_train = idx[:n_train]
    idx_test = idx[n_train:]
    x_train = x[idx_train]
    y_train = y[idx_train]
    x_test = x[idx_test]
    y_test = y[idx_test]
    return (x_train, y_train), (x_test, y_test)


def save_qq(expected, observed, label, outfile):
    quantiles = np.linspace(0, 1, 100+2)[1:-1]
    df = {}
    if label is not None:
        df['label'] = label
    df['expected'] = np.quantile(expected, quantiles)
    df['observed'] = np.quantile(observed, quantiles)
    df = pd.DataFrame(df)
    df.to_csv(outfile, index=False, header=False, sep='\t')
    print(outfile)


def gen_splits(n_observations, n_train, n_splits, outfile):
    indices = np.array([
            np.random.choice(n_observations, n_train, replace=False)
            for __ in range(n_splits)]).T
    np.savetxt(outfile, indices, fmt='%d', delimiter='\t')
    print(outfile)


def gaussian_loss(mean, std, y, reduction=True):
    '''
    Compute the Gaussian mixture loss (negative log-likelihood)
    of the samples of a distribution with respect to an observed value.
    Axis 0 is treated as an array of `n_realizations`
    components in a Gaussian mixture distribution.
    In the special case of `n_realizations == 1`,
    the distribution is a Gaussian distribution,
    and the returnd value is the negative log-likelihood
    of a Gaussian distribution with mean `mean`
    and standard deviation `std`
    with respect to data `y`.
    Args:
        mean: Array of means.  Has shape
            (n_realizations, n_states, n_chains, n_observations, n_targets)
        std: Array of standard deviations.  Has shape
            (n_realizations, n_states, n_chains, n_observations, n_targets)
        y: Array of observed values
            Has shape (n_observations, n_targets).
        reduction: If true, take the mean across the last two axes.
    Returns:
        nll: Value of the loss function.
            If `reduction == True`, has shape
            (n_states, n_chains);
            else has shape
            (n_states, n_chains, n_observations, n_targets).
    '''
    n_observations, n_targets = y.shape[-2:]
    batch_shape = mean.shape[:-2]
    assert mean.shape[-2:] == (n_observations, n_targets)
    assert std.shape[:-2] == batch_shape
    assert std.shape[-2] in (n_observations, 1)
    assert std.shape[-1] in (n_targets, 1)

    logprob = (
            (-0.5) * np.log(2 * np.pi * std**2)
            + (-0.5) * (y - mean)**2 / std**2)
    n_samples = logprob.shape[0]
    logprob = logsumexp(logprob, axis=0) - np.log(n_samples)
    nlp = logprob * (-1)
    if reduction:
        nlp = nlp.mean((-2, -1))
    return nlp
