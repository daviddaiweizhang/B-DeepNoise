import numpy as np
from scipy import stats
from scipy.special import logsumexp
import warnings
import tensorflow_probability as tfp

from dalea.dist_grid import (
        normal_log_prob_grid, normal_inv_cdf_grid)
from dalea.numeric import (
        normal_log_prob_safe, normal_log_den,
        normal_inv_cdf, normal_inv_sf, log1mexp)
from dalea.utils import is_broadcastable

tfd = tfp.distributions


# grid approximation of normal distribution
def get_normal_grid_z(size, x_max, x_inf):

    assert 0 < x_max < x_inf
    # use N(0, 1) within (-3, 3) to specify the spacing of z
    z = stats.norm.ppf(np.linspace(0, 1, size)[1:-1]) * x_max / 3.0
    z = np.concatenate([[-x_inf], z, [x_inf]])
    u = stats.norm.cdf(z)
    assert u[0] == 0.0
    assert u[-1] == 1.0
    return z, u


def take_expand_dims(arr, indices, axis):
    bshape = np.broadcast(arr, indices).shape
    arr_extra_dims = np.expand_dims(
            arr, list(range(len(bshape) - arr.ndim)))
    indices_extra_dims = np.expand_dims(
            indices, list(range(len(bshape) - indices.ndim)))
    return np.take_along_axis(arr_extra_dims, indices_extra_dims, axis)


def sample_heteroscedastic_linear_regression(sample_shape=(), **kwargs):
    '''
    Posterior sampler of heteroscedastic Bayesian linear regression.
    See `fit_heteroscedastic_linear_regression`
    for further information.
    Args:
        sample_shape: A tuple to indicate the shape of samples to draw.
    Returns:
        sample: A sample from the posterior distribution of the effects.
            Has shape `sample_shape + (..., n_features)`.
    '''
    out = fit_heteroscedastic_linear_regression(
            return_decomposition=True,
            **kwargs)
    loc, precision, precision_d, precision_u = out
    precision_d = np.clip(precision_d, 1e-15, None)
    # TODO: check whether precision, precision_u can be negative
    scale = (
            precision_u
            / np.expand_dims(np.sqrt(precision_d), -2))
    # assert np.allclose(
    #         scale @ scale.swapaxes(-1, -2),
    #         np.linalg.inv(precision))
    kernel_sample = sample_multivariate_normal(
        loc=loc,
        scale=scale,
        sample_shape=sample_shape)
    return kernel_sample


def fit_heteroscedastic_linear_regression(
        y, x, sigmasq, mu, kappasq, return_decomposition=False):
    '''
    Fit Bayesian linear regression
    with independent but heteroscedastic noise.
    The model is
    ```none
        beta ~ MVN(mu, diag(kappasq))
        y | beta ~ MVN(x beta, diag(sigmasq)),
    ```
    which gives the posterior conditional distribution
    ```none
        beta | y ~ MVN(nu, lamda^(-1))
        lamda = diag(1/kappasq) + x^T diag(1/sigmasq) x
        nu = (
            lamda^(-1)
            (x^T diag(1/sigmasq) y + diag(1/kappasq) mu))
    ```
    See ``Pattern Recognition and Machine Learning``
    by Christopher Bishop, Sec 2.3.3
    Args:
        y: Observations of the target variables.
            Has shape `(..., n_observations)`.
        x: Observations of the feature variables.
            Has shape `(..., n_observations, n_features)`.
        sigmasq: Precision of the noise.
            Has shape `(..., n_observations)`.
        mu: Prior mean of the effects.
            Has shape `(..., n_features)`.
        kappasq: Prior precision of the effects.
            Has shape `(..., n_features)`.
    Returns:
        nu: Posterior mean of the effects.
            Has shape `(..., n_features)`.
        lamda: Posterior precision matrix of the effects.
            Has shape `(..., n_features, n_features)`.
    '''

    n_observations = y.shape[-1]
    n_features = x.shape[-1]
    assert x.shape[-2] == n_observations
    assert sigmasq.shape[-1] in (n_observations, 1)
    assert mu.shape[-1] == n_features
    assert kappasq.shape[-1] == n_features

    y = np.expand_dims(y, -1)
    sigmasq = np.expand_dims(sigmasq, -1)
    mu = np.expand_dims(mu, -1)
    kappasq = np.expand_dims(kappasq, -1)
    c = x * sigmasq
    lamda = c.swapaxes(-1, -2) @ x
    di = np.arange(n_features)  # indices of the diagonal
    lamda[..., di, di] += kappasq[..., 0]
    lamda_d, lamda_u = np.linalg.eigh(lamda)
    covariance = lamda_u @ (
            lamda_u.swapaxes(-1, -2)
            / lamda_d[..., np.newaxis])
    nu = covariance @ (c.swapaxes(-1, -2) @ y + mu * kappasq)
    nu = nu[..., 0]
    out = (nu, lamda)
    if return_decomposition:
        out += (lamda_d, lamda_u)
    return out


# TODO: change to taking precision_mat as input
# Currently, np.linalg.inv takes most of the time
def fit_bayesian_linear_regression(
        y, x, mu_prior, lambda_mat_prior,
        a_prior=None, b_prior=None):

    """ Update conjugate prior of Bayesian linear regression.

    See `sample_bayesian_linear_regression`.
    """

    sigmasq_is_known = (
            a_prior is None
            or b_prior is None)
    N, Q = x.shape[-2:]
    K = y.shape[-1]

    assert y.shape[-2] == N
    assert mu_prior.shape[-2:] == (Q, K)
    assert lambda_mat_prior.shape[-3:] in [
            (K, Q, Q), (1, Q, Q)]
    input_batch_shape_list = [
            x.shape[:-2],
            y.shape[:-2],
            mu_prior.shape[:-2],
            lambda_mat_prior.shape[:-3]]
    if not sigmasq_is_known:
        assert np.shape(a_prior)[-1] == K
        assert np.shape(b_prior)[-1] == K
        input_batch_shape_list += [
                np.shape(a_prior)[:-1],
                np.shape(b_prior)[:-1]]
    assert is_broadcastable(*input_batch_shape_list)

    mu_prior_colvec = np.swapaxes(mu_prior, -1, -2)[..., np.newaxis]
    x_mult_x = np.swapaxes(x, -1, -2) @ x
    lambda_mat_posterior = (
            x_mult_x[..., np.newaxis, :, :]
            + lambda_mat_prior)
    y_mult_x = np.swapaxes(y, -1, -2) @ x
    y_mult_x_colvec = y_mult_x[..., np.newaxis]
    mu_posterior_colvec = (
            np.linalg.inv(lambda_mat_posterior)
            @ (
                lambda_mat_prior @ mu_prior_colvec
                + y_mult_x_colvec))

    if not sigmasq_is_known:
        y_mult_y = (y**2).sum(-2)
        mu_mult_lambda_mat_mult_mu_prior_extra_dims = (
                np.swapaxes(mu_prior_colvec, -1, -2)
                @ lambda_mat_prior @ mu_prior_colvec)
        mu_mult_lambda_mat_mult_mu_posterior_extra_dims = (
                np.swapaxes(mu_posterior_colvec, -1, -2)
                @ lambda_mat_posterior @ mu_posterior_colvec)
        mu_mult_lambda_mat_mult_mu_prior = (
                mu_mult_lambda_mat_mult_mu_prior_extra_dims[..., 0, 0])
        mu_mult_lambda_mat_mult_mu_posterior = (
                mu_mult_lambda_mat_mult_mu_posterior_extra_dims[..., 0, 0])
        a_posterior = a_prior + 0.5 * N
        b_posterior = b_prior + 0.5 * (
                y_mult_y
                + mu_mult_lambda_mat_mult_mu_prior
                - mu_mult_lambda_mat_mult_mu_posterior)
    else:
        a_posterior = None
        b_posterior = None

    mu_posterior = np.swapaxes(mu_posterior_colvec[..., 0], -1, -2)
    return mu_posterior, lambda_mat_posterior, a_posterior, b_posterior


def sample_multivariate_normal(
        loc, scale, sample_shape=()):
    """ Sample from a multivariate normal distribution.
    Args:
        loc: Mean of the distribution.
            Has shape `(..., p)`.
        scale: A decomposition of the covariance matrix,
            such that the covariance is `Sigma = scale @ scale.T`.
            In other words, scale can be a cholesky decomposition
            of the covariance matrix.
            Has shape `(..., p, p)`.
        sample_shape: Shape of the random samples.
            This parameter controls the number and shape
            of the samples drawn from the distribution.
    Returns:
        sample: A random sample from the distribution.
    """

    assert scale.shape[-1] == loc.shape[-1]
    assert scale.shape[-2] == loc.shape[-1]
    batch_event_shape = np.broadcast(loc, scale[..., 0]).shape
    z = np.random.randn(*sample_shape, *batch_event_shape)
    sample = (scale @ z[..., np.newaxis])[..., 0] + loc
    return sample


def sample_bayesian_linear_regression(
        y, x, mu_prior, lambda_mat_prior, sigmasq=None,
        a_prior=None, b_prior=None, sample_shape=()):

    """ Sample posterior parameters from Bayesian linear regression

    The model is
    ```none
        y = x beta + epsilon
        beta ~ MultivariateNormal(mu_prior, sigmasq * lambda_mat_prior^(-1))
        epsilon ~ Normal(0, sigmasq)
        sigmasq ~ InverseGamma(a_prior, b_prior)
    ```
    If noise variance `sigmsq` is provided, it is assumed to be known.
    Otherwise, given, `sigmsq` is assumed to have
    a prior inverse gamma distribution with parameters
    `a_prior` and `b_prior`.
    See https://en.wikipedia.org/wiki/Bayesian_linear_regression

    Args:
        y: Observations of the target variables.
            Has shape `(..., n_observations, n_targets)`.
        x: Observations o the feature variables.
            Has shape `(..., n_observations, n_features)`.
        mu_prior: Prior mean of the effects.
            Has shape `(..., n_features, n_targets)`.
        lambda_mat_prior: Prior precision matrix of the effects
            divided by the variance of the noise.
            Has shape `(..., n_targets, n_features, n_features)`.
        sigmasq: Variance of the noise, if known.
            Has shape `(..., n_targets)`.
        a_prior: Shape parameter
            in the prior (inverse gamma) distribution of `sigmasq`.
            Ignored if `sigmasq` is provided.
            Has shape `(..., n_targets)`.
        b_prior: Scale parameter
            in the prior (inverse gamma) distribution of `sigmasq`.
            Ignored if `sigmasq` is provided.
            Has shape `(..., n_targets)`.
        sample_shape: Shape of the random samples.
    Returns:
        beta_sample: A random sample from the posterior distribution
            of the effects.
            The returnd value has shape
            `sample_shape + batch_shape + event_shape`,
            where `event_shape == (n_features, n_targets)`
            and `batch_shape` is determined by
            brodcasting the data and the prior parameters.
    """

    if sigmasq is not None:
        sigmasq_is_known = True
    elif a_prior is not None or b_prior is not None:
        sigmasq_is_known = False
    else:
        raise ValueError(
            'sigmasq or its prior distribution must be specified')

    input_batch_shape_list = [
            x.shape[:-2],
            y.shape[:-2],
            mu_prior.shape[:-2],
            lambda_mat_prior.shape[:-3]]
    if sigmasq_is_known:
        input_batch_shape_list += [sigmasq.shape[:-1]]
    else:
        input_batch_shape_list += [
                np.shape(a_prior)[:-1],
                np.shape(b_prior)[:-1]]
    assert is_broadcastable(*input_batch_shape_list)

    if sigmasq_is_known:
        (mu_posterior,
         lambda_mat_posterior) = fit_bayesian_linear_regression(
                 y, x, mu_prior, lambda_mat_prior)[:2]
        sample_shape_adjusted = sample_shape
    else:
        (mu_posterior,
         lambda_mat_posterior,
         a_posterior,
         b_posterior) = fit_bayesian_linear_regression(
                y, x, mu_prior, lambda_mat_prior,
                a_prior, b_prior)
        # sigmasq_posterior_distribution = tfd.InverseGamma(
        #         a_posterior, b_posterior)
        # sigmasq = sigmasq_posterior_distribution.sample(sample_shape).numpy()
        sigmasq = sample_invgamma(
                a_posterior, b_posterior,
                sample_shape=sample_shape)
        sample_shape_adjusted = ()

    mu_posterior_transposed = np.swapaxes(mu_posterior, -1, -2)
    # TODO: avoid cholesky by letting
    # fit_bayesian_linear_regression return decomposed
    # directly. This could be possible for simple
    # prior precision_mat

    sigmasq_extra_dims = sigmasq[..., np.newaxis, np.newaxis]

    # eigendecomposition of precision matrix
    lambda_w, lambda_v = np.linalg.eigh(lambda_mat_posterior)
    # cholesky decomposition of covariance matrix
    # lambda_v is its own inverse
    kappa_a = lambda_v / np.sqrt(lambda_w[..., np.newaxis, :])
    beta_sample_transposed = sample_multivariate_normal(
        loc=mu_posterior_transposed,
        scale=np.sqrt(sigmasq_extra_dims) * kappa_a,
        sample_shape=sample_shape_adjusted)

    # # Same distribution as above but much slower
    # kappa_mat_posterior = np.linalg.inv(lambda_mat_posterior)
    # scale_tril = tf.linalg.cholesky(kappa_mat_posterior * sigmasq_extra_dims)
    # beta_posterior_distribution_transposed = tfd.MultivariateNormalTriL(
    #         loc=mu_posterior_transposed,
    #         scale_tril=scale_tril)
    # beta_sample_transposed_1 = (
    #         beta_posterior_distribution_transposed
    #         .sample(sample_shape_adjusted).numpy())

    beta_sample = np.swapaxes(beta_sample_transposed, -1, -2)
    return beta_sample


def sample_mixture_adjacent_truncated_normal(
        border, mu, sigmasq, pi, sample_shape=()):
    """ Sample from a mixture of adjacent truncated normal distributions

    This is the same as `sample_heterogeneous_normal_special`
    except that the density function is not necessarily continuous.
    The interval between `border[..., j-1]` and `border[..., j]`
    has is assigned with a probability of pi[..., j].
    """
    if mu.shape[-1] != sigmasq.shape[-1]:
        raise ValueError(
            "The rightmost dimension of mu and sigmasq don't match:"
            f"mu: [..., {mu.shape[-1]}], sigmasq: [..., {sigmasq.shape[-1]}].")
    if border.shape[-1] != mu.shape[-1] - 1:
        raise ValueError(
            f"The rightmost dimension of border and mu (sigmasq) don't match: "
            f"mu: [..., {mu.shape[-1]}], border: [..., {border.shape[-1]}].")

    # # TODO: handle extreme values
    # dist_mu_border_max = 100
    # dist_mu_border = np.max(np.stack([
    #         (border - mu[..., :-1]) / sigmasq[..., :-1],
    #         (border - mu[..., 1:]) / sigmasq[..., 1:]
    #         ]))
    # if dist_mu_border > dist_mu_border_max:
    #     raise ValueError('At least one border is too far away from mu')

    border = concatenate_broadcast(
        [np.array([-np.inf]), border, np.array([np.inf])], )
    lowers = border[..., :-1]
    uppers = border[..., 1:]
    stds = np.sqrt(sigmasq)

    lower_tn = np.nan_to_num(lowers, nan=-np.inf)
    upper_tn = np.nan_to_num(uppers, nan=np.inf)
    tn = tfd.TruncatedNormal(loc=mu, scale=stds, low=lower_tn, high=upper_tn)
    mx = tfd.Categorical(probs=pi)
    # TODO: Improve speed.  # It accounts for 25% of
    # runtime of sample_heterogeneous_normal_special
    hn = tfp.distributions.MixtureSameFamily(
        mixture_distribution=mx, components_distribution=tn)

    # TODO: verify it's tensorflow's internal problem
    with warnings.catch_warnings():
        warnings.filterwarnings(
                'ignore', 'Falling back to stateful sampling')
        # TODO: try to improve speed
        hn_sample = hn.sample(sample_shape).numpy()
    return hn_sample


def sample_heterogeneous_normal(
        border, mean, std, log_denrat, sample_shape=()):
    """ Sample from a heterogeneous normal distribution.

    The rightmost axis of `border`
    specifies a length-`(J+1)` array
    with the elements in an ascending order,
    which partitions the real line into `J` intervals.
    Inside the interval
    between `border[..., j]` and `border[..., j+1]`,
    the random variable follows a normal distribution with
    mean `mu[..., j]` and standard deviation `std[..., j]`.
    At boundary `border[..., j]` (for `j` in `{1, ..., J-1}`),
    the difference of the right-hand limit of the log PDF
    from the left-hand limit of the log PDF is
    `log_denrat[j]`.

    Args:
        border: Has shape `(..., J+1)`.
            A sequence of ascending float numbers
            (in the rightmost axis)
            that specify the borders of the partition
            of the real line.
        mean:
            Means of the distriubtions inside the components.
            Has shape `(..., J)`.
        std: Standard deviations of the distriubtions
            inside the components.
            Has shape `(..., J)`.
        log_denrat: Log of the density ratio between the
            right component and the left component at each border.
            (Left is the reference.)
            Has shape `(..., J-1)`.
        sample_shape: Shape of the generated sample,
            in addition to the existing batch shape of
            the parameters.
            A tuple of positive integers.

    Returns:
        A sample from the heterogeneous normal distribution.
    """

    n_components = border.shape[-1] - 1
    if np.shape(log_denrat)[-1] != n_components - 1:
        raise ValueError(
                'Rightmost axis of `log_denrat` '
                f'does not have {n_components-1} elements.')
    if np.shape(mean)[-1] != n_components:
        raise ValueError(
                'Rightmost axis of `mean` '
                f'does not have {n_components} elements.')
    if np.shape(std)[-1] != n_components:
        raise ValueError(
                'Rightmost axis of `std` '
                f'does not have {n_components} elements.')

    z_lower = (border[..., :-1] - mean) / std
    z_upper = (border[..., 1:] - mean) / std

    # find the amplification needed to obtain target log density ratio
    log_prob_standard = normal_log_prob_grid(z_lower, z_upper)
    log_denrat_standard = (
            normal_log_den(z_lower[..., 1:])
            - normal_log_den(z_upper[..., :-1]))
    log_stdrat = (
            np.log(std[..., 1:]) - np.log(std[..., :-1]))
    log_amp = np.cumsum(log_denrat - log_denrat_standard + log_stdrat, -1)
    log_amp = np.concatenate(
            [np.zeros_like(log_amp[..., [0]]), log_amp], axis=-1)

    # compute the probability of each component
    log_weight = log_prob_standard + log_amp
    log_weightsum = logsumexp(log_weight, axis=-1, keepdims=True)
    prob = np.exp(log_weight - log_weightsum)
    prob_cumuupper = np.cumsum(prob, -1)

    # choose a component
    v0 = np.random.rand(*sample_shape, *prob.shape[:-1], 1)
    j = (v0 > prob_cumuupper).sum(-1, keepdims=True)

    # choose a grid point inside the component
    z_lower_chosen = take_expand_dims(z_lower, j, -1)
    u_lower_chosen = np.exp(normal_log_prob_grid(-np.inf, z_lower_chosen))
    u_diff_chosen = take_expand_dims(np.exp(log_prob_standard), j, -1)
    # # save a round of uniform sampling
    # v1 = (
    #         (take_expand_dims(mixprob_cumuupper, j, -1) - v0)
    #         / take_expand_dims(mixprob, j, -1))
    v1 = np.random.rand(*z_lower_chosen.shape)
    u_sample = u_lower_chosen + u_diff_chosen * v1
    z_sample, z_sample_info = normal_inv_cdf_grid(u_sample, return_info=True)

    # replace infinity with exponential samples
    # Ref: "Fast simulation of truncated Gaussian distributions"
    # by Nicolas Chopin (2011).
    isinf_neg = z_sample == -np.inf
    isinf_pos = z_sample == np.inf
    z_sample_limneg = z_sample_info['finite_min']
    z_sample_limpos = z_sample_info['finite_max']
    z_sample[isinf_neg] = z_sample_limneg - np.random.exponential(
            scale=-1/z_sample_limneg, size=isinf_neg.sum())
    z_sample[isinf_pos] = z_sample_limpos + np.random.exponential(
            scale=1/z_sample_limpos, size=isinf_pos.sum())

    # find the corresponding point on the target distribution
    mean_chosen = take_expand_dims(mean, j, -1)
    std_chosen = take_expand_dims(std, j, -1)
    x = z_sample * std_chosen + mean_chosen
    x = x[..., 0]

    return x


def sample_heterogeneous_normal_grid_naive(
        border, mean, std, log_denrat, sample_shape=()):
    """ Sample from a heterogeneous normal distribution.

    The rightmost axis of `border`
    specifies a length-`(J+1)` array
    with the elements in an ascending order,
    which partitions the real line into `J` intervals.
    Inside the interval
    between `border[..., j]` and `border[..., j+1]`,
    the random variable follows a normal distribution with
    mean `mu[..., j]` and standard deviation `std[..., j]`.
    At boundary `border[..., j]` (for `j` in `{1, ..., J-1}`),
    the difference of the right-hand limit of the log PDF
    from the left-hand limit of the log PDF is
    `logpdf_diff[j]`.

    Args:
        border: Has shape `(..., J+1)`.
            A sequence of ascending float numbers
            (in the rightmost axis)
            that specify the borders of the partition
            of the real line.
        mean:
            Means of the distriubtions inside the components.
            Has shape `(..., J)`.
        std: Standard deviations of the distriubtions
            inside the components.
            Has shape `(..., J)`.
        log_denrat: Difference of the log PDF between the
            right component and the left component at each border.
            (Left is the reference.)
            Has shape `(..., J-1)`.
        sample_shape: Shape of the generated sample,
            in addition to the existing batch shape of
            the parameters.
            A tuple of positive integers.

    Returns:
        A sample from the heterogeneous normal distribution.
    """

    n_components = border.shape[-1] - 1
    if np.shape(log_denrat)[-1] != n_components - 1:
        raise ValueError(
                'Rightmost axis of `difflogpdf` '
                f'does not have {n_components-1} elements.')
    if np.shape(mean)[-1] != n_components:
        raise ValueError(
                'Rightmost axis of `mean` '
                f'does not have {n_components} elements.')
    if np.shape(std)[-1] != n_components:
        raise ValueError(
                'Rightmost axis of `std` '
                f'does not have {n_components} elements.')

    # temprarily turned off
    # z = normal_grid_z
    # u = normal_grid_u
    z = None
    u = None

    lower = (border[..., :-1] - mean) / std
    upper = (border[..., 1:] - mean) / std

    i_lower = (z < lower[..., np.newaxis]).sum(-1)
    i_upper = (z < upper[..., np.newaxis]).sum(-1) - 1
    if not (i_lower < i_upper).all():
        raise ValueError(
                'Normal distribution approximation overflow/underflow.')

    # approximate density ratios at borders before amplification
    dz_upper = z[i_upper[..., :-1]] - z[i_upper[..., :-1]-1]
    dz_lower = z[i_lower[..., 1:]+1] - z[i_lower[..., 1:]]
    du_upper = u[i_upper[..., :-1]] - u[i_upper[..., :-1]-1]
    du_lower = u[i_lower[..., 1:]+1] - u[i_lower[..., 1:]]
    log_denrat_current = (
            (
                np.log(du_lower)
                - np.log(dz_lower)
                - np.log(std[..., 1:]))
            - (
                np.log(du_upper)
                - np.log(dz_upper)
                - np.log(std[..., :-1])))
    log_amp = np.cumsum(log_denrat - log_denrat_current, -1)
    log_amp = np.concatenate(
            [np.zeros_like(log_amp[..., [0]]), log_amp], axis=-1)

    # compute the probability of each component
    weight = (u[i_upper] - u[i_lower]) * np.exp(log_amp)
    mixprob = weight / weight.sum(-1, keepdims=True)
    mixprob_cumuupper = np.cumsum(mixprob, -1)

    # choose a component
    v0 = np.random.rand(*sample_shape, *mixprob.shape[:-1], 1)
    j = (v0 > mixprob_cumuupper).sum(-1, keepdims=True)

    # choose a grid point inside the component
    ilo = take_expand_dims(i_lower, j, -1)
    iup = take_expand_dims(i_upper, j, -1)
    # save a round of uniform sampling
    # v1 = (
    #         (take_expand_dims(mixprob_cumuupper, j, -1) - v0)
    #         / take_expand_dims(mixprob, j, -1))
    v1 = np.random.rand(*ilo.shape)
    u_sample = u[ilo] * (1-v1) + u[iup] * v1
    i_sample = (u_sample > u).sum(-1, keepdims=True) - 1

    # save a round of uniform sampling
    # v2 = u_sample - u[i_sample]  # incorrect, maybe off by one
    v2 = np.random.rand(*i_sample.shape)
    # sample uniformly between the chosen grid point
    # and the next grid point
    z_sample = z[i_sample] * (1-v2) + z[i_sample+1] * v2

    # find the corresponding point on the target distribution
    mn = take_expand_dims(mean, j, -1)
    st = take_expand_dims(std, j, -1)
    x = z_sample * st + mn
    x = x[..., 0]

    return x


def sample_heterogeneous_normal_invcdf(
        border, mean, scale, difflogpdf, sample_shape=()):
    """ Sample from a heterogeneous normal distribution.

    The rightmost axis of `border`
    specifies a length-`(J+1)` array
    with the elements in an ascending order,
    which partitions the real line into `J` intervals.
    Inside the interval
    between `border[..., j]` and `border[..., j+1]`,
    the random variable follows a normal distribution with
    mean `mu[..., j]` and standard deviation `scale[..., j]`.
    At boundary `border[..., j]` (for `j` in `{1, ..., J-1}`),
    the difference of the right-hand limit of the log PDF
    from the left-hand limit of the log PDF is
    `logpdf_diff[j]`.

    Args:
        border: Has shape `(..., J+1)`.
            A sequence of ascending float numbers
            (in the rightmost axis)
            that specify the borders of the partition
            of the real line.
        mean:
            Means of the distriubtions inside the components.
            Has shape `(..., J)`.
        scale: Variances of the distriubtions
            Standard deviations of the distriubtions
            inside the components.
            Has shape `(..., J)`.
        difflogpdf: Difference of the log PDF between the
            right component and the left component.
            (Left is the reference.)
            Has shape `(..., J-1)`.
        sample_shape: Shape of the generated sample,
            in addition to the existing batch shape of
            the parameters.
            A tuple of positive integers.

    Returns:
        A sample from the heterogeneous normal distribution.
    """

    n_components = border.shape[-1] - 1
    if np.shape(difflogpdf)[-1] != n_components - 1:
        raise ValueError(
                'Rightmost axis of `difflogpdf` '
                f'does not have {n_components-1} elements.')
    if np.shape(mean)[-1] != n_components:
        raise ValueError(
                'Rightmost axis of `mean` '
                f'does not have {n_components} elements.')
    if np.shape(scale)[-1] != n_components:
        raise ValueError(
                'Rightmost axis of `scale` '
                f'does not have {n_components} elements.')

    (
            logdiffcdf_standard,
            logcdflower_standard,
            logsfupper_standard) = normal_log_prob_safe(
            border[..., :-1], border[..., 1:], mean, scale,
            return_lower=True, return_upper=True)
    difflogpdf_standard = (
            normal_log_den(
                border[..., 1:-1],
                mean[..., 1:],
                scale[..., 1:])
            - normal_log_den(
                border[..., 1:-1],
                mean[..., :-1],
                scale[..., :-1]))
    logamp = np.cumsum(
            difflogpdf - difflogpdf_standard, axis=-1)
    logamp = np.concatenate(
            [np.zeros_like(logamp[..., [0]]), logamp], -1)
    weight = np.exp(logdiffcdf_standard + logamp)
    pi = weight / weight.sum(-1, keepdims=True)
    cumupi_upper = np.cumsum(pi, axis=-1)
    cumupi_lower = np.concatenate([
        np.zeros_like(cumupi_upper[..., [0]]),
        cumupi_upper[..., :-1]],
        axis=-1)
    batch_shape = np.broadcast(
            border[..., 0], mean[..., 0],
            scale[..., 0], difflogpdf[..., 0]).shape
    # quantile in heterogeneous normal distribution
    quantile_global = np.random.rand(*sample_shape, *batch_shape, 1)
    idx = (quantile_global > cumupi_lower).sum(-1, keepdims=True) - 1
    # quantile within component distribution
    mean_local = take_expand_dims(mean, idx, axis=-1)
    scale_local = take_expand_dims(scale, idx, axis=-1)
    quantile_lower_global = take_expand_dims(
            cumupi_lower, idx, axis=-1)
    pi_local = take_expand_dims(pi, idx, axis=-1)
    diffcdf_local = take_expand_dims(
            np.exp(logdiffcdf_standard), idx, axis=-1)
    # amp_local = take_expand_dims(np.exp(logamp), idx, axis=-1)
    quantile_lower_local = take_expand_dims(
            np.exp(logcdflower_standard), idx, axis=-1)
    quantile_local_cdf = (
            (quantile_global - quantile_lower_global)
            / pi_local * diffcdf_local
            + quantile_lower_local)
    x_cdf = normal_inv_cdf(quantile_local_cdf, mean_local, scale_local)

    x = x_cdf
    if not np.isfinite(x).all():
        quantile_upper_global = take_expand_dims(
                cumupi_upper, idx, axis=-1)
        iquantile_upper_local = take_expand_dims(
                np.exp(logsfupper_standard), idx, axis=-1)
        iquantile_local_sf = (
                (quantile_upper_global - quantile_global)
                / pi_local * diffcdf_local
                + iquantile_upper_local)
        x_sf = normal_inv_sf(iquantile_local_sf, mean_local, scale_local)
        x[~np.isfinite(x)] = x_sf[~np.isfinite(x)]
    if not np.isfinite(x).all():
        raise ValueError('Inverse CDF sampling underflow')
    x = x[..., 0]
    return x


# TODO: make this faster by using piecewise linear form for activation
def sample_heterogeneous_normal_special(border, mu, sigmasq, sample_shape=()):
    """ Sample from a heterogeneous normal distribution.

    The rightmost axis of `border`
    specifies a length-`(J-1)` array
    with the elements in an ascending order,
    which partitions the real line into `J` intervals.
    Inside the interval
    between `border[..., j-1]` and `border[..., j]`,
    the random variable follows
    a truncated normal distribution with (unstandardized)
    mean `mu[..., j]` and variance `sigmasq[..., j]`.
    The weights of the intervals are assigned
    so that the overall density is a continuous function.

    Args:
        border: Specifies the borders of the adjacent
            truncated normal distributions.
            The rightmost dimension has length `J-1`.
        mu: Means of the distriubtions inside the intervals.
            The rightmost dimension has length `J`.
        sigmasq: Variances of the distriubtions
            inside the intervals.
            The rightmost dimension has length `J`.
        sample_shape: Shape of the generated sample.

    Returns:
        x: A sample from the heterogeneous normal distribution
    """
    if mu.shape[-1] != sigmasq.shape[-1]:
        raise ValueError(
            "The rightmost dimension of mu and sigmasq don't match:"
            f"mu: [..., {mu.shape[-1]}], sigmasq: [..., {sigmasq.shape[-1]}].")
    if border.shape[-1] != mu.shape[-1] - 1:
        raise ValueError(
            f"The rightmost dimension of border and mu (sigmasq) don't match: "
            f"mu: [..., {mu.shape[-1]}], border: [..., {border.shape[-1]}].")

    if not np.isfinite(border).all():
        raise ValueError('All borders must be finite.')
    ext_border = concatenate_broadcast(
        [np.array([-np.inf]), border, np.array([np.inf])], )
    lowers = ext_border[..., :-1]
    uppers = ext_border[..., 1:]

    stds = np.sqrt(sigmasq)

    # uppers[..., :-1] == lowers[..., 1:] == border
    log_pdf_left = truncated_normal_pdf(
            x=uppers[..., :-1],
            lower=lowers[..., :-1],
            upper=uppers[..., :-1],
            mean=mu[..., :-1],
            std=stds[..., :-1],
            return_log=True)
    log_pdf_right = truncated_normal_pdf(
            x=lowers[..., 1:],
            lower=lowers[..., 1:],
            upper=uppers[..., 1:],
            mean=mu[..., 1:],
            std=stds[..., 1:],
            return_log=True)
    log_rate = log_pdf_left - log_pdf_right
    log_weight = concatenate_broadcast([np.array([0]), log_rate])
    log_weight = np.cumsum(log_weight, axis=-1)
    log_weight = log_weight - log_weight.max(-1, keepdims=1)
    weight = np.exp(log_weight)
    pi = weight / weight.sum(-1, keepdims=1)

    # TODO: how to use sample_shape in our algorithm?
    return sample_mixture_adjacent_truncated_normal(
        border, mu, sigmasq, pi, sample_shape)


# TODO: Improve speed.
# It accounts for 70% of runtime of sample_heterogeneous_normal_special
# logpdf, logcdf, logsf, log1mexp takes up most of the time
def truncated_normal_pdf(x, lower, upper, mean, std, return_log=True):
    if np.any(np.isinf(mean)):
        raise ValueError('Invalid mean: contains infty.')
    if np.any(np.isinf(std)):
        raise ValueError('Invalid std: contains infty.')

    if not (lower < upper).all():
        raise ValueError(
                'Lower border must be '
                'strictly less than upper border')

    x_standardized = (x - mean) / std
    upper_standardized = (upper - mean) / std
    lower_standardized = (lower - mean) / std

    log_pdf_standardized = stats.norm.logpdf(x_standardized)

    # Compute log probability using both CDF and surfival func
    # to avoid numeric underflow
    cdf_log_upper = stats.norm.logcdf(upper_standardized)
    cdf_log_lower = stats.norm.logcdf(lower_standardized)
    sf_log_upper = stats.norm.logsf(upper_standardized)
    sf_log_lower = stats.norm.logsf(lower_standardized)
    if not np.logical_or(
            cdf_log_lower < cdf_log_upper,
            sf_log_lower > sf_log_upper).all():
        raise ValueError('CDF and SF underflow')

    log_prob_cdf = cdf_log_upper + log1mexp(cdf_log_upper - cdf_log_lower)
    log_prob_sf = sf_log_lower + log1mexp(sf_log_lower - sf_log_upper)
    # Underflow causes log_prob_cdf or log_prob_sf to be zero
    log_prob = np.minimum(log_prob_cdf, log_prob_sf)

    log_pdf_truncated = log_pdf_standardized - log_prob - np.log(std)

    if return_log:
        out = log_pdf_truncated
    else:
        out = np.exp(log_pdf_truncated)
    return out


def concatenate_broadcast(arrays, axis=-1):
    """
    A generalized concatenation of arrays.

    Credits can be found at:
    https://stackoverflow.com/questions/56357047/concatenate-with-broadcast
    """

    def broadcast(x, shape):
        shape = [*shape]  # weak copy
        shape[axis] = x.shape[axis]
        return np.broadcast_to(x, shape)

    shapes = [list(a.shape) for a in arrays]
    for s in shapes:
        s[axis] = 1

    broadcast_shape = np.broadcast(
        *[np.broadcast_to(0, s) for s in shapes]).shape

    arrays = [broadcast(a, broadcast_shape) for a in arrays]
    return np.concatenate(arrays, axis=axis)


def sample_categorical(logits):
    withzeros = np.concatenate([
        logits, np.zeros_like(logits[..., -1:])], axis=-1)
    return tfd.Categorical(logits=withzeros).sample().numpy()[..., np.newaxis]


def sample_invgamma(
        concentration, scale, sample_shape,
        backend='numpy'):
    if backend == 'numpy':
        sample_shape = (
                sample_shape
                + np.broadcast(concentration, scale).shape)
        sample = 1 / np.random.gamma(concentration, 1/scale, sample_shape)
    elif backend == 'scipy':
        sample = stats.invgamma.rvs(
                concentration, scale=scale, size=sample_shape)
    elif backend == 'tensorflow':
        dtype = np.float64
        sample = (
                tfd.InverseGamma(dtype(concentration), dtype(scale))
                .sample(sample_shape).numpy())
    return sample


# # Standard normal cdf for z > 8.5 is underflown to 1.0.
# # Not using the maximal possible x_inf
# # because transformation may be need after sampling
# # from standard normal.
# normal_grid_z, normal_grid_u = get_normal_grid_z(
#         size=int(1e3), x_max=8,
#         x_inf=sys.float_info.max * 1e-3)
