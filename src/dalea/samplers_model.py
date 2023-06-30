from .samplers_basic import (
        sample_invgamma,
        sample_bayesian_linear_regression,
        sample_heteroscedastic_linear_regression,
        sample_heterogeneous_normal,
        normal_log_den)
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions


hard_sigmoid_slope = 0.2


def sample_features(
        targets,
        kernel,
        bias,
        noise_scale,
        features_mean,
        features_scale):
    """ Sample posterior features in Bayesian linear regression
        with fixed targets, kernel, bias, and noise scale

    Args:
        targets: Has shape `(..., n_observations, n_targets)`
        kernel: Has shape `(..., n_features, n_targets)`.
        bias: Has shape `(..., 1, n_targets)`.
        noise_scale: Has shape `(..., 1, n_targets)`
        features_mean: Has shape `(..., n_observations, n_features)`.
        features_scale: Has shape `(..., 1, n_features)`

    Returns:
        features: A random sample from the posterior distribution
            of the features.
            Has shape `(..., n_observations, n_features)`.
    """
    n_observations, n_targets = targets.shape[-2:]
    n_features = kernel.shape[-2]
    assert kernel.shape[-1] == n_targets
    assert bias.shape[-2:] == (1, n_targets)
    assert noise_scale.shape[-2:] == (1, n_targets)
    assert features_mean.shape[-2:] == (n_observations, n_features)
    assert features_scale.shape[-2:] == (1, n_features)
    features_sample = sample_heteroscedastic_linear_regression(
            y=(targets - bias),
            x=np.expand_dims(kernel.swapaxes(-1, -2), -3),
            sigmasq=1/noise_scale**2,
            mu=features_mean,
            kappasq=1/features_scale**2)
    return features_sample


def sample_kernel(*args, **kwargs):
    return sample_kernel_heteroscedastic(*args, **kwargs)


def sample_kernel_heteroscedastic(
        features,
        targets,
        noise_scale,
        kernel_mean,
        kernel_scale):
    """ Sample posterior weights (no bias) from Bayesian linear regression.
        The ouput is identical to that of `sample_kernel_conjugate`,
        but this function calls `sample_heteroscedastic_linear_regression`
        instead of `sample_bayesian_linear_regression`.

    Args:
        features: Has shape `(..., n_observations, n_features)`.
        targets: Has shape `(..., n_observations, n_targets)`
        noise_scale: Has shape `(..., n_observations, n_targets)`
        kernel_mean: Has shape `(..., n_features, n_targets)`.
        kernel_scale: Has shape `(..., n_features, n_targets)`

    Returns:
        kernel: A random sample from the posterior distribution
            of the weights (no bias).
            Has shape `(..., n_features, n_targets)`.
    """

    n_observations, n_features = features.shape[-2:]
    n_targets = targets.shape[-1]
    assert targets.shape[-2] == n_observations
    assert noise_scale.shape[-1] == n_targets
    assert kernel_mean.shape[-2:] == (n_features, n_targets)
    assert kernel_scale.shape[-2:] == (n_features, n_targets)

    kernel_sample = sample_heteroscedastic_linear_regression(
            y=targets.swapaxes(-1, -2),
            x=np.expand_dims(features, -3),
            sigmasq=(1/noise_scale**2).swapaxes(-1, -2),
            mu=kernel_mean.swapaxes(-1, -2),
            kappasq=(1/kernel_scale**2).swapaxes(-1, -2))
    kernel_sample = kernel_sample.swapaxes(-1, -2)
    return kernel_sample


def sample_kernel_conjugate(
        features,
        targets,
        noise_scale,
        kernel_mean,
        kernel_scale):
    """ Sample posterior weights (no bias) from Bayesian linear regression

    Args:
        features: Has shape `(..., n_observations, n_features)`.
        targets: Has shape `(..., n_observations, n_targets)`
        noise_scale: Has shape `(..., 1, n_targets)`
        kernel_mean: Has shape `(..., n_features, n_targets)`.
        kernel_scale: Has shape `(..., n_features, n_targets)`

    Returns:
        kernel: A random sample from the posterior distribution
            of the weights (no bias).
            Has shape `(..., n_features, n_targets)`.
    """

    n_observations, n_features = features.shape[-2:]
    n_targets = targets.shape[-1]
    assert targets.shape[-2] == n_observations
    assert noise_scale.shape[-2:] == (1, n_targets)
    assert kernel_mean.shape[-2:] == (n_features, n_targets)
    assert kernel_scale.shape[-2:] == (n_features, n_targets)

    noise_variance = noise_scale**2
    precision = noise_variance / kernel_scale**2
    precision = precision.swapaxes(-1, -2)
    kernel_precision_mat = np.zeros(precision.shape + (n_features,))
    idx = np.diag_indices(n_features)
    kernel_precision_mat[..., idx[0], idx[1]] = precision
    noise_variance = noise_variance[..., 0, :]
    output = sample_bayesian_linear_regression(
            y=targets,
            x=features,
            mu_prior=kernel_mean,
            lambda_mat_prior=kernel_precision_mat,
            sigmasq=noise_variance)
    return output


def sample_kernel_bias_zero_mean(
        features,
        targets,
        noise_scale,
        kernel_scale,
        bias_scale):
    """ Sample posterior kernel and bias from Bayesian linear regression
        with prior mean equal to zero.

    Args:
        features: Has shape `(..., n_observations, n_features)`.
        targets: Has shape `(..., n_observations, n_targets)`
        noise_scale: Has shape `(..., 1, n_targets)`
        kernel_scale: Has shape `(..., n_features, n_targets)`
        bias_scale: Has shape `(..., 1, n_targets)`

    Returns:
        kernel: A random sample from the posterior distribution
            of the kernel (in joint with bias).
            Has shape `(..., n_features, n_targets)`.
        bias: A random sample from the posterior distribution
            of the bias (in joint with kernel).
            Has shape `(..., 1, n_targets)`.
    """

    n_observations, n_features = features.shape[-2:]
    n_targets = targets.shape[-1]
    assert targets.shape[-2] == n_observations
    assert noise_scale.shape[-2:] == (1, n_targets)
    assert kernel_scale.shape[-2:] == (n_features, n_targets)
    assert bias_scale.shape[-2:] == (1, n_targets)

    ones = np.ones_like(features[..., -1:])
    features_with_ones = np.concatenate([features, ones], -1)
    n_features = features.shape[-1]
    kernel_bias_scale = np.concatenate([kernel_scale, bias_scale], -2)
    n_targets = targets.shape[-1]
    kernel_bias_mean = np.zeros((n_features+1, n_targets))

    kernel_bias = sample_kernel(
            features_with_ones,
            targets,
            noise_scale,
            kernel_bias_mean,
            kernel_bias_scale)
    kernel = kernel_bias[..., :-1, :]
    bias = kernel_bias[..., -1:, :]
    return kernel, bias


def sample_scale_known_mean(
        observations,
        mean,
        axis,
        prior_concentration,
        prior_scale,
        sample_shape=()):
    """ Sample posterior scale (i.e. standard deviation).
        Observations follow a normal distribution
        with known mean.
        Scale follows an inverse gamma distribution.
        Observations are aggregated eacross `axis`
        after broadcasting `mean` and `observations`.

    Args:
        observations: Observed measurements.
        mean: Mean of the normal distribution.
        axis: A tuple. Axis across which to aggregate the observations.
        prior_concentration: Concentration of prior InvGamma distribution.
        prior_scale: Scale of prior InvGamma distribution..
        shape: Shape of sample.

    Returns:
        scale_sample: A sample from the posterior distribution.
    """

    x = observations - mean
    sum_of_squares = (x**2).sum(axis, keepdims=True)
    n_observations = x.size // sum_of_squares.size
    posterior_concentration = prior_concentration + n_observations / 2
    posterior_scale = prior_scale + sum_of_squares / 2
    variance_sample = sample_invgamma(
            concentration=posterior_concentration,
            scale=posterior_scale,
            sample_shape=sample_shape)
    scale_sample = np.sqrt(variance_sample)

    # variance_sample = tfd.InverseGamma(
    #         concentration=posterior_concentration,
    #         scale=posterior_scale).sample()
    # precision_sample = tfd.Gamma(
    #         concentration=posterior_concentration,
    #         rate=posterior_scale).sample()

    # precision_sample = np.random.gamma(
    #         shape=posterior_concentration,
    #         scale=1/posterior_scale)
    # variance_sample = 1 / precision_sample
    # scale_sample = np.sqrt(variance_sample)

    return scale_sample


def param_mean_known_scale(
        prior_mean,
        prior_scale,
        observation,
        observation_scale):
    """ get posterior mean and scale

    Args:
        prior_mean: Has shape `(...)`.
        prior_scale: Has shape `(...)`.
        observation: Has shape `(...)`.
        observation_scale: Has shape `(...)`.

    Returns:
        posterior_mean: Has shape `(...)`.
        posterior_scale: Has shape `(...)`.
    """

    prior_precision = prior_scale**(-2)
    observation_precision = observation_scale**(-2)
    posterior_precision = (
            prior_precision + observation_precision)
    posterior_mean = ((
        prior_precision * prior_mean
        + observation_precision * observation)
        / posterior_precision)
    posterior_scale = posterior_precision**(-0.5)
    return posterior_mean, posterior_scale


def sample_pre_activation(
        pre_act_mean, pre_act_scale,
        post_act_observation, post_act_scale,
        border, slope, intercept):
    """ Sample pre-activation values
        from heterogeneous normal distributions.
        Component `j` has range `(border[j-1], border[j])`
        (`border[-1] = -Inf`
        and `border[n_components] = Inf`
        by definition).
        Inside this range, the activation function is
        `f(u) = slope[j] * u + intercept[j]`

    Args:
        pre_act_mean: Has shape
            `(..., n_features)`.
        pre_act_scale: Has shape `(..., n_features)`.
        post_act_observation: Has shape
            `(..., n_features)`.
        post_act_scale: Has shape `(..., n_features)`.
        border: Has shape (n_components-1,).
        slope: Has shape (n_components,).
        intercept: Has shape (n_components,).

    Returns:
        Has shape `(..., n_features)`.
    """

    assert np.ndim(border) == 1
    n_components = len(border) + 1
    assert (np.diff(border) > 0).all()
    assert np.isfinite(border).all()
    assert np.shape(slope) == (n_components,)
    assert np.shape(intercept) == (n_components,)
    n_features = pre_act_mean.shape[-1]
    assert post_act_observation.shape[-1] == n_features
    assert pre_act_scale.shape[-1] == n_features
    assert post_act_scale.shape[-1] == n_features
    assert np.isfinite(pre_act_scale).all()
    assert np.isfinite(post_act_scale).all()
    assert (pre_act_scale > 0).all()
    assert (post_act_scale > 0).all()

    pre_act_mean = np.expand_dims(pre_act_mean, -1)
    pre_act_scale = np.expand_dims(pre_act_scale, -1)
    post_act_observation = np.expand_dims(post_act_observation, -1)
    post_act_scale = np.expand_dims(post_act_scale, -1)

    scale = (
            pre_act_scale**(-2)
            + slope**2 * post_act_scale**(-2))**(-0.5)
    # Note that `slope` is not squared in `mean`,
    # that is, `weight * (post_act_observation - intercept) / slope`
    mean = (
            pre_act_scale**(-2) / scale**(-2)
            * pre_act_mean
            + slope * post_act_scale**(-2) / scale**(-2)
            * (post_act_observation - intercept))
    log_denrat = (
            normal_log_den(
                post_act_observation,
                slope[1:] * border + intercept[1:],
                post_act_scale)
            - normal_log_den(
                post_act_observation,
                slope[:-1] * border + intercept[:-1],
                post_act_scale))

    pre_act = sample_heterogeneous_normal(
            border=np.concatenate([[-np.inf], border, [np.inf]], axis=-1),
            mean=mean,
            std=scale,
            log_denrat=log_denrat)
    return pre_act


def sample_pre_activation_special(
        pre_act_mean,
        pre_act_scale,
        post_act_observations,
        post_act_scale,
        border):
    """ Sample pre-activation values
        from heterogeneous normal distributions

    Args:
        pre_act_mean: Has shape
            `(..., n_observations, n_features)`.
        pre_act_scale: Has shape `(...)`.
        post_act_observations: Has shape
            `(..., n_observations, n_features)`.
        post_act_scale: Has shape `(...)`.
        border: Has shape (2,).
            Use np.inf for 2-component
            heterogeneous normal distributions (e.g. ReLU)

    Returns:
        Has shape `(..., n_observations, n_features)`.
    """

    assert border.shape == (2,)
    assert border[0] < border[1]
    assert np.isfinite(border).any()

    mean_left = mean_right = pre_act_mean
    scale_left = scale_right = np.expand_dims(np.expand_dims(
        pre_act_scale, -1), -1)

    mean_middle, scale_middle = param_mean_known_scale(
            pre_act_mean,
            np.expand_dims(np.expand_dims(
                pre_act_scale, -1), -1),
            post_act_observations,
            np.expand_dims(np.expand_dims(
                post_act_scale, -1), -1))

    mean_list = [mean_middle]
    scale_list = [scale_middle]
    if np.isfinite(border[0]):
        mean_list = [mean_left] + mean_list
        scale_list = [scale_left] + scale_list
    if np.isfinite(border[1]):
        mean_list = mean_list + [mean_right]
        scale_list = scale_list + [scale_right]
    mean_components = np.stack(mean_list, -1)
    scale_components = np.stack(scale_list, -1)
    border = border[np.isfinite(border)]

    variance_components = scale_components**2
    pre_act = sample_heterogeneous_normal(
            border=border,
            mu=mean_components,
            sigmasq=variance_components)
    return pre_act


def sample_pre_categorization(
        pre_cat_mean,
        pre_cat_scale,
        pre_cat_observations_old,
        post_cat_observations):
    """ Sample pre-categorization values
        from 3-component heterogeneous normal distributions

    Args:
        pre_cat_mean: Has shape
            `(..., n_observations, n_features)`.
        pre_cat_scale: Has shape `(...)`.
        pre_cat_observations_old: Has shape
            `(..., n_observations, n_features)`.
        post_cat_observations: Indices of categories.
            Has shape `(..., n_observations)`.

    Returns:
        Has shape `(..., n_observations, n_features)`.
    """
    n_observations, n_features = pre_cat_mean.shape[-2:]
    pre_cat_observations_old.shape[-2:] == n_observations, n_features
    assert post_cat_observations.shape[-1] == n_observations
    assert np.issubdtype(post_cat_observations.dtype, np.integer)
    n_features = pre_cat_mean.shape[-1]
    assert post_cat_observations.max() <= n_features
    scale_left = scale_right = np.expand_dims(pre_cat_scale, -1)
    pre_cat_observations = np.zeros_like(pre_cat_observations_old)
    for i in range(n_features):
        if n_features == 1:
            a = np.zeros(pre_cat_observations_old.shape[:-1])
        else:
            # TODO: use better slicing and handle overflow
            a = np.log(
                    np.exp(pre_cat_observations_old).sum(-1) + 1
                    - np.exp(pre_cat_observations_old[..., i]))
        # TODO: softwire hard_sigmoid_slope
        border = np.stack([
            a - 0.5 / hard_sigmoid_slope,
            a + 0.5 / hard_sigmoid_slope],
            axis=-1)
        log_denrat = np.zeros_like(border)
        border = np.concatenate([
            np.full_like(border[..., [0]], -np.inf),
            border,
            np.full_like(border[..., [0]], np.inf)], -1)
        s = (i == post_cat_observations) * (-2) + 1
        mean_left = (
                pre_cat_mean[..., i]
                - (s-1)/2 * np.expand_dims(pre_cat_scale, -1)**2)
        mean_right = (
                pre_cat_mean[..., i]
                - (s+1)/2 * np.expand_dims(pre_cat_scale, -1)**2)
        mean_middle, scale_middle = param_mean_known_scale(
                prior_mean=pre_cat_mean[..., i],
                prior_scale=np.expand_dims(pre_cat_scale, -1),
                observation=a - s * 0.5 / hard_sigmoid_slope,
                observation_scale=np.array(hard_sigmoid_slope)**(-0.5))
        mean_components = np.stack(
                [mean_left, mean_middle, mean_right], -1)
        scale_components = np.stack(
                [scale_left, scale_middle, scale_right], -1)
        pre_cat_observations[..., i] = sample_heterogeneous_normal(
                border=border,
                mean=mean_components,
                std=scale_components,
                log_denrat=log_denrat)
    return pre_cat_observations
