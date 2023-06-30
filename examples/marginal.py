from time import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf

from dalea.onelayer import DALEAOneHiddenLayer
from dalea.onelayer import bounded_relu
from dalea.samplers_onelayer import (
        sample_kernel, sample_kernel_bias_zero_mean)

tfd = tfp.distributions


def transpose(x):
    return np.swapaxes(x, -1, -2)


def trace(x):
    return np.trace(x, axis1=-1, axis2=-2)


def lp_marginal_wrapped(
        beta, gamma, v, v_prev, sigma, tau, rho, xi):
    '''
        f(beta, gamma | vs, variances) + C(vs, variances)
    '''
    n_observations = v.shape[-2]
    n_features = beta.shape[-2]
    u = np.random.randn(n_observations, n_features)
    joint = lp_joint_wrapped(
            u, beta, gamma, v, v_prev, sigma, tau, rho, xi)
    conditional = lp_conditional_wrapped(
            u, beta, gamma, v, v_prev, sigma, tau)
    marginal = joint - conditional
    return marginal


def lp_joint_wrapped(
        u, beta, gamma, v, v_prev, sigma, tau, rho, xi):
    '''
        f(u, beta, gamma | vs, variances) + C(vs, variances)
    '''
    n_observations, n_features = u.shape[-2:]
    n_targets = v.shape[-1]
    assert beta.shape[-2:] == (n_features, n_targets)
    assert beta.shape[-1] == n_targets
    n_hidden_layers = 1
    n_hidden_nodes = n_features

    model = DALEAOneHiddenLayer(
            n_features=1,
            n_targets=n_targets,
            n_hidden_nodes=n_hidden_nodes,
            n_hidden_layers=n_hidden_layers,
            target_type='continuous')

    layer = 1
    model.set_param('u', layer, u)
    model.set_param('beta', layer, beta)
    model.set_param('gamma', layer, gamma)
    if layer == n_hidden_layers:
        model.set_data(
                x=np.random.randn(
                    n_observations, model.n_features),
                y=v)
    else:
        model.set_param('v', layer, v)
    model.set_param('v', layer-1, v_prev)
    model.set_param('sigma', layer, sigma)
    model.set_param('tau', layer, tau)
    model.set_param('rho', layer, rho)
    model.set_param('xi', layer, xi)
    return model.log_prob_beta_u_v(layer)


def test_joint_wrapped_pure(
        n_features=2, n_targets=4, n_observations=3,
        n_reps=5):

    v = np.random.randn(n_observations, n_targets)
    v_prev = np.random.randn(n_observations, n_features)
    sigma = np.random.rand()
    tau = np.random.rand()
    rho = np.random.rand()
    xi = np.random.rand()

    u = np.random.randn(n_reps, n_observations, n_features)
    beta = np.random.randn(n_features, n_targets)
    gamma = np.random.randn(1, n_targets)

    lp_wrapped = lp_joint_wrapped(
            u, beta, gamma, v, v_prev, sigma, tau, rho, xi)
    lp_pure = lp_joint_pure(
            u, beta, gamma, v, v_prev, sigma, tau, rho, xi)
    lp_diff_wrapped = lp_wrapped[1:] - lp_wrapped[0]
    lp_diff_pure = lp_pure[1:] - lp_pure[0]
    assert np.allclose(lp_diff_wrapped, lp_diff_pure)
    del u, beta, gamma, lp_wrapped, lp_pure, lp_diff_wrapped, lp_diff_pure

    u = np.random.randn(n_observations, n_features)
    beta = np.random.randn(n_reps, n_features, n_targets)
    gamma = np.random.randn(n_reps, 1, n_targets)

    lp_wrapped = lp_joint_wrapped(
            u, beta, gamma, v, v_prev, sigma, tau, rho, xi)
    lp_pure = lp_joint_pure(
            u, beta, gamma, v, v_prev, sigma, tau, rho, xi)
    lp_diff_wrapped = lp_wrapped[1:] - lp_wrapped[0]
    lp_diff_pure = lp_pure[1:] - lp_pure[0]
    assert np.allclose(lp_diff_wrapped, lp_diff_pure, 1e-4, 1e-4)


def test_conditional_wrapped_pure(
        n_features=2, n_targets=4, n_observations=3,
        n_reps=5):

    v = np.random.randn(n_observations, n_targets)
    v_prev = np.random.randn(n_observations, n_features)
    sigma = np.random.rand()
    tau = np.random.rand()

    u = np.random.randn(n_reps, n_observations, n_features)
    beta = np.random.randn(n_features, n_targets)
    gamma = np.random.randn(1, n_targets)

    lp_wrapped = lp_conditional_wrapped(
            u, beta, gamma, v, v_prev, sigma, tau)
    lp_pure = lp_conditional_pure(
            u, beta, gamma, v, v_prev, sigma, tau)
    lp_diff_wrapped = lp_wrapped[1:] - lp_wrapped[0]
    lp_diff_pure = lp_pure[1:] - lp_pure[0]
    assert np.allclose(lp_diff_wrapped, lp_diff_pure)


def test_joint_minus_conditional_wrapped(
        n_features=5, n_targets=50, n_observations=3,
        n_reps=100):

    v = np.random.randn(n_observations, n_targets)
    v_prev = np.random.randn(n_observations, n_features)
    sigma = np.random.rand()
    tau = np.random.rand()
    rho = np.random.rand()
    xi = np.random.rand()

    u = np.random.randn(n_reps, n_observations, n_features)
    beta = np.random.randn(n_features, n_targets)
    gamma = np.random.randn(1, n_targets)

    joint = lp_joint_wrapped(
            u, beta, gamma, v, v_prev, sigma, tau, rho, xi)
    conditional = lp_conditional_wrapped(
            u, beta, gamma, v, v_prev, sigma, tau)
    marginal = lp_marginal_wrapped(
            beta, gamma, v, v_prev, sigma, tau, rho, xi)

    joint_diff = joint[1:] - joint[0]
    conditional_diff = conditional[1:] - conditional[0]
    assert np.allclose(joint_diff, conditional_diff)
    joint_conditional_diff = joint - conditional
    assert np.allclose(marginal, joint_conditional_diff)


def test_joint_minus_conditional_pure(
        n_features=5, n_targets=50, n_observations=3,
        n_reps=100):

    v = np.random.randn(n_observations, n_targets)
    v_prev = np.random.randn(n_observations, n_features)
    sigma = np.random.rand()
    tau = np.random.rand()
    rho = np.random.rand()
    xi = np.random.rand()

    u = np.random.randn(n_reps, n_observations, n_features)
    beta = np.random.randn(n_features, n_targets)
    gamma = np.random.randn(1, n_targets)

    joint = lp_joint_pure(
            u, beta, gamma, v, v_prev, sigma, tau, rho, xi)
    conditional = lp_conditional_pure(
            u, beta, gamma, v, v_prev, sigma, tau)
    marginal = lp_marginal_pure(
            beta, gamma, v, v_prev, sigma, tau, rho, xi)

    joint_diff = joint[1:] - joint[0]
    conditional_diff = conditional[1:] - conditional[0]
    assert np.allclose(joint_diff, conditional_diff)
    joint_conditional_diff = joint - conditional
    assert np.allclose(marginal, joint_conditional_diff)


def test_marginal_wrapped_pure(
        n_features=2, n_targets=4, n_observations=3,
        n_reps=5):

    v = np.random.randn(n_observations, n_targets)
    v_prev = np.random.randn(n_observations, n_features)
    sigma = np.random.rand()
    tau = np.random.rand()
    rho = np.random.rand() * 1e-2
    xi = np.random.rand()

    beta = np.random.randn(n_reps, n_features, n_targets)
    gamma = np.random.randn(n_reps, 1, n_targets)

    lp_wrapped = lp_marginal_wrapped(
            beta, gamma, v, v_prev, sigma, tau, rho, xi)
    lp_pure = lp_marginal_pure(
            beta, gamma, v, v_prev, sigma, tau, rho, xi)
    lp_pure_simplified = lp_marginal_pure_simplified(
            beta, gamma, v, v_prev, sigma, tau, rho, xi)
    lp_pure_arithmetic = lp_marginal_pure_arithmetic(
            beta, gamma, v, v_prev, sigma, tau, rho, xi)
    lp_pure_arithmetic_semiquadratic = (
            lp_marginal_pure_arithmetic_semiquadratic(
                beta, gamma, v, v_prev, sigma, tau, rho, xi))
    lp_diff_wrapped = lp_wrapped[1:] - lp_wrapped[0]
    lp_diff_pure = lp_pure[1:] - lp_pure[0]
    lp_diff_pure_simplified = lp_pure_simplified[1:] - lp_pure_simplified[0]
    lp_diff_pure_arithmetic = lp_pure_arithmetic[1:] - lp_pure_arithmetic[0]
    lp_diff_pure_arithmetic_semiquadratic = (
            lp_pure_arithmetic_semiquadratic[1:]
            - lp_pure_arithmetic_semiquadratic[0])
    assert np.allclose(lp_diff_wrapped, lp_diff_pure, 1e-4, 1e-4)
    assert np.allclose(lp_diff_pure, lp_diff_pure_simplified)
    assert np.allclose(lp_diff_pure, lp_diff_pure_arithmetic)
    assert np.allclose(lp_diff_pure, lp_diff_pure_arithmetic_semiquadratic)


def hmc(v, v_prev, sigma, tau, rho, xi, n_chains, n_states):

    def unnormalized_log_prob(beta, gamma):
        return (-0.5) * lp_marginal_pure_arithmetic(
                beta, gamma, v, v_prev, sigma, tau, rho, xi)

    n_observations, n_features = v_prev.shape[-2:]
    n_targets = v.shape[-1]
    beta_init = get_beta_term(
        beta=None,
        rho=rho*np.ones((n_features, n_targets)),
        sample_shape=(n_chains,))
    gamma_init = get_gamma_term(
        gamma=None,
        v=v, tau=tau, xi=xi,
        sample_shape=(n_chains,))
    current_state = [beta_init, gamma_init]
    step_size = [rho, (xi**(-2) + tau**(-2) * n_observations)**(-0.5)]

    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_log_prob,
                num_leapfrog_steps=20,
                step_size=step_size),
            num_adaptation_steps=int(n_states * 0.5))
    samples, is_accepted = tfp.mcmc.sample_chain(
            num_results=n_states,
            num_burnin_steps=0,
            current_state=current_state,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
    states = {
            'beta': samples[0].numpy(),
            'gamma': samples[1].numpy(),
            'is_accepted': is_accepted.numpy()}
    return states


def sample_u(beta, gamma, v, v_prev, sigma, tau):
    features = transpose(beta)
    targets = transpose(v - gamma)
    kernel_mean = transpose(bounded_relu(v_prev))
    n_features = features.shape[-1]
    kernel_scale = np.tile(np.expand_dims(sigma, -1), n_features)
    noise_scale = tau
    u = sample_kernel(
            features=features,
            targets=targets,
            noise_scale=noise_scale,
            kernel_mean=kernel_mean,
            kernel_scale=kernel_scale)
    return u


def sample_beta_gamma(u, v, tau, rho, xi):
    beta, gamma = sample_kernel_bias_zero_mean(
            features=u,
            targets=v,
            noise_scale=tau,
            kernel_scale=rho,
            bias_scale=xi)
    return beta, gamma


def gibbs_sampling(
        v, v_prev, sigma, tau, rho, xi, n_chains, n_states):
    n_observations, n_features = v_prev.shape[-2:]
    n_targets = v.shape[-1]
    beta = get_beta_term(
        beta=None,
        rho=rho*np.ones((n_features, n_targets)),
        sample_shape=(n_chains,))
    gamma = get_gamma_term(
        gamma=None,
        v=v, tau=tau, xi=xi,
        sample_shape=(n_chains,))

    states = {'beta': [], 'gamma': [], 'is_accepted': []}
    for __ in range(n_states):
        ut = sample_u(beta, gamma, v, v_prev, sigma, tau)
        u = transpose(ut)
        beta, gamma = sample_beta_gamma(u, v, tau, rho, xi)
        states['beta'].append(beta)
        states['gamma'].append(gamma)
    states['beta'] = np.stack(states['beta'])
    states['gamma'] = np.stack(states['gamma'])
    return states


def rejection_sampling(
        v, v_prev, sigma, tau, rho, xi, n_chains, n_states,
        method):

    n_features = v_prev.shape[-1]
    n_targets = v.shape[-1]

    if method == 'independent_static':

        def draw_beta(beta_current=0):
            return get_beta_term(
                    beta=None,
                    rho=rho*np.ones((n_features, n_targets)),
                    sample_shape=(n_chains,))

        def draw_gamma(gamma_current=0):
            return get_gamma_term(
                    gamma=None,
                    v=v, tau=tau, xi=xi,
                    sample_shape=(n_chains,))

        def lp_diff(beta, gamma):
            return (-0.5) * get_nonquadratic_term(
                    beta, gamma, v, v_prev, sigma, tau, rho, xi)

    elif method == 'independent_dynamic':

        def draw_beta(beta_current=0):
            return beta_current + get_beta_term(
                    beta=None,
                    rho=rho*np.ones((n_features, n_targets)),
                    sample_shape=(n_chains,))

        def draw_gamma(gamma_current=0):
            return gamma_current + get_gamma_term(
                    gamma=None,
                    v=v, tau=tau, xi=xi,
                    sample_shape=(n_chains,))

        def lp_diff(beta, gamma):
            return (-0.5) * lp_marginal_pure_arithmetic(
                    beta, gamma, v, v_prev, sigma, tau, rho, xi)

    elif method == 'dependent_dynamic':

        def draw_beta(beta_current=None):
            if beta_current is None:
                beta_current = np.random.randn(n_chains, n_features, n_targets)
            precision_mat = (
                    tau**(-2) * beta_current @ transpose(beta_current)
                    + sigma**(-2) * np.eye(n_features))
            beta_proposed = get_semiquadratic_term(
                    None, precision_mat, v, v_prev, sigma, tau, rho)
            return beta_proposed

        def draw_gamma(gamma_current=0):
            return gamma_current + get_gamma_term(
                    gamma=None,
                    v=v, tau=tau, xi=xi,
                    sample_shape=(n_chains,))

        def lp_diff(beta, gamma):
            return (-0.5) * get_seminonquadratic_term(
                    beta, gamma, v, v_prev, sigma, tau)

    def lp_cross_diff(beta, gamma, beta_current, gamma_current):
        return lp_diff(beta, gamma) - lp_diff(beta_current, gamma_current)

    states = {'beta': [], 'gamma': [], 'is_accepted': []}
    beta_current = draw_beta()
    gamma_current = draw_gamma()
    for __ in range(n_states):
        beta_proposed = draw_beta(beta_current)
        gamma_proposed = draw_gamma(gamma_current)
        logprob_accept = lp_cross_diff(
                beta_proposed, gamma_proposed, beta_current, gamma_current)
        prob_accept = np.exp(np.clip(logprob_accept, None, 0))
        accept = np.random.rand(n_chains) < prob_accept
        beta_chosen = (
                beta_proposed * accept[..., np.newaxis, np.newaxis]
                + beta_current * (~accept[..., np.newaxis, np.newaxis]))
        gamma_chosen = (
                gamma_proposed * accept[..., np.newaxis, np.newaxis]
                + gamma_current * (~accept[..., np.newaxis, np.newaxis]))
        beta_current, gamma_current = beta_chosen, gamma_chosen
        states['beta'].append(beta_chosen)
        states['gamma'].append(gamma_chosen)
        states['is_accepted'].append(accept)
    states['beta'] = np.stack(states['beta'])
    states['gamma'] = np.stack(states['gamma'])
    states['is_accepted'] = np.stack(states['is_accepted'])
    return states


def plot_trace(states, title=None):
    plt.subplot(2, 1, 1)
    plt.plot(states['beta'][..., 0, 0])
    plt.ylabel('beta')
    plt.title(title)
    plt.subplot(2, 1, 2)
    plt.plot(states['gamma'][..., 0, 0])
    plt.ylabel('gamma')
    plt.title(title)
    plt.show()


def plot_density(density, beta_range, gamma_range):
    be, ga = np.meshgrid(
            np.linspace(
                *beta_range, 100),
            np.linspace(
                *gamma_range, 100))
    logprob = density(
            be[..., np.newaxis, np.newaxis],
            ga[..., np.newaxis, np.newaxis])
    logprob = logprob - logprob.flatten()[logprob.size//2]
    plt.contourf(be, ga, logprob, 10)
    plt.colorbar()


def test_metrapolis_hastings(
        n_features=1, n_targets=1, n_observations=100,
        n_chains=7, n_states=500):

    sigma = np.random.rand() * 1e-1
    tau = np.random.rand()
    rho = np.random.rand() * 1e-2
    xi = np.random.rand() * 1e-1
    beta_truth = np.random.randn(n_features, n_targets)
    gamma_truth = np.random.randn(1, n_targets)
    v_prev = np.random.randn(n_observations, n_features)
    noise_u = np.random.randn(n_observations, n_features) * sigma
    noise_v = np.random.randn(n_observations, n_targets) * tau
    v = (bounded_relu(v_prev) + noise_u) @ beta_truth + gamma_truth + noise_v

    t0 = time()
    states_rsc = rejection_sampling(
            v, v_prev, sigma, tau, rho, xi, n_chains, n_states,
            method='independent_static')
    print('timing rsc: ', int(time() - t0))
    print('acceptance rsc:', states_rsc['is_accepted'].mean())
    t0 = time()
    states_rsd = rejection_sampling(
            v, v_prev, sigma, tau, rho, xi, n_chains, n_states,
            'independent_dynamic')
    print('timing rsd: ', int(time() - t0))
    print('acceptance rsd:', states_rsd['is_accepted'].mean())
    t0 = time()
    states_rsp = rejection_sampling(
            v, v_prev, sigma, tau, rho, xi, n_chains, n_states,
            'dependent_dynamic')
    print('timing rsp: ', int(time() - t0))
    print('acceptance rsp:', states_rsp['is_accepted'].mean())
    t0 = time()
    states_gibbs = gibbs_sampling(
            v, v_prev, sigma, tau, rho, xi, n_chains, n_states)
    print('timing gibbs: ', int(time() - t0))
    t0 = time()
    states_hmc = hmc(
            v, v_prev, sigma, tau, rho, xi, n_chains, n_states)
    print('timing hmc: ', int(time() - t0))
    print('acceptance hmc:', states_hmc['is_accepted'].mean())
    plot_trace(states_rsc, 'independent static')
    plot_trace(states_rsd, 'independent dynamic')
    plot_trace(states_rsp, 'dependent dynamic')
    plot_trace(states_gibbs, 'gibbs')
    plot_trace(states_hmc, 'hmc')
    if n_features == 1 and n_targets == 1:
        def unnormalized_log_prob(beta, gamma):
            return (-0.5) * lp_marginal_pure_arithmetic(
                    beta, gamma, v, v_prev, sigma, tau, rho, xi)
        plot_density(
                unnormalized_log_prob,
                np.quantile(
                    np.stack([
                        states_rsc['beta'],
                        states_rsd['beta'],
                        states_rsp['beta'],
                        states_gibbs['beta'],
                        states_hmc['beta'],
                        ]),
                    [0, 1]),
                np.quantile(
                    np.stack([
                        states_rsc['gamma'],
                        states_rsd['gamma'],
                        states_rsp['gamma'],
                        states_gibbs['gamma'],
                        states_hmc['gamma'],
                        ]),
                    [0, 1]))
        plt.plot(
                states_hmc['beta'][..., 0, 0].flatten(),
                states_hmc['gamma'][..., 0, 0].flatten(),
                's', alpha=0.5,
                label='hmc')
        plt.plot(
                states_rsp['beta'][..., 0, 0].flatten(),
                states_rsp['gamma'][..., 0, 0].flatten(),
                '^', alpha=0.5,
                label='dependent dynamic')
        plt.plot(
                states_rsd['beta'][..., 0, 0].flatten(),
                states_rsd['gamma'][..., 0, 0].flatten(),
                'v', alpha=0.5,
                label='independent dynamic')
        plt.plot(
                states_rsc['beta'][..., 0, 0].flatten(),
                states_rsc['gamma'][..., 0, 0].flatten(),
                '<', alpha=0.5,
                label='independent static')
        plt.plot(
                states_gibbs['beta'][..., 0, 0].flatten(),
                states_gibbs['gamma'][..., 0, 0].flatten(),
                'o', alpha=0.5,
                label='gibbs')
        plt.plot(
                beta_truth[0, 0], gamma_truth[0, 0],
                '*', markersize=20,
                label='truth')
        plt.xlabel('beta')
        plt.ylabel('gamma')
        plt.title('marginal density')
        plt.legend()
        plt.show()


def lp_marginal_pure_simplified(
        beta, gamma, v, v_prev, sigma, tau, rho, xi):

    n_features = beta.shape[-2]
    n_observations = v.shape[-2]
    w = bounded_relu(v_prev)
    u = np.random.randn(n_observations, n_features)

    u_padded = np.expand_dims(u, -2)
    # ubeta = u @ beta
    ubeta = (u_padded * transpose(np.expand_dims(beta, -3))).sum(-1)
    betabeta = (
            np.expand_dims(beta, -2)
            * np.expand_dims(beta, -3)).sum(-1)
    betabeta_padded = np.expand_dims(betabeta, -3)

    beta_term = (rho**(-2) * beta**2).sum((-1, -2))
    gamma_term = (xi**(-2) * gamma**2).sum((-1, -2))
    u_term = sigma**(-2) * (u**2 - 2 * u * w + w**2).sum((-1, -2))
    v_term = tau**(-2) * (
            v**2 + ubeta**2 + gamma**2
            - 2 * v * ubeta
            + 2 * ubeta * gamma
            - 2 * gamma * v).sum((-1, -2))
    lp_joint = (-0.5) * (
        beta_term + gamma_term + u_term + v_term)

    # quadratic_term = (
    #         u_padded
    #         @ precision_mat_padded
    #         @ transpose(u_padded)).sum((-1, -2, -3))
    # quadratic_term = (
    #         transpose(u_padded) * u_padded
    #         * precision_mat_padded).sum((-1, -2, -3))
    quadratic_term_0 = (
            tau**(-2) * betabeta_padded
            * transpose(u_padded) * u_padded
            ).sum((-1, -2, -3))
    quadratic_term_1 = (
            sigma**(-2) * transpose(u)**2).sum((-1, -2))
    quadratic_term = quadratic_term_0 + quadratic_term_1
    # linear_term = (-2) * (
    #         normalized_mean * transpose(u)).sum((-1, -2))
    linear_term = (-2) * ((
            tau**(-2) * (
                np.expand_dims(beta, -2)
                * np.expand_dims(v-gamma, -3)).sum(-1)
            + sigma**(-2) * transpose(w)) * transpose(u)).sum((-1, -2))
    precision_mat = (
            tau**(-2) * beta @ transpose(beta)
            + sigma**(-2) * np.eye(n_features))
    normalized_mean = (
            tau**(-2) * beta @ transpose(v - gamma)
            + sigma**(-2) * transpose(w))
    constant_term = (
            np.expand_dims(transpose(normalized_mean), -2)
            @ np.expand_dims(np.linalg.inv(precision_mat), -3)
            @ np.expand_dims(transpose(normalized_mean), -1)).sum((-1, -2, -3))
    scale_term = n_observations * (
            n_features * np.log(2*np.pi)
            - np.log(np.linalg.det(precision_mat)))
    lp_conditional = (-0.5) * (
            quadratic_term
            + linear_term
            + constant_term
            + scale_term)

    lp = lp_joint - lp_conditional
    return lp


def get_beta_term(beta, rho, sample_shape=()):
    # dist = tfd.Normal(loc=0.0, scale=rho)
    if beta is None:
        # out = dist.sample(sample_shape).numpy()
        out = np.random.randn(*sample_shape, *np.shape(rho)) * rho
    else:
        if 'tensor' in type(beta).__name__ or 'Tensor' in type(beta).__name__:
            # out = (-2) * dist.log_prob(beta).numpy().sum((-1, -2))
            out = rho**(-2) * tf.reduce_sum(beta**2, (-1, -2))
        else:
            out = rho**(-2) * (beta**2).sum((-1, -2))
    return out


def get_gamma_term(gamma, v, tau, xi, sample_shape=()):
    # gamma_term = (
    #     (xi**(-2) + tau**(-2) * n_observations) * (gamma**2)
    #     + ((-2) * tau**(-2) * v.sum(-2, keepdims=True) * gamma)
    #     ).sum((-1, -2))
    n_observations = v.shape[-2]
    scale = (xi**(-2) + tau**(-2) * n_observations)**(-0.5)
    loc = (tau**(-2) * v.sum(-2, keepdims=True)) * scale**2
    # dist = tfd.Normal(loc=loc, scale=scale)
    if gamma is None:
        # out = dist.sample(sample_shape).numpy()
        batch_shape = np.broadcast(loc, scale).shape
        out = np.random.randn(*sample_shape, *batch_shape) * scale + loc
    else:
        if (
                'tensor' in type(gamma).__name__
                or 'Tensor' in type(gamma).__name__):
            # out = (-2) * dist.log_prob(gamma).numpy().sum((-1, -2))
            out = scale**(-2) * tf.reduce_sum((gamma - loc)**2, (-1, -2))
        else:
            out = scale**(-2) * ((gamma - loc)**2).sum((-1, -2))
    return out


def get_nonquadratic_term(beta, gamma, v, v_prev, sigma, tau, rho, xi):

    n_observations = v.shape[-2]
    w = bounded_relu(v_prev)
    n_features = beta.shape[-2]

    precision_mat = (
            tau**(-2) * beta @ transpose(beta)
            + sigma**(-2) * np.eye(n_features))
    cov_mat = np.linalg.inv(precision_mat)
    normalized_mean = (
            tau**(-2) * beta @ transpose(v - gamma)
            + sigma**(-2) * transpose(w))
    inverse_term = (-1) * (
            np.expand_dims(transpose(normalized_mean), -2)
            @ np.expand_dims(cov_mat, -3)
            @ np.expand_dims(transpose(normalized_mean), -1)
            ).sum((-1, -2, -3))
    precision_mat_logdet = np.linalg.slogdet(precision_mat)[1]
    determinant_term = (-1) * n_observations * (
            n_features * np.log(2*np.pi)
            - precision_mat_logdet)
    return inverse_term + determinant_term


def lp_marginal_pure_arithmetic(
        beta, gamma, v, v_prev, sigma, tau, rho, xi):

    beta_term = get_beta_term(beta, rho)
    gamma_term = get_gamma_term(gamma, v, tau, xi)
    nonquadratic_term = get_nonquadratic_term(
            beta, gamma, v, v_prev, sigma, tau, rho, xi)
    logprob = (-0.5) * (
            beta_term + gamma_term + nonquadratic_term)
    return logprob


def single_roughen(x, m, n):
    assert x.ndim >= 1
    assert x.shape[-1] == m * n
    return x.reshape(*x.shape[:-1], m, n)


def single_flatten(x):
    assert x.ndim >= 2
    return x.reshape(*x.shape[:-2], np.prod(x.shape[-2:]))


def dual_flatten(x):
    assert x.ndim >= 4
    m, n = x.shape[-3], x.shape[-1]
    assert (x.shape[-4], x.shape[-2]) == (m, n)
    x_flat = x.swapaxes(-2, -3).reshape(*x.shape[:-4], m * n, m * n)
    return x_flat


def dual_roughen(x, m, n):
    assert x.ndim >= 2
    assert x.shape[-2] == x.shape[-1] == m * n
    x_rough = x.reshape(*x.shape[:-2], m, n, m, n).swapaxes(-2, -3)
    return x_rough


def get_seminonquadratic_term(
        beta, gamma, v, v_prev, sigma, tau):

    n_features = beta.shape[-2]
    n_observations, n_targets = v.shape[-2:]
    w = bounded_relu(v_prev)
    precision_mat = (
            tau**(-2) * beta @ transpose(beta)
            + sigma**(-2) * np.eye(n_features))
    cov_mat = np.linalg.inv(precision_mat)

    inverse_term_00_01 = (-1) * tau**(-4) * (
            beta[..., np.newaxis, np.newaxis, :, np.newaxis, :]
            * beta[..., np.newaxis, :, np.newaxis, :, np.newaxis]
            * gamma[..., np.newaxis, np.newaxis, np.newaxis, :]
            * v[..., np.newaxis, np.newaxis, :, np.newaxis]
            * cov_mat[..., np.newaxis, :, :, np.newaxis, np.newaxis]
            ).sum((-1, -2, -3, -4, -5))
    inverse_term_00_11 = (-1) * tau**(-4) * (
            beta[..., np.newaxis, np.newaxis, :, np.newaxis, :]
            * beta[..., np.newaxis, :, np.newaxis, :, np.newaxis]
            * gamma[..., np.newaxis, np.newaxis, :, np.newaxis]
            * gamma[..., np.newaxis, np.newaxis, np.newaxis, :]
            * cov_mat[..., np.newaxis, :, :, np.newaxis, np.newaxis]
            ).sum((-1, -2, -3, -4, -5)) * n_observations
    inverse_term_10_1 = (-1) * tau**(-2) * sigma**(-2) * (
            np.expand_dims(beta, -3)
            * np.expand_dims(gamma, -2) * (
                np.expand_dims(cov_mat, -3)
                @ np.expand_dims(w, -1))
            ).sum((-1, -2, -3))
    inverse_term_11 = (-1) * sigma**(-4) * (
            np.expand_dims(w, -2)
            @ np.expand_dims(cov_mat, -3)
            @ np.expand_dims(w, -1)).sum((-1, -2, -3))
    precision_mat_logdet = np.linalg.slogdet(precision_mat)[1]
    determinant_term = (-1) * n_observations * (
            n_features * np.log(2*np.pi)
            - precision_mat_logdet)
    seminonquadratic_term = (
            - 2 * inverse_term_00_01
            + inverse_term_00_11
            - 2 * inverse_term_10_1
            + inverse_term_11
            + determinant_term)
    return seminonquadratic_term


def get_semiquadratic_term(
        beta, precision_mat, v, v_prev, sigma, tau, rho,
        sample_shape=()):

    n_features = v_prev.shape[-1]
    n_observations, n_targets = v.shape[-2:]
    w = bounded_relu(v_prev)

    # beta_term = rho**(-2) * (beta**2).sum((-1, -2))
    beta_term_precision_mat = (
            np.eye(n_features * n_targets) * rho**(-2))
    # beta_term = (
    #         single_flatten(beta)[..., np.newaxis, :]
    #         @ beta_term_precision_mat
    #         @ single_flatten(beta)[..., np.newaxis])[..., 0, 0]

    cov_mat = np.linalg.inv(precision_mat)
    semiquadratic_precision_mat = dual_flatten(
            cov_mat[..., np.newaxis, :, :, np.newaxis, np.newaxis]
            * v[..., np.newaxis, np.newaxis, :, np.newaxis]
            * v[..., np.newaxis, np.newaxis, np.newaxis, :])
    posterior_precision_mat = (
            beta_term_precision_mat / n_observations
            - tau**(-4) * semiquadratic_precision_mat)
    posterior_covariance_mat = np.linalg.inv(posterior_precision_mat)
    posterior_mean = tau**(-2) * sigma**(-2) * (
            posterior_covariance_mat
            @ semiquadratic_precision_mat
            @ single_flatten(
                w[..., :, np.newaxis]
                / v[..., np.newaxis, :]
                / n_targets)[..., np.newaxis])[..., 0]
    # semiquadratic_normalized_mean = (
    #         semiquadratic_precision_mat
    #         * semiquadratic_mean[..., np.newaxis, :, np.newaxis, :]
    #         ).sum((-1, -3))
    # semiquadratic_normalized_mean = (
    #         semiquadratic_precision_mat
    #         @ semiquadratic_mean[..., np.newaxis])[..., 0]
    posterior_normalized_mean = (
            posterior_precision_mat
            @ posterior_mean[..., np.newaxis])[..., 0]
    posterior_allobs_precision_mat = posterior_precision_mat.sum(-3)
    posterior_allobs_covariance_mat = np.linalg.inv(
            posterior_allobs_precision_mat)
    posterior_allobs_mean = (
            posterior_allobs_covariance_mat
            @ posterior_normalized_mean.sum(-2)[..., np.newaxis])[..., 0]
    # semiquadratic_normalized_mean = (
    #         posterior_precision_mat @ posterior_covariance_mat
    #         * semiquadratic_precision_mat
    #         * semiquadratic_mean[..., np.newaxis, :, np.newaxis, :]
    #         ).sum((-1, -3))

    # inverse_term_00_00 = (-1) * tau**(-4) * (
    #         beta[..., np.newaxis, np.newaxis, :, np.newaxis, :]
    #         * beta[..., np.newaxis, :, np.newaxis, :, np.newaxis]
    #         * semiquadratic_precision_mat
    #         ).sum((-1, -2, -3, -4, -5))
    # inverse_term_00_00 = (-1) * tau**(-4) * (
    #         single_flatten(beta)[..., np.newaxis, np.newaxis, :]
    #         @ semiquadratic_precision_mat
    #         @ single_flatten(beta)[..., np.newaxis, :, np.newaxis]
    #         )[..., 0, 0].sum(-1)
    # assert np.allclose(
    #         beta_inverse_00_00_term,
    #         beta_term + inverse_term_00_00)

    # inverse_term_10_0 = tau**(-2) * sigma**(-2) * (
    #         single_flatten(beta)[..., np.newaxis, :]
    #         * semiquadratic_normalized_mean
    #         ).sum((-1, -2))
    # NOTE: distribution is not normal (or even proper)
    # if precision matrix is not positive definite
    # Whether precision is PD depends on data
    # Thus this approach does not work
    precision_mat_is_positive_definite = (
            np.linalg.eigvalsh(posterior_precision_mat) > 0).all()
    if precision_mat_is_positive_definite:
        # dist = tfd.MultivariateNormalTriL(
        #         loc=posterior_mean,
        #         scale_tril=np.linalg.cholesky(posterior_covariance_mat))
        dist = tfd.MultivariateNormalTriL(
                loc=posterior_allobs_mean,
                scale_tril=np.linalg.cholesky(posterior_allobs_covariance_mat))
        if beta is None:
            out = single_roughen(
                    dist.sample(sample_shape).numpy(),
                    n_features, n_targets)
        else:
            out = (-2) * dist.log_prob(
                    single_flatten(beta)
                    ).numpy()
    else:
        if beta is None:
            raise ValueError(
                    'posterior precision matrix'
                    'not positive definite')
        else:

            beta_inverse_term_00_00 = (
                    single_flatten(beta)[..., np.newaxis, np.newaxis, :]
                    @ posterior_precision_mat
                    @ single_flatten(beta)[..., np.newaxis, :, np.newaxis]
                    )[..., 0, 0].sum(-1)
            inverse_term_10_0 = (
                    posterior_normalized_mean[..., np.newaxis, :]
                    @ single_flatten(beta)[..., np.newaxis, :, np.newaxis]
                    )[..., 0, 0].sum(-1)
            out = (
                    beta_inverse_term_00_00
                    - 2 * inverse_term_10_0)
    return out


def lp_marginal_pure_arithmetic_semiquadratic(
        beta, gamma, v, v_prev, sigma, tau, rho, xi,
        quadratic_only_proposal=True):
    n_observations, n_targets = v.shape[-2:]

    n_features = beta.shape[-2]

    gamma_term = get_gamma_term(gamma, v, tau, xi)
    precision_mat = (
            tau**(-2) * beta @ transpose(beta)
            + sigma**(-2) * np.eye(n_features))
    semiquadratic_term = get_semiquadratic_term(
            beta, precision_mat, v, v_prev, sigma, tau, rho)
    seminonquadratic_term = get_seminonquadratic_term(
            beta, gamma, v, v_prev, sigma, tau)

    lp = (-0.5) * (
            gamma_term
            + semiquadratic_term
            + seminonquadratic_term)

    # quadratic_term = beta_term + gamma_term
    # inverse_term_10 = inverse_term_10_0 + inverse_term_10_1
    # inverse_term_00 = (
    #         inverse_term_00_00 - 2 * inverse_term_00_01 + inverse_term_00_11)
    # inverse_term = inverse_term_00 - 2 * inverse_term_10 + inverse_term_11
    # lp = (-0.5) * (
    #         beta_term + gamma_term
    #         + inverse_term + determinant_term)
    # inverse_term_00 = (-1) * tau**(-4) * (
    #         np.expand_dims(transpose(beta_v_gamma), -2)
    #         @ np.expand_dims(cov_mat, -3)
    #         @ np.expand_dims(transpose(beta_v_gamma), -1)).sum((-1, -2, -3))
    # inverse_term_00_00 = (-1) * tau**(-4) * (
    #         np.expand_dims(v @ transpose(beta), -2)
    #         @ np.expand_dims(cov_mat, -3)
    #         @ np.expand_dims(v @ transpose(beta), -1)
    #         ).sum((-1, -2, -3))
    # inverse_term_00_01 = (-1) * tau**(-4) * (
    #         np.expand_dims(v @ transpose(beta), -2)
    #         @ np.expand_dims(cov_mat, -3)
    #         @ np.expand_dims(gamma @ transpose(beta), -1)
    #         ).sum((-1, -2, -3))
    # inverse_term_00_11 = (-1) * tau**(-4) * (
    #         np.expand_dims(gamma @ transpose(beta), -2)
    #         @ np.expand_dims(cov_mat, -3)
    #         @ np.expand_dims(gamma @ transpose(beta), -1)
    #         ).sum((-1, -2, -3)) * n_observations
    # inverse_term_10 = (-2) * tau**(-2) * sigma**(-2) * (
    #         np.expand_dims(transpose(beta_v_gamma), -2)
    #         @ np.expand_dims(cov_mat, -3)
    #         @ np.expand_dims(w, -1)).sum((-1, -2, -3))
    # inverse_term_10_0 = tau**(-2) * sigma**(-2) * (
    #         np.expand_dims(v @ transpose(beta), -2)
    #         @ np.expand_dims(cov_mat, -3)
    #         @ np.expand_dims(w, -1)).sum((-1, -2, -3))
    # inverse_term_10_1 = (-1) * tau**(-2) * sigma**(-2) * (
    #         np.expand_dims(gamma @ transpose(beta), -2)
    #         @ np.expand_dims(cov_mat, -3)
    #         @ np.expand_dims(w, -1)).sum((-1, -2, -3))

    return lp


def lp_marginal_pure(
        beta, gamma, v, v_prev, sigma, tau, rho, xi):

    n_features = beta.shape[-2]
    n_observations = v.shape[-2]
    w = bounded_relu(v_prev)
    u = np.random.randn(n_observations, n_features)

    beta_term = rho**(-2) * beta**2
    gamma_term = xi**(-2) * gamma**2
    u_term = sigma**(-2) * (u - w)**2
    v_term = tau**(-2) * (v - u @ beta - gamma)**2
    lp_joint = (-0.5) * (
        beta_term.sum((-1, -2))
        + gamma_term.sum((-1, -2))
        + u_term.sum((-1, -2))
        + v_term.sum((-1, -2)))

    precision_mat = (
            tau**(-2) * beta @ transpose(beta)
            + sigma**(-2) * np.eye(n_features))
    normalized_mean = (
            tau**(-2) * beta @ transpose(v - gamma)
            + sigma**(-2) * transpose(w))
    quadratic_term = trace(
            u @ precision_mat @ transpose(u))
    linear_term = (-2) * trace(
            transpose(normalized_mean) @ transpose(u))
    constant_term = trace(
            transpose(normalized_mean)
            @ np.linalg.inv(precision_mat)
            @ normalized_mean)
    scale_term = n_observations * (
            n_features * np.log(2*np.pi)
            - np.log(np.linalg.det(precision_mat)))
    lp_conditional = (-0.5) * (
            quadratic_term
            + linear_term
            + constant_term
            + scale_term)

    lp = lp_joint - lp_conditional
    return lp


def lp_joint_pure(
        u, beta, gamma, v, v_prev, sigma, tau, rho, xi):

    beta_term = rho**(-2) * beta**2
    gamma_term = xi**(-2) * gamma**2
    u_term = sigma**(-2) * (u - bounded_relu(v_prev))**2
    v_term = tau**(-2) * (v - u @ beta - gamma)**2
    lp = (-0.5) * (
        beta_term.sum((-1, -2))
        + gamma_term.sum((-1, -2))
        + u_term.sum((-1, -2))
        + v_term.sum((-1, -2)))
    return lp


def lp_conditional_pure(
        u, beta, gamma, v, v_prev, sigma, tau):
    n_observations, n_features = u.shape[-2:]
    assert v.shape[-2] == n_observations
    n_targets = v.shape[-1]
    assert beta.shape[-2:] == (n_features, n_targets)
    assert gamma.shape[-2:] == (1, n_targets)
    assert v_prev.shape[-2:] == (n_observations, n_features)

    precision_mat = (
            tau**(-2) * beta @ transpose(beta)
            + sigma**(-2) * np.eye(n_features))
    normalized_mean = (
            tau**(-2) * beta @ transpose(v - gamma)
            + sigma**(-2) * transpose(bounded_relu(v_prev)))
    quadratic_term = trace(
            u @ precision_mat @ transpose(u))
    linear_term = (-2) * trace(
            transpose(normalized_mean) @ transpose(u))
    constant_term = trace(
            transpose(normalized_mean)
            @ np.linalg.inv(precision_mat)
            @ normalized_mean)
    scale_term = n_observations * (
            n_features * np.log(2*np.pi)
            - np.log(np.linalg.det(precision_mat)))
    lp = (-0.5) * (
            quadratic_term
            + linear_term
            + constant_term
            + scale_term)
    return lp


def lp_conditional_wrapped(u, beta, gamma, v, v_prev, sigma, tau):
    '''
        f(u | beta, gamma, vs, variances)
    '''
    n_observations, n_features = u.shape[-2:]
    n_targets = v.shape[-1]
    assert beta.shape[-2:] == (n_features, n_targets)
    assert beta.shape[-1] == n_targets
    n_hidden_layers = 1
    n_hidden_nodes = n_features

    model = DALEAOneHiddenLayer(
            n_features=1,
            n_targets=n_targets,
            n_hidden_nodes=n_hidden_nodes,
            n_hidden_layers=n_hidden_layers,
            target_type='continuous')

    layer = 1
    model.set_param('u', layer, u)
    model.set_param('beta', layer, beta)
    model.set_param('gamma', layer, gamma)
    if layer == n_hidden_layers:
        model.set_data(
                x=np.random.randn(
                    n_observations, model.n_features),
                y=v)
    else:
        model.set_param('v', layer, v)
    model.set_param('v', layer-1, v_prev)
    model.set_param('sigma', layer, sigma)
    model.set_param('tau', layer, tau)
    return model.update_u(layer, skip_update=True)


seed = None
if seed is None:
    seed = np.random.choice(1000)
print(f'seed: {seed}')
np.random.seed(seed)
tf.random.set_seed(seed)

test_joint_minus_conditional_wrapped()
test_conditional_wrapped_pure()
test_joint_wrapped_pure()
test_joint_minus_conditional_pure()
test_marginal_wrapped_pure()
print('All tests passed')
test_metrapolis_hastings(n_features=1, n_targets=1)
test_metrapolis_hastings(n_features=8, n_targets=16)
