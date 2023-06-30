from time import time
import sys
import pickle

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from scipy.stats import spearmanr
import matplotlib
from synthesize import get_data as get_data_synthetic
from einops import rearrange, repeat

from utils.visual import plot_density
from utils.utils import gaussian_loss

tf.enable_v2_behavior()
tfd = tfp.distributions
matplotlib.use('Agg')


# Generate some data
def f(x, w):
    # Pad x with 1's so we can add bias via matmul
    assert w.shape[-2] == x.shape[-2] + 1
    pad_width = [[0, 0]] * (len(x.shape)-2)
    pad_width += [[1, 0], [0, 0]]
    x = tf.pad(x, pad_width, constant_values=1)
    linop = tf.linalg.LinearOperatorFullMatrix(w)
    result = linop.matmul(x, adjoint=True)
    return result[..., :]


def forward(x, params, random_latent=False, realization_shape=()):
    w0, w1, w2, s0_log, s1_log, s2_log, e0, e1 = params

    u0 = np.tile(x, realization_shape + (1,) * (2+x.ndim))
    v0_mean = f(u0, w0)

    if random_latent:
        s0 = tf.exp(s0_log)
        e0 = tf.random.normal(v0_mean.shape, dtype=v0_mean.dtype) * s0
    v0 = v0_mean + e0
    u1 = tf.keras.activations.hard_sigmoid(v0)
    v1_mean = f(u1, w1)

    if random_latent:
        s1 = tf.exp(s1_log)
        e1 = tf.random.normal(v1_mean.shape, dtype=v1_mean.dtype) * s1
    v1 = v1_mean + e1
    u2 = tf.keras.activations.hard_sigmoid(v1)
    v2_mean = f(u2, w2)

    s2 = tf.exp(s2_log)

    return v0_mean, v1_mean, v2_mean, s2


# Define the joint_log_prob function, and our unnormalized posterior.
def joint_log_prob(params, x, y):

    w0, w1, w2, s0_log, s1_log, s2_log, e0, e1 = params
    v0_mean, v1_mean, v2_mean, __ = forward(x, params)
    v2 = y
    e2 = v2 - v2_mean

    # priors for weights
    n0 = w0.shape.num_elements()
    rv_w0 = tfd.MultivariateNormalDiag(
      loc=np.zeros(n0),
      scale_diag=np.ones(n0))

    n1 = w1.shape.num_elements()
    rv_w1 = tfd.MultivariateNormalDiag(
      loc=np.zeros(n1),
      scale_diag=np.ones(n1))

    n2 = w2.shape.num_elements()
    rv_w2 = tfd.MultivariateNormalDiag(
      loc=np.zeros(n2),
      scale_diag=np.ones(n2))

    s0 = tf.exp(s0_log)
    rv_e0 = tfd.Normal(0.0, s0)

    s1 = tf.exp(s1_log)
    rv_e1 = tfd.Normal(0.0, s1)

    s2 = tf.exp(s2_log)
    rv_e2 = tfd.Normal(0.0, s2)

    lp_w0 = rv_w0.log_prob(tf.reshape(w0, (n0,)))
    lp_w1 = rv_w1.log_prob(tf.reshape(w1, (n1,)))
    lp_w2 = rv_w2.log_prob(tf.reshape(w2, (n2,)))
    lp_weights = lp_w0 + lp_w1 + lp_w2

    lp_e0 = tf.reduce_sum(rv_e0.log_prob(e0))
    lp_e1 = tf.reduce_sum(rv_e1.log_prob(e1))
    lp_e2 = tf.reduce_sum(rv_e2.log_prob(e2))
    lp_latent = lp_e0 + lp_e1 + lp_e2

    logprob = lp_weights + lp_latent

    return logprob


def make_toy_data(num_features, num_examples, noise_scale, true_w):
    x = np.random.uniform(-1., 1., [num_features, num_examples])
    noise = np.random.normal(0., noise_scale, size=num_examples)
    z = f(x, true_w).numpy()
    y = np.sin(z) + noise
    return x, y


def get_data_toy(num_features, num_examples, noise_scale, true_w):
    x_train, y_train = make_toy_data(
            num_features, num_examples, noise_scale, true_w)
    x_test, y_test = make_toy_data(
            num_features, num_examples, noise_scale, true_w)
    return (x_train, y_train), (x_test, y_test)


def train_model(x, y):

    def unnormalized_posterior(*params):
        return joint_log_prob(
                params=params,
                x=x, y=y)

    num_features, num_examples = x.shape
    num_targets = y.shape[0]

    # Create an HMC TransitionKernel
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_posterior,
            step_size=[
                np.float64(0.01),
                np.float64(0.01),
                np.float64(0.01),
                np.float64(1e-3),
                np.float64(1e-3),
                np.float64(1e-3),
                np.float64(0.01),
                np.float64(0.01),
                ],
            num_leapfrog_steps=10)

    @tf.function
    def run_chain(
            initial_state, num_results=100, num_burnin_steps=0,
            num_steps_between_results=9):
        if num_burnin_steps == 0:
            num_adaptation_steps = int(
                    0.4 * num_results * (num_steps_between_results + 1))
        else:
            num_adaptation_steps = int(0.8 * num_burnin_steps)
        adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            hmc_kernel,
            num_adaptation_steps=num_adaptation_steps,
            target_accept_prob=np.float64(0.65))

        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_steps_between_results=num_steps_between_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=adaptive_kernel,
            trace_fn=lambda cs, kr: kr)

    t0 = time()
    num_chains = 5
    num_nodes = 25
    initial_state = [
            np.random.randn(num_chains, num_features+1, num_nodes) * 0.1,
            np.random.randn(num_chains, num_nodes+1, num_nodes) * 0.1,
            np.random.randn(num_chains, num_nodes+1, num_targets) * 0.1,
            np.full((num_chains, num_nodes, 1), np.log(0.1)),
            np.full((num_chains, num_nodes, 1), np.log(0.1)),
            np.full((num_chains, num_targets, 1), np.log(0.1)),
            np.random.randn(num_chains, num_nodes, num_examples) * 0.1,
            np.random.randn(num_chains, num_nodes, num_examples) * 0.1,
            ]
    samples, kernel_results = run_chain(initial_state=initial_state)
    print('Runtime:', int(time() - t0))
    print(
            'Acceptance rate:',
            kernel_results.inner_results.is_accepted.numpy().mean())

    return samples


def plot_model(
        params, x_observed, y_observed, x_quantiles, prefix,
        n_realizations):
    y_mean, y_std = forward(
            x_quantiles, params,
            random_latent=True, realization_shape=(n_realizations,))[-2:]
    x_quantiles = x_quantiles.swapaxes(-1, -2)
    y_mean = rearrange(y_mean.numpy(), 'r s c t n -> (r s c) n t')
    y_std = repeat(y_std.numpy(),  's c t n -> (r s c) n t', r=n_realizations)
    x_observed = x_observed.swapaxes(-1, -2)
    y_observed = y_observed.swapaxes(-1, -2)
    plot_density(
            mean=y_mean, std=y_std,
            x_quantiles=x_quantiles,
            x_observed=x_observed, y_observed=y_observed,
            n_samples=50, prefix=prefix)


def save_nll(params, x, y, n_realizations, prefix):
    mean, std = forward(
            x, params, random_latent=True,
            realization_shape=(n_realizations,))[-2:]
    mean = mean.numpy().swapaxes(-1, -2)
    std = repeat(std.numpy(),  's c t n -> r s c n t', r=n_realizations)
    x = x.swapaxes(-1, -2)
    y = y.swapaxes(-1, -2)
    nlp = gaussian_loss(mean, std, y, reduction=True)
    outfile = prefix + 'nll.pickle'
    pickle.dump(nlp, open(outfile, 'wb'))
    print(outfile)


def eval_model(params, x, y, n_realizations):
    y_mean = forward(
            x, params, random_latent=True,
            realization_shape=(n_realizations,))[-2]
    y_mean = y_mean.numpy()
    corr = spearmanr(y_mean.mean((0, 1, 2))[0], y[0])[0]
    print(f'Correlation: {corr}')


def process_data(data):
    (x_train, y_train), (x_test, y_test) = data
    x_train = x_train.T.astype(np.float64)
    y_train = y_train.T.astype(np.float64)
    x_test = x_test.T.astype(np.float64)
    y_test = y_test.T.astype(np.float64)
    data = (x_train, y_train), (x_test, y_test)
    return data


def main():

    dataset = sys.argv[1] if len(sys.argv) > 1 else None
    prefix = 'dalea-hmc-'
    n_realizations = 10

    if dataset is None:
        data = get_data_toy(
                num_features=2,
                num_examples=80,
                noise_scale=0.5,
                true_w=np.array([-1., 2., 3.])[..., np.newaxis])
    else:
        data = get_data_synthetic(
                dataset=dataset,
                split=00)
        x_quantiles, __ = data[0]
        data = process_data(data[1:])
        x_quantiles = x_quantiles.swapaxes(-1, -2).astype(np.float64)

    (x_train, y_train), (x_test, y_test) = data

    params = train_model(x_train, y_train)

    eval_model(params, x_train, y_train, n_realizations=n_realizations)
    eval_model(params, x_test, y_test, n_realizations=n_realizations)
    plot_model(
            params=params, x_observed=x_train, y_observed=y_train,
            x_quantiles=x_quantiles, n_realizations=n_realizations,
            prefix=prefix)
    save_nll(
            params=params, x=x_test, y=y_test,
            n_realizations=n_realizations, prefix=prefix)


if __name__ == '__main__':
    main()
