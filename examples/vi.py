import sys
import os
import pickle
from time import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from dalea.plot import plot_curve
# from evaluate import evaluate_samples


def predict_states(model, x, n_states=100):
    predicted = []
    for __ in range(n_states):
        predicted.append(model(x).numpy())
    predicted = np.stack(predicted)
    return predicted


def train_model(
        model, loss,
        x, y,
        learning_rate, epochs, batch_size,
        verbose=0):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=loss,
        metrics=['mse']
    )
    loss_hist = model.fit(
            x, y,
            epochs=epochs, batch_size=batch_size,
            verbose=verbose)
    print('Training evaluation:', model.evaluate(x, y, verbose=0))
    loss_hist = loss_hist.history
    return loss_hist


# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def create_bnn_model(
        n_features, n_targets, hidden_layer_sizes,
        n_observations, fit_noise_scale,
        activation):

    inputs = layers.Input(shape=(n_features,))
    features = inputs

    # Create hidden layers with weight uncertainty
    # using the DenseVariational layer.
    for units in hidden_layer_sizes:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / n_observations,
            activation=activation
        )(features)

    if fit_noise_scale:
        # Create a probabilisticå output (Normal distribution),
        # and use the `Dense` layer
        # to produce the parameters of the distribution.
        # We set units=2*n_targets to learn both the mean and the variance
        # of the Normal distribution.
        units = tfp.layers.IndependentNormal.params_size(n_targets)
        distribution_params = layers.Dense(units=units)(features)
        outputs = tfp.layers.IndependentNormal(n_targets)(
                distribution_params)
    else:
        # The output is deterministic: a single point estimate.
        outputs = layers.Dense(units=n_targets)(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


def predict(model, x, state_shape=(), realization_shape=()):
    n_states = np.prod(state_shape)
    y_samples_flat_list = []
    for __ in range(n_states):
        y_samples_single_state = (
                model(x).sample(realization_shape).numpy())
        y_samples_flat_list.append(y_samples_single_state)
    y_samples_flat = np.stack(y_samples_flat_list)
    realization_ndim = len(realization_shape)
    # put realizations before states
    axes_order = np.concatenate([
            np.arange(realization_ndim) + 1,
            [0],
            np.arange(
                realization_ndim+1,
                y_samples_flat.ndim)
            ])
    y_samples_reordered = np.transpose(
            y_samples_flat, axes_order)
    y_samples = y_samples_flat.reshape(
            *realization_shape,
            *state_shape,
            *y_samples_reordered
            .shape[(realization_ndim+1):])
    return y_samples


def run_bnn_vi(
        x, y,
        hidden_layer_sizes, activation, fit_noise_scale,
        learning_rate, n_epochs, batch_size, verbose):

    n_observations, n_features = x.shape[-2:]
    n_targets = y.shape[-1]
    assert y.shape[-2] == n_observations

    model = create_bnn_model(
            n_features=n_features,
            n_targets=n_targets,
            hidden_layer_sizes=hidden_layer_sizes,
            n_observations=n_observations,
            fit_noise_scale=fit_noise_scale,
            activation=activation)

    if fit_noise_scale:
        loss = negative_loglikelihood
    else:
        loss = 'mse'

    loss_hist = train_model(
            x=x, y=y,
            model=model, loss=loss,
            learning_rate=learning_rate,
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=verbose)

    return model, loss_hist


def gen_simple_data(
        n_observations, n_features, n_targets, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.randn(n_observations, n_features)
    effects = np.random.rand(n_features, n_targets) + 2
    noise = np.random.randn(n_observations, n_targets) * 0.2
    y = x @ effects + noise
    return x, y


def main():

    seed = '0001' if len(sys.argv) <= 1 else sys.argv[1]
    studyname = 'a' if len(sys.argv) <= 2 else sys.argv[2]

    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))

    saveid = f'{studyname}_{seed}'
    print('saveid:', saveid)
    outdir = f'results/{studyname}/{seed}/vi'
    os.makedirs(outdir, exist_ok=True)
    print('outdir:', outdir)

    infile_data = f'pickles/{saveid}_data.pickle'
    print(f'Using data from {infile_data}')
    data = pickle.load(open(infile_data, 'rb'))
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    # y_mean_truth_train = data['y_mean_train']
    y_mean_truth_test = data['y_mean_test']
    # y_std_truth_train = data['y_std_train']
    # y_std_truth_test = data['y_std_test']

    # # n_features = 10
    # # n_targets = 1
    # # n_train = 1000
    # n_test = 2000
    # # x_train = np.random.randn(n_train, n_features)
    # # y_train = np.random.randn(n_train, n_targets)
    # x_train = np.loadtxt('x.txt', ndmin=2)
    # y_train = np.loadtxt('y.txt', ndmin=2)
    # n_features = x_train.shape[1]
    # n_targets = y_train.shape[1]
    # x_test = np.random.randn(n_test, n_features)
    # y_test = np.random.randn(n_test, n_targets)
    # y_mean_truth_test = None

    n_hidden_layers = 1
    n_hidden_nodes = 32
    activation = 'hard_sigmoid'
    fit_noise_scale = True

    learning_rate = 1e-3
    n_epochs = 500
    n_observations_train = x_train.shape[-2]
    batch_size = max(32, n_observations_train // 8)
    verbose = 2

    n_states = 600
    n_realizations = 20
    n_chains = 5
    falserate = 0.10
    n_strata = 5

    tstart = time()
    hidden_layer_sizes = [n_hidden_nodes] * n_hidden_layers
    model, loss_hist = run_bnn_vi(
            x=x_train, y=y_train,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            fit_noise_scale=fit_noise_scale,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=verbose)
    print('VI training time:', time() - tstart)
    # outfile_model = f'pickles/{saveid}_vi.keras'
    # model.save(outfile_model)
    # print('VI model saved to', outfile_model)
    state_shape = (n_states, n_chains)
    realization_shape = (n_realizations,)
    tstart = time()
    y_dist_test = predict(model, x_test, state_shape, realization_shape)
    print('VI sampling time:', time() - tstart)

    evaluate_samples(
            y_dist_test=y_dist_test,
            y_test=y_test,
            y_mean_truth_test=y_mean_truth_test,
            n_strata=n_strata, falserate=falserate,
            outfile=f'{outdir}/evaluation.txt')

    plt.figure(figsize=(12, 12))
    plt.plot(loss_hist['loss'])
    plt.title('training loss')
    plt.yscale('log')
    plt.savefig(f'{outdir}/losshist.png', dpi=300)
    plt.close()

    # plot y vs x curve
    n_targets = y_train.shape[-1]
    if n_targets == 1:
        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(8, 3))
        plot_curve(
                y_dist_test, x_test, y_test,
                y_mean_truth_test, falserate)
        # plt.title('Posterior estimation by VI')
        plt.tick_params(axis='x', direction='in', pad=-15)
        plt.text(
                0.95, 0.2, 'VI', fontsize=20,
                horizontalalignment='right',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.savefig(
            f'{outdir}/curve.png',
            dpi=200, bbox_inches='tight')
        plt.close()
        print(f'{outdir}/curve.png')


if __name__ == '__main__':
    main()
