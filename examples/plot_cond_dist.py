#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from dalea.onelayer import DALEAOneHiddenLayer


def cond_dist_dalea(x):

    n_hidden_nodes = 2
    n_features = 1
    n_targets = 1
    n_hidden_layers = 1
    target_type = 'continuous'
    activation = 'bounded_relu'

    model = DALEAOneHiddenLayer(
            n_features,
            n_targets,
            n_hidden_nodes=n_hidden_nodes,
            n_hidden_layers=n_hidden_layers,
            target_type=target_type,
            activation=activation,
            save_all_params=False,
            save_priors_history=True)
    model.reset_params(
            n_observations=1, chain_shape=(),
            reset_beta_gamma='random',
            reset_u_v='random')
    model.set_param('beta', 0, np.array([[10, 10]]))
    model.set_param('gamma', 0, np.array([[0.5, 0.5]]))
    model.set_param('tau', 0, 10.0)
    model.set_param('sigma', 1, 0.1)
    model.set_param('beta', 1, np.array([[1], [1]]))
    model.set_param('gamma', 1, np.array([[-1]]))
    model.set_param('tau', 1, 0.01)
    model.save_params()

    n_realizations = 10000
    y_dist = model.predict(
            x[:, np.newaxis],
            realization_shape=(n_realizations,))[:, 0, :, 0]
    return y_dist


def cond_dist_bnn_hetero(y_dist):
    y_mean = y_dist.mean(0, keepdims=True)
    y_std = y_dist.std(0, keepdims=True)
    noise = np.random.randn(*y_dist.shape) * y_std
    y_dist_new = y_mean + noise
    return y_dist_new


def cond_dist_bnn_homo(y_dist):
    y_mean = y_dist.mean(0, keepdims=True)
    y_std = np.square(y_dist - y_mean).mean()**0.5
    noise = np.random.randn(*y_dist.shape) * y_std
    y_dist_new = y_mean + noise
    return y_dist_new


def plot_cond_dist(
        y_dist, alpha=0.05,
        ylim=(-2, 2), cmap='viridis', vmin=None, vmax=None,
        title=None):
    y_mean = y_dist.mean(0)
    y_ci = np.quantile(y_dist, [alpha/2, 1-alpha/2], axis=0)
    n_observations = x.shape[0]
    n_realizations = y_dist.shape[0]
    x_tile = np.tile(x, (n_realizations, 1)).flatten()
    cmap = plt.get_cmap(cmap)
    plt.hist2d(
            x_tile.flatten(), y_dist.flatten(),
            bins=(n_observations, 100), cmap=cmap,
            vmin=vmin, vmax=vmax, density=True)
    linewidth = 2
    plt.plot(x, y_mean, linewidth=linewidth, color='tab:red')
    plt.plot(
            x, y_ci[0], color='tab:orange',
            linewidth=linewidth, linestyle='--')
    plt.plot(
            x, y_ci[1], color='tab:orange',
            linewidth=linewidth, linestyle='--')
    plt.grid(axis='x')
    plt.ylim(*ylim)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.gca().xaxis.set_label_coords(1.03, 0.03)
    plt.gca().set_facecolor(cmap(0))
    cbar = plt.colorbar()
    cbar.set_label('density')
    plt.tick_params(
            axis='x', direction='in', pad=-15, colors='white')
    if title is not None:
        plt.text(
                0.05, 0.90, title,
                horizontalalignment='left',
                verticalalignment='top',
                transform=plt.gca().transAxes,
                backgroundcolor='white', color='black')


n_observations = 100
x = np.linspace(-3.5, 3.5, n_observations)
y_dist_dalea = cond_dist_dalea(x)
y_dist_bnn_hetero = cond_dist_bnn_hetero(y_dist_dalea)
y_dist_bnn_homo = cond_dist_bnn_homo(y_dist_dalea)
vmin = 0.0
vmax = 0.4

plt.figure(figsize=(8, 3))
plot_cond_dist(
        y_dist_bnn_homo, vmin=vmin, vmax=vmax,
        title='Homoscedastic Gaussian')
plt.savefig(
        'tablesfigures/cond_dist_homo.png',
        dpi=200, bbox_inches='tight')

plt.figure(figsize=(8, 3))
plot_cond_dist(
        y_dist_bnn_hetero, vmin=vmin, vmax=vmax,
        title='Heteroscedastic Gaussian')
plt.savefig(
        'tablesfigures/cond_dist_hetero.png',
        dpi=200, bbox_inches='tight')

plt.figure(figsize=(8, 3))
plot_cond_dist(y_dist_dalea, vmin=vmin, vmax=vmax, title='DALEA')
plt.savefig(
        'tablesfigures/cond_dist_dalea.png',
        dpi=200, bbox_inches='tight')
