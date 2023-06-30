import matplotlib.pyplot as plt
import numpy as np
from utils.utils import gaussian_loss


def plot_qq(expected, observed, outfile):
    quantiles = np.linspace(0, 1, 100+2)[1:-1]
    x = np.quantile(expected, quantiles)
    y = np.quantile(observed, quantiles)
    plt.figure(figsize=(8, 8))
    plt.plot(x, x, linewidth=2, color='tab:blue')
    plt.plot(x, y, 'o', alpha=0.5, color='tab:orange')
    plt.xlabel('true values')
    plt.ylabel('predicted values')
    plt.title('true vs predicted values at quantiles 0.01, ..., 0.99')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(outfile)


def plot_quantiles(
        x_quantiles, y_quantiles,
        x_observed, y_observed,
        quantiles, outfile):
    x_tiled = np.tile(
            x_quantiles[np.newaxis, ...],
            [y_quantiles.shape[0], 1])
    plt.figure(figsize=(16, 8))
    y_quantiles = np.quantile(y_quantiles, quantiles, axis=0)
    for xq, yq in zip(x_tiled, y_quantiles):
        plt.plot(xq, yq, color='tab:orange')
    if x_observed is not None and y_observed is not None:
        plt.plot(
                x_observed, y_observed, 'o',
                color='tab:blue', alpha=0.5)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(outfile)


def plot_history(history, outfile):
    plt.figure(figsize=(16, 8))
    plt.plot(history['train'], label='training')
    n_last = len(history['train']) // 10
    train_last = np.mean(history['train'][-n_last:])
    title = f'training: {train_last:7.3f}'
    if 'valid' in history.keys():
        valid_last = np.mean(history['valid'][-n_last:])
        plt.plot(history['valid'], label='validation')
        title += f', validation: {valid_last:7.3f}'
    if 'smooth' in history.keys():
        plt.plot(history['smooth'], label='smooth')
    if 'epoch_best' in history.keys():
        epoch_best = history['epoch_best']
        plt.axvline(
                x=epoch_best, linestyle='--', color='tab:gray')
        loss_best = history['smooth'][epoch_best]
        title = f'best epoch: {epoch_best}'
        title = f'best smooth loss: {loss_best:8.3f}'
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()

    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close()
    print(outfile)


def plot_density(
        mean, std, x_quantiles, x_observed, y_observed,
        n_samples, prefix):
    '''
    Visualize predictive density.
    Shapes:
        mean: (n_realizations, n_observations, n_targets)
        std: (n_realizations, n_observations, n_targets)
        x_quantiles: (n_quantiles, n_features)
        x_observed: (n_observations, n_features)
        y_observed: (n_observations, n_targets)
    '''
    x_quantiles = np.tile(x_quantiles, [n_samples, 1, 1])
    y_min, y_max = -0.9, 1.5
    y_quantiles = np.linspace(y_min, y_max, n_samples)
    y_quantiles = y_quantiles[..., np.newaxis, np.newaxis]
    y_quantiles = np.zeros_like(x_quantiles) + y_quantiles
    mean = np.tile(mean[:, np.newaxis], [1, n_samples, 1, 1])
    std = np.tile(std[:, np.newaxis], [1, n_samples, 1, 1])
    nlp = gaussian_loss(
            mean, std, y_quantiles, reduction=False)
    prob = np.exp(-0.1*nlp)

    # set figure size
    fig = plt.figure(figsize=(4.0, 2))
    # fig = plt.figure(figsize=(3.8, 2))
    # fig = plt.figure(figsize=(4.2, 2))

    ax = fig.add_subplot(1, 1, 1)
    plt.plot(
            x_observed[:, 0], y_observed[:, 0], '+',
            color='tab:blue', markersize=0.5)
    plt.pcolormesh(
            x_quantiles[..., 0], y_quantiles[..., 0], prob[..., 0],
            shading='gouraud', cmap='magma')
    plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=10)

    # # Turn off y axis tick and labels
    # ax.axes.yaxis.set_visible(False)

    # # Show colorbar
    # cb = plt.colorbar(pad=0.02)
    # cb.ax.tick_params(labelsize=10)

    # Set marging
    plt.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.97)
    # plt.subplots_adjust(left=0.05, right=0.97, bottom=0.12, top=0.97)
    # plt.subplots_adjust(left=0.05, right=1.03, bottom=0.12, top=0.97)

    outfile = prefix+'fit.png'
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(outfile)


def plot_nll_trace(nll, outfile, thinning=1):
    n_states = nll.shape[0]
    steps = np.arange(n_states) * thinning
    for i, x in enumerate(nll.T):
        plt.plot(steps, x, label=f'Chain {i}')
    plt.legend()
    plt.xlabel('posterior sample')
    plt.ylabel('negative log-likelihood')
    plt.savefig(outfile, bbox_inches='tight', dpi=200)
    plt.close()
    print(outfile)
