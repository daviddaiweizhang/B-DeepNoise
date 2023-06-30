from time import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
# from scipy import stats

from .nn import initialize_dalea_with_sgd
from .plot import show_performance, variance_vs_residsq, plot_curve


def gaussian_loss(y_observed, y_mean, y_std):
    return (
            0.5 * np.square((y_observed - y_mean) / y_std)
            + np.log(y_std)).mean((-1, -2))


def potential_scale_reduction(x):
    return float(tfp.mcmc.potential_scale_reduction(x, split_chains=False))


def cool_variance(model, name, decay):
    if name not in ['tau', 'sigma']:
        raise ValueError('Variance parameter name not recognized')
    for layer in range(model.n_hidden_layers+1):
        if name != 'sigma' or layer > 0:
            a_old = model.get_prior(f'a_{name}', layer)
            b_old = model.get_prior(f'b_{name}', layer)
            a_new = a_old / np.sqrt(decay)
            b_new = b_old * np.sqrt(decay)
            model.set_prior(f'a_{name}', layer, a_new)
            model.set_prior(f'b_{name}', layer, b_new)


def init_variance_prior(model, name, mean, n_observations):
    # TODO: replace (model.n_hidden_nodes+1) with 1?
    a = (
            n_observations * (model.n_hidden_nodes+1)
            / np.sqrt(mean) / 2)
    b = a * mean
    for layer in range(model.n_hidden_layers+1):
        if name != 'sigma' or layer > 0:
            model.set_prior(f'a_{name}', layer, a)
            model.set_prior(f'b_{name}', layer, b)


def get_rhat(x, start=0, plot_file=None, baseline=None):
    rhat = potential_scale_reduction(x[start:])
    n_chains = x.shape[-1]
    if plot_file is not None:
        cmap = plt.get_cmap('tab10')
        for i in range(n_chains):
            plt.plot(x[:, i], color=cmap(i))
            if baseline is not None:
                plt.axhline(
                        baseline[i], linestyle='--',
                        color=cmap(i))
        if start < 0:
            start += x.shape[0]
        plt.axvline(start, linestyle='--', color='tab:gray')
        plt.title(f'rhat: {rhat:.3f}')
        plt.savefig(plot_file, dpi=300)
        plt.close()
    return rhat


def plot_fit(
        x, y, y_dist, ci_falserate,
        y_mean_truth=None, xlim=None, outpref=None):

    assert x.ndim == 2
    assert y.ndim == 2
    assert y_dist.ndim == 3
    assert y.shape[-1] == y_dist.shape[-1] == 1

    y_mean = np.mean(y_dist, 0)

    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    show_performance(y_mean, y, y_dist, ci_falserate, name=outpref)
    plt.subplot(1, 2, 2)
    variance_vs_residsq(y, y_dist, name=outpref)
    if outpref is None:
        plt.show()
    else:
        plt.savefig(outpref + '_0.png', dpi=300)
        plt.close()

    plot_curve(y_dist, x, y, y_mean_truth, ci_falserate)

    if outpref is None:
        plt.show()
    else:
        plt.savefig(outpref + '_1.png', dpi=300)
        plt.close()


def plot_cooling(
        model,
        x, y, y_mean_train,
        x_test, y_test, y_mean_test,
        n_realizations, n_burnin, ci_falserate, outpref):
    y_dist_raw = model.predict(x, (n_realizations,), start=-n_burnin)
    test = (x_test is not None) and (y_test is not None)
    n_chains = y_dist_raw.shape[-3]
    if test:
        y_dist_raw_test = model.predict(
                x_test, (n_realizations,),
                start=-n_burnin)
    for chain in range(n_chains):
        y_dist = (
                y_dist_raw[:, -n_burnin:, chain, :, :]
                .reshape(-1, *y_dist_raw.shape[-2:]))
        outpref_plot_fit = (
                f'{outpref}_fit_train_chain{chain:03d}')
        plot_fit(
                x, y, y_dist, ci_falserate,
                y_mean_truth=y_mean_train,
                outpref=outpref_plot_fit)
        if test:
            y_dist_test = (
                    y_dist_raw_test[:, -n_burnin:, chain, :, :]
                    .reshape(-1, *y_dist_raw_test.shape[-2:]))
            outpref_plot_fit = (
                    f'{outpref}_fit_test_chain{chain:03d}')
            plot_fit(
                    x_test, y_test, y_dist_test,
                    ci_falserate,
                    y_mean_truth=y_mean_test,
                    outpref=outpref_plot_fit)
    y_dist_all = (
            y_dist_raw[:, -n_burnin:, :, :, :]
            .reshape(-1, *y_dist_raw.shape[-2:]))
    plot_fit(
            x, y, y_dist_all, ci_falserate,
            y_mean_truth=y_mean_train,
            outpref=f'{outpref}_fit_train_chainALL')
    if test:
        y_dist_all_test = (
                y_dist_raw_test[:, -n_burnin:, :, :, :]
                .reshape(-1, *y_dist_raw_test.shape[-2:]))
        plot_fit(
                x_test, y_test, y_dist_all_test, ci_falserate,
                y_mean_truth=y_mean_test,
                outpref=f'{outpref}_fit_test_chainALL')


def sample_with_cooling(
            model, x, y, init,
            n_states, n_thinning, n_chains,
            n_cooling, n_attempts, n_realizations,
            variance_decay_rate, variance_prior_mean_init,
            rhat_threshold, ci_falserate,
            y_mean_train, saveid,
            batch_size=None, plot=False,
            x_test=None, y_test=None, y_mean_test=None):

    # get validation observations
    n_observations = x.shape[-2]
    n_observations_valid = min(n_observations, 25)
    idx_valid = np.linspace(
            0, n_observations, n_observations_valid,
            endpoint=False, dtype=int)
    x_valid = x[..., idx_valid, :]
    y_valid = y[..., idx_valid, :]

    print('Initializing dalea...')
    t0 = time()
    if variance_prior_mean_init is not None:
        init_variance_prior(
                model, 'tau', variance_prior_mean_init,
                n_observations)
        init_variance_prior(
                model, 'sigma', variance_prior_mean_init,
                n_observations)
    if init == 'random':
        model.reset_params(
                n_observations=n_observations,
                chain_shape=(n_chains,),
                reset_beta_gamma='random',
                reset_u_v='random')
        y_mean_baseline_valid = None
        y_std_baseline_valid = None
    elif init == 'sgd':
        model_nn_mean_list, model_nn_logvar_list = initialize_dalea_with_sgd(
                model, x, y, chain_shape=(n_chains,),
                learning_rate=0.1,
                batch_size=32,
                epochs=10000,
                earlystop_patience=1000,
                reducelr_patience=200,
                verbose=0,
                x_test=x_test,
                y_test=y_test)
        y_mean_baseline_valid = np.stack([
            e.predict(x_valid) for e in model_nn_mean_list])
        y_std_baseline_valid = np.stack([
            np.exp(e.predict(x_valid))**0.5 for e in model_nn_logvar_list])
    elapse = int(time() - t0)
    print(f'Done ({elapse} sec).')

    adjustment_idx = []
    for i in range(n_cooling):
        print(f'cooling stage {i}')
        rhat = np.inf
        n_states_cooling = 0
        for j in range(n_attempts):
            model.sample_states(
                    x, y, n_states=n_states,
                    n_burnin=0, n_thinning=n_thinning,
                    reset='skip', chain_shape=(n_chains,),
                    batch_size=batch_size)
            adjustment_idx += [i] * n_states
            n_states_cooling += n_states
            n_burnin = int(n_states_cooling * 0.5)

            # plot fitting results
            if plot:
                outpref = f'results/{saveid}_{i:02d}'
                plot_cooling(
                        model=model,
                        x=x, y=y,
                        y_mean_train=y_mean_train,
                        x_test=x_test, y_test=y_test,
                        y_mean_test=y_mean_test,
                        n_realizations=n_realizations,
                        n_burnin=n_burnin,
                        ci_falserate=ci_falserate,
                        outpref=outpref)

            # compute rhat
            y_dist_raw_valid = model.predict(
                    x_valid, (n_realizations,), start=-n_burnin)
            y_mean_valid = y_dist_raw_valid.mean(0)
            y_std_valid = y_dist_raw_valid.std(0)
            gaussian_loss_valid = gaussian_loss(
                    y_valid, y_mean_valid, y_std_valid)
            if (
                    (y_mean_baseline_valid is not None)
                    and (y_std_baseline_valid is not None)):
                gaussian_loss_baseline_valid = gaussian_loss(
                        y_valid, y_mean_baseline_valid,
                        y_std_baseline_valid)
            else:
                gaussian_loss_baseline_valid = None
            rhat_plot_file = f'results/{saveid}/{i:02d}_trace.png'
            if n_states == 1:
                continue
            else:
                rhat = get_rhat(
                        gaussian_loss_valid, start=-n_burnin,
                        baseline=gaussian_loss_baseline_valid,
                        plot_file=rhat_plot_file)
                print(f'rhat: {rhat:.3f}')
                # check convergence
                if rhat < rhat_threshold:
                    break
                if j == n_attempts - 1:
                    print(
                            'Warning: MCMC has not converged '
                            '(as measured by rhat)')

        if i < n_cooling - 1:
            cool_variance(model, 'tau', variance_decay_rate)
        # TODO: adjust sigma?
        # cool_variance(model, 'sigma', variance_decay_rate)

    return adjustment_idx
