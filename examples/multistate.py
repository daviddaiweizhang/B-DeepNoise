import argparse
import os
from os.path import dirname
import pickle

import numpy as np
import torch

from realdata import get_data as get_data_real
from categorical import get_data as get_data_cat
from categorical import is_categorical_dataset
from synthesize import get_data as get_data_synthetic
from dalea.model import DaleaGibbs, load_from_disk
from utils.utils import standardize
from utils.evaluate import (
        evaluate_real, evaluate_synthetic,
        print_eval_by_state, get_nll_trace,
        evaluate_categorical)
from utils.visual import plot_density, plot_nll_trace


def fit_slope(x, y):
    return (x * y).sum() / (x * x).sum()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--dataset',
            type=str,
            default=None,
            help=(
                'Name of dataset. '
                'Examples: None (default), `kin8nm`'))
    parser.add_argument(
            '--x-distribution',
            type=str,
            default='uniform',
            help=(
                'Distribution of x.'
                'Examples: uniform (default), interval. '))
    parser.add_argument(
            '--y-deviation',
            type=str,
            default='heteroscedastic',
            help=(
                'Noise distribution of y.'
                'Examples: `heteroscedastic` (default), '
                '`skewed`, `multimodal`. '))
    parser.add_argument(
            '--n-features',
            type=int,
            default=1,
            help=(
                'Dimension of the input variable.'
                'Examples: 1(default), 6'))
    parser.add_argument(
            '--n-train',
            type=int,
            default=100,
            help=(
                'Number of training samples. '
                'Examples: 100 (default), 500, 2000'))
    parser.add_argument(
            '--split',
            type=int,
            default=0,
            help=(
                'Index of the training-testing data split. '
                'Only for UCI datasets. '
                'Examples: `0` (default)'))
    parser.add_argument(
            '--batch-size',
            type=int,
            default=None,
            help=(
                'Size of minibatches for training. '
                'Examples: None (one batch contains all samples), 32'))
    parser.add_argument(
            '--n-states',
            type=int,
            default=100,
            help=(
                'Number of posterior samples to draw. '
                'Examples: 100 (default)'))
    parser.add_argument(
            '--n-thinning',
            type=int,
            default=1,
            help=(
                'Save every `n_thinning` states. '
                'Examples: 1 (default)'))
    parser.add_argument(
            '--n-samples',
            type=int,
            default=100,
            help=(
                'Number of noise samples to draw for evaluation. '
                'Examples: 100 (default)'))
    parser.add_argument(
            '--n-layers',
            type=int,
            default=4,
            help=(
                'Number of hidden layers in neural network. '
                'Examples: 4 (default)'))
    parser.add_argument(
            '--n-nodes',
            type=int,
            default=50,
            help=(
                'Number of hidden nodes per layer in neural network. '
                'Examples: 50 (default)'))
    parser.add_argument(
            '--seed',
            type=int,
            default=None,
            help=(
                'Random seed for experiments. '
                'Examples: 0. '))
    parser.add_argument(
            '--prefix',
            type=str,
            default=None,
            help=(
                'Prefix of output files. '
                'Default: Generate based on settings.'))
    parser.add_argument(
            '--debug',
            action='store_true',
            help='Turn on debugging funcitons.')
    parser.add_argument(
            '--no-eval',
            dest='eval',
            action='store_false',
            help='Skip model evaluation.')
    return parser.parse_args()


def set_seed(seed):
    if seed is None:
        seed = np.random.choice(1000)
    np.random.seed(seed)
    print(f'seed: {seed}')


def convert_to_multistate_param(model, name, layer):
    if name == 'beta':
        weight = model.net[layer].linear.weight.detach().numpy()
        param = weight.swapaxes(-1, -2)
    elif name == 'gamma':
        bias = model.net[layer].linear.bias.detach().numpy()
        param = np.expand_dims(bias, 0)
    elif name == 'sigma':
        logscale = model.net[layer].noise_postact.log_scale.detach().numpy()
        param = np.expand_dims(np.exp(logscale), -2)
    elif name == 'tau':
        logscale = model.net[layer].noise_preact.log_scale.detach().numpy()
        param = np.expand_dims(np.exp(logscale), -2)
    elif name == 'rho':
        weight = model.net[layer].linear.weight.detach().numpy()
        param = np.abs(weight).swapaxes(-1, -2)
    elif name == 'xi':
        bias = model.net[layer].linear.bias.detach().numpy()
        param = np.expand_dims(np.abs(bias), -2)
    return param


def set_multistate_params_from_unistate(multistate, unistate, x, y):
    n_observations = x.shape[0]
    multistate.reset_params(
            n_observations=n_observations,
            reset_sigma_tau='random', reset_rho_xi='random',
            reset_beta_gamma='random', reset_u_v='random',
            x=x)
    multistate.set_data(x=x, y=y)

    for name in ['beta', 'gamma', 'sigma', 'tau', 'rho', 'xi']:
        for layer in range(multistate.n_hidden_layers+1):
            if multistate.param_exists(name, layer):
                value = np.stack([
                    convert_to_multistate_param(mod, name, layer)
                    for mod in unistate.models
                    ], axis=0)
                multistate.set_param(name, layer, value)

    for name in ['sigma', 'tau']:
        for layer in range(multistate.n_hidden_layers+1):
            if multistate.param_exists(name, layer):
                param = multistate.get_param(name, layer)
                a_old = multistate.get_prior(f'a_{name}', layer)
                a_new = np.ones_like(a_old) * n_observations / 2
                b_new = param**2 * a_new
                multistate.set_prior(f'a_{name}', layer, a_new)
                multistate.set_prior(f'b_{name}', layer, b_new)

    multistate.reset_params(
            n_observations,
            reset_sigma_tau='skip',
            reset_rho_xi='skip',
            reset_beta_gamma='skip',
            reset_u_v='semideterministic',
            x=x)

    layer_list = list(range(multistate.n_hidden_layers+1))
    layer_list = layer_list[::-1] + layer_list
    for i in range(1):
        for layer in layer_list:
            if layer < multistate.n_hidden_layers:
                multistate.update_v(layer)
            if layer > 0:
                multistate.update_u(layer)


def fit_model_dalea_multistate(
        x, y, n_layers, n_nodes, activation,
        n_states, batch_size, n_thinning,
        model_unistate, prefix, **kwargs):

    activation_dict = {
            'hardtanh': 'hard_tanh',
            'leakyrelu': 'leaky_relu'}
    activation = activation_dict[activation]

    __, (x_mean, x_std) = standardize(x, return_info=True)
    __, (y_mean, y_std) = standardize(y, return_info=True)
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std
    print('target normalization:', y_mean, y_std)

    n_observations, n_features = x.shape
    n_targets = y.shape[-1]
    n_observations, n_features = x.shape
    n_targets = y.shape[-1]

    model = DaleaGibbs(
            n_features=n_features,
            n_targets=n_targets,
            n_hidden_nodes=n_nodes,
            n_hidden_layers=n_layers,
            target_type='continuous',
            activation=activation,
            chain_shape=(n_chains,),
            save_all_params=False,
            save_priors_history=False)
    if model_unistate is None:
        reset = 'random'
    else:
        reset = 'skip'
        set_multistate_params_from_unistate(
                multistate=model, unistate=model_unistate, x=x, y=y)

    model.set_norm_info(
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std)
    model.sample_states(
            x, y,
            n_states=n_states,
            batch_size=batch_size,
            n_thinning=n_thinning,
            n_burnin=0, reset=reset,
            verbose=True)

    return model


def test_dist_mean_std(model, model_unistate, x, n_samples):
    y_dist_unistate = model_unistate(
            torch.tensor(x, dtype=torch.float32),
            n_samples=n_samples).detach().numpy()
    y_dist = model.predict(
            x, realization_shape=(n_samples,),
            start=0, stop=1)
    slope_mean = fit_slope(
            y_dist_unistate.mean(0),
            y_dist.reshape(-1, *y_dist.shape[-2:]).mean(0))
    assert 0.9 < slope_mean < 1.1
    slope_std = fit_slope(
            np.log(y_dist_unistate.std(0)),
            np.log(y_dist.reshape(-1, *y_dist.shape[-2:]).std(0)))
    assert 0.9 < slope_std < 1.1


def test_dist_quantiles(model, model_unistate, x, n_samples):
    add_random_effects = True
    if not add_random_effects:
        n_samples = 1

    yd_multi = model.predict(
            x=x,
            realization_shape=(n_samples,),
            add_random_effects=add_random_effects,
            start=0, stop=1)[:, 0]

    if not add_random_effects:
        for mod in model_unistate.models:
            for i, lay in enumerate(mod.net):
                if i > 0:
                    lay.noise_postact.log_scale = torch.nn.Parameter(
                            torch.tensor([-np.inf]))
                lay.noise_preact.log_scale = torch.nn.Parameter(
                        torch.tensor([-np.inf]))
    yd_uni = np.stack([
        mod(
            torch.tensor(x).float().cpu(),
            n_samples=n_samples).detach().numpy()
        for mod in model_unistate.models],
        axis=1)

    if add_random_effects:
        quantiles = (np.arange(20) + 0.5) / 20
    else:
        quantiles = [0.5]
    yq_uni = np.quantile(yd_uni, quantiles, axis=0)
    yq_multi = np.quantile(yd_multi, quantiles, axis=0)
    for yqu, yqm in zip(yq_uni, yq_multi):
        for yquu, yqmm in zip(yqu, yqm):
            slope = fit_slope(yquu.flatten(), yqmm.flatten())
            assert 0.9 < slope < 1.1


def check_eval_by_state(model, x, y, n_samples):
    n_batches = 10
    x_batches = np.split(x, n_batches)
    out = [
            model.predict(
                x=xb, realization_shape=(n_samples,),
                return_distribution=True)
            for xb in x_batches]
    mean = np.concatenate([e[1] for e in out], axis=-2)
    std = out[0][2]
    print_eval_by_state(mean=mean, std=std, y=y)


def debug_model_dalea_multistate(model, x, y, n_samples, model_unistate):
    check_eval_by_state(model, x, y, n_samples=n_samples)
    if model_unistate is not None:
        test_dist_mean_std(model, model_unistate, x, n_samples=n_samples)
        test_dist_quantiles(model, model_unistate, x, n_samples=n_samples)


def eval_model_synthetic(
        model, x_quantiles, y_quantiles, x_observed, y_observed,
        n_samples, prefix, thinning=None):
    y_dist, mean, std = model.predict(
            x=x_quantiles,
            realization_shape=(n_samples,),
            return_distribution=True)
    plot_density(
            mean.reshape((-1,)+mean.shape[-2:]),
            std.reshape((-1,)+std.shape[-2:]),
            x_quantiles, x_observed, y_observed,
            n_samples=50, prefix=prefix)

    n_observations = x_quantiles.shape[0]
    std = np.tile(std, (n_observations, 1))

    # mean and std are for the posterior predictive distributions
    # of x_quantiles
    nll = get_nll_trace(mean, std, y_quantiles)
    outfile = prefix + 'nll.pickle'
    pickle.dump(nll, open(outfile, 'wb'))
    print(outfile)
    plot_nll_trace(nll, thinning=thinning, outfile=prefix+'trace-nll.png')

    y_dist = y_dist.reshape(-1, *y_dist.shape[-2:])
    evaluate_synthetic(
            x_quantiles=x_quantiles,
            y_quantiles=y_quantiles,
            x_observed=x_observed,
            y_observed=y_observed,
            y_dist=y_dist,
            prefix=prefix)


def eval_model_real(model, x, y, n_samples, prefix, categorical, batch_size):

    n_features = x.shape[-1]
    if n_features == 1:
        order = x[:, 0].argsort()
        x = x[order]
        y = y[order]

    y_dist, mean, std = model.predict(
            x=x, realization_shape=(n_samples,),
            return_distribution=True)

    y_dist = y_dist.reshape((-1,) + y_dist.shape[-2:])
    mean = mean.reshape((-1,) + mean.shape[-2:])
    std = std.reshape((-1,) + std.shape[-2:])

    if categorical:
        evaluate_categorical(
                y=y, y_dist=y_dist, model=model, x=x,
                n_samples=n_samples,
                batch_size=batch_size,
                prefix=prefix)
    else:
        evaluate_real(
                y=y, y_dist=y_dist,
                mean=mean, std=std,
                outfile=f'{prefix}eval.txt')


args = get_args()
set_seed(args.seed)
prefix = args.prefix

data_is_synthetic = args.dataset is None
if args.dataset is None:
    data_is_categorical = False
else:
    data_is_categorical = is_categorical_dataset(args.dataset)
if data_is_synthetic:
    dataset = (
            f'features{args.n_features}-'
            f'{args.x_distribution}-'
            f'{args.y_deviation}-'
            f'ntrain{args.n_train:04d}')
    if prefix is None:
        prefix = f'results/{dataset}/{args.split:02d}/'
    data = get_data_synthetic(
            dataset=dataset,
            split=args.split)
    x_quantiles, y_quantiles = data[0]
    x_train, y_train = data[1]
    x_test, y_test = data[2]
    x_test, y_test = x_test[::10], y_test[::10]
    idx = np.round(
            np.linspace(0, 1, 100)
            * (x_quantiles.shape[0]-1)).astype(int)
    x_quantiles, y_quantiles = x_quantiles[idx], y_quantiles[:, idx]
else:
    if data_is_categorical:
        if prefix is None:
            prefix = (f'results/{args.dataset}/00/')
        (x_train, y_train), (x_test, y_test) = get_data_cat(
                args.dataset)
    else:
        if prefix is None:
            prefix = (f'results/{args.dataset}/{args.split:02d}/')
        (x_train, y_train), (x_test, y_test) = get_data_real(
                dataset=args.dataset, split=args.split)
prefix = prefix + 'gibbs/'

unistate_file = f'{dirname(dirname(prefix))}/ensemble/model.pt'
if os.path.exists(unistate_file):
    model_unistate = torch.load(unistate_file, map_location='cpu')
    n_chains = len(model_unistate.models)
    print(f'Initial model loaded from {unistate_file}')
else:
    model_unistate = None
    n_chains = 5

model_file = f'{prefix}model.pickle'
if os.path.exists(model_file):
    model = load_from_disk(model_file)
    print(f'Model loaded from {model_file}')
else:
    os.makedirs(
            os.path.dirname(model_file),
            exist_ok=True)
    model = fit_model_dalea_multistate(
            x=x_train, y=y_train,
            n_layers=args.n_layers,
            n_nodes=args.n_nodes,
            activation='hardtanh',
            n_states=args.n_states,
            batch_size=args.batch_size,
            n_thinning=args.n_thinning,
            n_chains=n_chains,
            model_unistate=model_unistate,
            prefix=prefix)
    model.save_to_disk(model_file)
    print(f'Model saved to {model_file}')
if args.debug:
    debug_model_dalea_multistate(
            model=model, model_unistate=model_unistate,
            x=x_train, y=y_train, n_samples=args.n_samples)
if args.eval:

    if data_is_synthetic:
        eval_model_synthetic(
                model=model,
                x_quantiles=x_quantiles,
                y_quantiles=y_quantiles,
                x_observed=x_train,
                y_observed=y_train,
                n_samples=args.n_samples,
                prefix=prefix,
                thinning=args.n_thinning)

    eval_model_real(
            model=model, x=x_test, y=y_test,
            n_samples=args.n_samples, prefix=prefix,
            categorical=data_is_categorical,
            batch_size=args.batch_size)
