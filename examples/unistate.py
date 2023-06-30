import argparse
import os
import pickle
import sys

import torch
import numpy as np

from dalea.unistate import (
        DaleaUniState, DaleaUniStateEnsemble,
        train_model, nll_loss, min_euclid_err)
from realdata import get_data as get_data_real
from categorical import get_data as get_data_cat
from categorical import is_categorical_dataset
from synthesize import get_data as get_data_synthetic
from utils.utils import standardize
from utils.visual import plot_history, plot_density
from utils.evaluate import (
        evaluate_real, evaluate_synthetic,
        evaluate_influence, evaluate_individual,
        evaluate_categorical)

import matplotlib.pyplot as plt


def eval_hidden(model, x, y, n_samples, prefix):

    order = x[:, 0].argsort()
    x = x[order]
    y = y[order]

    model.cpu()
    model.eval()
    n_layers = len(model.net)
    x_tensor = torch.tensor(
            np.tile(x, [n_samples, 1, 1]),
            dtype=torch.float, device='cpu')

    for j in range(n_layers):
        z = model.net[:(j+1)](x_tensor).detach().numpy()
        n_nodes = z.shape[-1]
        plt.figure(figsize=(16, 8))
        for i in range(n_nodes):
            plt.subplot(4, 4, i+1)
            zi = z[:, :, i]
            q_list = (np.arange(20) + 0.5) / 20
            for q in q_list:
                zq = np.quantile(zi, q, axis=0)
                plt.plot(x[:, 0], zq, color='tab:blue', alpha=0.2)
            plt.title(i)
        outfile = f'{prefix}hidden-{j}.png'
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(outfile)


def fit_model_dalea_unistate(
        x, y, n_layers, n_nodes, activation, regularization,
        learning_rate, batch_size, epochs, n_samples, prefix,
        x_valid=None, y_valid=None, patience=None, verbose=False):

    validate = (x_valid is not None) and (y_valid is not None)

    if validate:
        x_all = np.concatenate([x, x_valid])
        y_all = np.concatenate([y, y_valid])
    else:
        x_all, y_all = x, y
    __, (x_mean, x_std) = standardize(x_all, return_info=True)
    __, (y_mean, y_std) = standardize(y_all, return_info=True)
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std
    if validate:
        x_valid = (x_valid - x_mean) / x_std
        y_valid = (y_valid - y_mean) / y_std
    print('target normalization:', y_mean, y_std)

    n_targets = y.shape[-1]
    n_train, n_features = x.shape

    if regularization is not None:
        regularization /= n_train

    model_file = f'{prefix}model.pt'
    history_file = f'{prefix}history.pickle'
    if os.path.exists(model_file):
        model = torch.load(model_file, map_location='cpu')
        print(f'Model loaded from {model_file}')
        history = pickle.load(open(history_file, 'rb'))
        print(f'History loaded from {history_file}')
    else:
        activation_dict = {
                'hardtanh': torch.nn.Hardtanh(),
                'leakyrelu': torch.nn.LeakyReLU(0.1)}
        activation = activation_dict[activation]
        model = DaleaUniState(
                n_features=n_features,
                n_targets=n_targets,
                n_layers=n_layers,
                n_nodes=n_nodes,
                activation=activation)
        model.to(device)
        # TODO: choose the best training and stopping criterion
        history = train_model(
                model=model,
                x=torch.tensor(x, dtype=torch.float, device=device),
                y=torch.tensor(y, dtype=torch.float, device=device),
                criterion=nll_loss,
                criterion_stopping=min_euclid_err,
                criterion_kwargs=dict(
                    n_samples=n_samples),
                criterion_stopping_kwargs=dict(
                    n_samples=100),
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                x_valid=torch.tensor(
                    x_valid, dtype=torch.float, device=device),
                y_valid=torch.tensor(
                    y_valid, dtype=torch.float, device=device),
                patience=patience,
                verbose=verbose)
        model.set_norm_info(
                x_mean=torch.tensor(
                    x_mean, dtype=torch.float, device=device),
                x_std=torch.tensor(
                    x_std, dtype=torch.float, device=device),
                y_mean=torch.tensor(
                    y_mean, dtype=torch.float, device=device),
                y_std=torch.tensor(
                    y_std, dtype=torch.float, device=device))
        model.cpu()
        torch.save(model, model_file)
        print(f'Model saved to {model_file}')
        pickle.dump(history, open(history_file, 'wb'))
        print(f'History saved to {history_file}')
        plot_history(
                history=history,
                outfile=f'{prefix}history.png')
    return model, history


def eval_model_synthetic(
        model, x_quantiles, y_quantiles, x_observed, y_observed,
        n_samples, prefix):

    if model is None:
        y_dist = y_quantiles
    else:
        model.cpu()
        model.eval()

        mean, std = model.distribution(
                torch.tensor(x_quantiles, dtype=torch.float),
                n_samples=n_samples)
        plot_density(
                mean.detach().numpy(), std.detach().numpy(),
                x_quantiles, x_observed, y_observed,
                n_samples=50, prefix=prefix)

        y_dist = (
                model(
                    torch.tensor(x_quantiles),
                    n_samples=n_samples)
                .cpu().detach().numpy())
    evaluate_synthetic(
            x_quantiles=x_quantiles,
            y_quantiles=y_quantiles,
            x_observed=x_observed,
            y_observed=y_observed,
            y_dist=y_dist,
            prefix=prefix)


def eval_model_real(
        model, x, y, n_samples, prefix,
        batch_size=None, categorical=False):
    model.cpu()
    model.eval()
    n_observations = x.shape[0]
    if batch_size is None:
        idx_list = [list(range(n_observations))]
    else:
        n_observations = x.shape[0]
        n_batches = (n_observations + batch_size - 1) // batch_size
        idx_list = np.array_split(list(range(n_observations)), n_batches)
    y_dist = np.concatenate([
            model(
                torch.tensor(x[idx], dtype=torch.float, device='cpu'),
                n_samples=n_samples)
            .cpu().detach().numpy()
            for idx in idx_list], axis=1)
    if categorical:
        evaluate_categorical(
                y=y, y_dist=y_dist, model=model, x=x,
                n_samples=n_samples,
                batch_size=batch_size,
                prefix=prefix)
    else:
        evaluate_real(
                y=y, y_dist=y_dist, model=model, x=x, n_samples=n_samples,
                batch_size=batch_size, outfile=f'{prefix}eval.txt')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--dataset',
            type=str,
            default=None,
            help=(
                'Name of dataset.'
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
                'Index of the training-testing data split.'
                'Only for real datasets.'
                'Examples: `0` (default)'))
    parser.add_argument(
            '--batch-size',
            type=int,
            default=200,
            help=(
                'Size of minibatches for training.'
                'Examples: 32 (default)'))
    parser.add_argument(
            '--epochs',
            type=int,
            default=2000,
            help=(
                'Number of samples for model training.'
                'Examples: 1000 (default)'))
    parser.add_argument(
            '--patience',
            type=int,
            default=200,
            help=(
                'Number of epochs to wait before checking for stopping.'
                'Examples: 200 (default)'))
    parser.add_argument(
            '--n-layers',
            type=int,
            default=4,
            help=(
                'Number of hidden layers in neural network.'
                'Examples: 4 (default)'))
    parser.add_argument(
            '--n-nodes',
            type=int,
            default=50,
            help=(
                'Number of hidden nodes per layer in neural network.'
                'Examples: 50 (default)'))
    parser.add_argument(
            '--samples',
            type=int,
            default=1000,
            help=(
                'Number of noise samples to generate during training.'
                'Examples: 1000 (default)'))
    parser.add_argument(
            '--lr',
            type=float,
            default=1e-3,
            help=(
                'Learning rate.'
                'Examples: 0.001 (default), 1e-4'))
    parser.add_argument(
            '--regularization',
            type=float,
            default=None,
            help=(
                'Regularization weight.'
                'Examples: 0 (default), 0.1'))
    parser.add_argument(
            '--fold',
            type=int,
            default=None,
            help=(
                'Fold index for validation. '
                'Default: no validation. '
                'Examples: 0.'))
    parser.add_argument(
            '--seed',
            type=int,
            default=None,
            help=(
                'Random seed for experiments.'
                'Examples: 0. '))
    parser.add_argument(
            '--device',
            type=str,
            default=None,
            help=(
                'Computation device to use for model training.'
                'Default: Use gpu if available; otherwise cpu.'
                'Examples: `cuda`, `cpu`'))
    parser.add_argument(
            '--prefix',
            type=str,
            default=None,
            help=(
                'Prefix of output files.'
                'Default: Generate based on settings.'))
    parser.add_argument(
            '--no-eval',
            dest='eval',
            action='store_false',
            help='Skip model evaluation.')
    parser.add_argument(
            '--ensemble-only',
            action='store_true',
            help='Evaluate ensemble only.')
    parser.add_argument(
            '--eval-influence',
            action='store_true',
            help='Evaluate influence.')
    parser.add_argument(
            '--eval-individual',
            action='store_true',
            help='Evaluate individuals.')
    parser.add_argument(
            '--plot-truth',
            action='store_true',
            help='Plot ground truth.')
    return parser.parse_args()


def set_seed(seed):
    if seed is None:
        seed = np.random.choice(1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f'seed: {seed}')


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def get_fold(x, fold, n_folds):
    assert fold < n_folds
    n = x.shape[0]
    idx_1 = list(range(fold, n, n_folds))
    idx_0 = list(set(range(n)).difference(idx_1))
    return x[idx_0], x[idx_1]


args = get_args()
set_seed(args.seed)
device = args.device
if device is None:
    device = get_device()
print(f'device: {device}')
prefix = args.prefix

data_is_synthetic = args.dataset is None
data_is_categorical = False
if not data_is_synthetic:
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

os.makedirs(
        os.path.dirname(f'{prefix}{args.fold}/'),
        exist_ok=True)
print('n_features: ', x_train.shape[1])
print('n_train: ', x_train.shape[0])
print('n_test: ', x_test.shape[0])

if args.fold is None:
    x_valid, y_valid = None, None
else:
    n_folds = 5
    x_train, x_valid = get_fold(
            x_train, fold=args.fold, n_folds=n_folds)
    y_train, y_valid = get_fold(
            y_train, fold=args.fold, n_folds=n_folds)

model, history = fit_model_dalea_unistate(
        x=x_train, y=y_train,
        n_layers=args.n_layers,
        n_nodes=args.n_nodes,
        activation='hardtanh',
        regularization=args.regularization,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_samples=args.samples,
        x_valid=x_valid, y_valid=y_valid,
        patience=args.patience,
        prefix=f'{prefix}{args.fold}/',
        verbose=True)

if not args.eval:
    sys.exit()

set_seed(args.seed)

if args.plot_truth:
    eval_model_synthetic(
            model=None,
            x_quantiles=x_quantiles,
            y_quantiles=y_quantiles,
            x_observed=None,
            y_observed=None,
            n_samples=None,
            prefix=f'{prefix}{args.fold}/truth-')

if args.n_nodes <= 16:
    eval_hidden(
            model=model, x=x_test, y=y_test,
            n_samples=1000, prefix=f'{prefix}{args.fold}/')

n_samples_test = 100 if data_is_categorical else 10000
batch_size_test = 100

if not args.ensemble_only:
    eval_model_real(
            model=model,
            x=x_test, y=y_test,
            n_samples=n_samples_test,
            batch_size=batch_size_test,
            prefix=f'{prefix}{args.fold}/test-',
            categorical=data_is_categorical)

    if data_is_synthetic:
        eval_model_synthetic(
                model=model,
                x_quantiles=x_quantiles,
                y_quantiles=y_quantiles,
                x_observed=x_train,
                y_observed=y_train,
                n_samples=10000,
                prefix=f'{prefix}{args.fold}/')

    if args.eval_influence:
        for method in ['mean', 'variance']:
            evaluate_influence(
                    model=model,
                    x=x_test,
                    method=method,
                    outfile=f'{prefix}{args.fold}/influence-{method}.txt')

    if args.eval_individual:
        evaluate_individual(
                model=model,
                x=x_test, y=y_test,
                prefix=f'{prefix}{args.fold}/individual-')

modfile_list = [
        f'{prefix}{fo}/model.pt'
        for fo in range(n_folds)]
hisfile_list = [
        f'{prefix}{fo}/history.pickle'
        for fo in range(n_folds)]
if all([os.path.exists(f) for f in modfile_list]):
    model_list = [torch.load(f, map_location='cpu') for f in modfile_list]
    if all([os.path.exists(f) for f in hisfile_list]):
        history_list = [pickle.load(open(f, 'rb')) for f in hisfile_list]
        loss_list = np.array([e['smooth'][-1] for e in history_list])
    else:
        loss_list = None
    model_ensemble = DaleaUniStateEnsemble(model_list, loss_list)
    os.makedirs(
            os.path.dirname(f'{prefix}ensemble/'),
            exist_ok=True)
    eval_model_real(
            model=model_ensemble,
            x=x_test, y=y_test,
            n_samples=n_samples_test,
            batch_size=batch_size_test,
            prefix=f'{prefix}ensemble/',
            categorical=data_is_categorical)
    if data_is_synthetic:
        eval_model_synthetic(
                model=model_ensemble,
                x_quantiles=x_quantiles,
                y_quantiles=y_quantiles,
                x_observed=x_train,
                y_observed=y_train,
                n_samples=10000,
                prefix=f'{prefix}ensemble/')

    outfile_ensemble = f'{prefix}ensemble/model.pt'
    torch.save(model_ensemble, outfile_ensemble)
    print(outfile_ensemble)

    if args.eval_influence:
        for method in ['mean', 'variance']:
            evaluate_influence(
                    model=model,
                    x=x_test,
                    method=method,
                    outfile=f'{prefix}ensemble/influence-{method}.txt')

    if args.eval_individual:
        evaluate_individual(
                model=model,
                x=x_test,
                y=y_test,
                prefix=f'{prefix}ensemble/individual-')
