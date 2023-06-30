import argparse
import os
from time import time

import torch

from utils.utils import standardize
from utils.evaluate import evaluate_real, evaluate_synthetic
from dnne_experiment import set_seed, get_gpus, get_data
from vbnn import get_model as get_model_vbnn
from utils.visual import plot_density


class VariationalNN():

    def __init__(self, net, noise_scale, x_mean, x_std, y_mean, y_std):
        self.net = net
        self.noise_scale = noise_scale
        self.x_mean = torch.tensor(x_mean)
        self.x_std = torch.tensor(x_std)
        self.y_mean = torch.tensor(y_mean)
        self.y_std = torch.tensor(y_std)

    def eval(self):
        pass

    def sample(self, x, n_samples):
        y_mean, y_std = self.distribution(x=x, n_samples=n_samples)
        noise = torch.randn(*y_mean.shape)
        y_dist = noise * y_std + y_mean
        return y_dist

    def distribution(self, x, n_samples):
        x = (x - self.x_mean) / self.x_std
        y_mean = self.net.predict(
                x, num_predictions=n_samples, aggregate=False)
        y_mean = y_mean * self.y_std + self.y_mean
        y_std = self.noise_scale * self.y_std
        return y_mean, y_std


def eval_model_real(model, x, y, n_samples, prefix):
    y_dist = model.sample(torch.tensor(x), n_samples=n_samples)
    y_dist = y_dist.detach().numpy()
    y_dist = y_dist.reshape(-1, *y_dist.shape[-2:])
    evaluate_real(
            y=y, y_dist=y_dist, model=model, x=x, n_samples=n_samples,
            outfile=f'{prefix}eval.txt')


def eval_model_synthetic(
        model, x_quantiles, y_quantiles, x_observed, y_observed,
        n_samples, prefix):

    mean, std = model.distribution(
            torch.tensor(x_quantiles, dtype=torch.float),
            n_samples=n_samples)
    plot_density(
            mean.detach().numpy(), std.detach().numpy(),
            x_quantiles, x_observed, y_observed,
            n_samples=50, prefix=prefix)

    y_dist = model.sample(torch.tensor(x_quantiles), n_samples=n_samples)
    y_dist = y_dist.detach().numpy()
    y_dist = y_dist.reshape(-1, *y_dist.shape[-2:])
    evaluate_synthetic(
            x_quantiles=x_quantiles,
            y_quantiles=y_quantiles,
            x_observed=x_observed,
            y_observed=y_observed,
            y_dist=y_dist,
            prefix=prefix)


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
            '--split',
            type=int,
            default=0,
            help=(
                'Index of the training-testing data split.'
                'Only for real datasets.'
                'Examples: `0` (default)'))
    parser.add_argument(
            '--lr',
            type=float,
            default=1e-3,
            help=(
                'Learning rate'
                'Examples: 0.001 (default)'))
    parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help=(
                'Size of minibatches for training.'
                'Examples: 100 (default)'))
    parser.add_argument(
            '--epochs',
            type=int,
            default=2000,
            help=(
                'Number of samples for model training.'
                'Examples: 2000 (default)'))
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
    return parser.parse_args()


def get_model(x, y, n_layers, n_nodes, lr, batch_size, epochs, gpus, prefix):

    __, (x_mean, x_std) = standardize(x, return_info=True)
    __, (y_mean, y_std) = standardize(y, return_info=True)
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std

    n_features, n_targets = x.shape[-1], y.shape[-1]
    n_nodes = [n_features] + [n_nodes]*n_layers + [n_targets]
    t0 = time()
    net, noise_scale = get_model_vbnn(
            x=x, y=y, n_nodes=n_nodes,
            lr=lr, batch_size=batch_size, epochs=epochs)
    print('runtime:', int(time()-t0), 'sec')

    model = VariationalNN(
            net, noise_scale,
            x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    return model


def eval_model(model, data, prefix):
    eval_model_real(
            model=model,
            x=data['x_test'], y=data['y_test'],
            n_samples=1000,
            prefix=prefix)
    eval_synthetic = all([
        e in data.keys()
        for e in ['x_quantiles', 'y_quantiles', 'x_train', 'y_train']])
    if eval_synthetic:
        eval_model_synthetic(
                model=model,
                x_quantiles=data['x_quantiles'],
                y_quantiles=data['y_quantiles'],
                x_observed=data['x_train'],
                y_observed=data['y_train'],
                n_samples=1000,
                prefix=prefix)


def main():
    args = get_args()
    set_seed(args.seed)
    gpus = get_gpus(args.device)
    dataset = args.dataset
    data = get_data(dataset=dataset, split=args.split)

    prefix = args.prefix
    if prefix is None:
        prefix = f'results/vi/{dataset}/{args.split:02d}/'
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    model = get_model(
            x=data['x_train'], y=data['y_train'],
            n_layers=args.n_layers,
            n_nodes=args.n_nodes,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            gpus=gpus,
            prefix=prefix)

    if args.eval:
        set_seed(args.seed)
        eval_model(
                model=model,
                data=data,
                prefix=prefix)


if __name__ == '__main__':
    main()
