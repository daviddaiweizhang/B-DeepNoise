import argparse
import os
from time import time

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np

from dnne.dnne import MlpGaussianEnsemble
from realdata import get_data as get_data_real
from synthesize import get_data as get_data_synthetic
from utils.utils import standardize
from utils.evaluate import evaluate_real, evaluate_synthetic
from utils.visual import plot_density


class GenericDataset(Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y


def fit_model(
        model_class, x, y, n_layers, n_nodes, activation, learning_rate,
        batch_size, epochs, n_nets, gpus, prefix):

    __, (x_mean, x_std) = standardize(x, return_info=True)
    __, (y_mean, y_std) = standardize(y, return_info=True)
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std
    print('target normalization:', y_mean, y_std)

    dataset = GenericDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size)

    n_targets = y.shape[-1]
    n_train, n_features = x.shape
    arch = [n_features] + [n_nodes] * n_layers + [n_targets]

    activation_dict = {'relu': torch.nn.ReLU()}
    activation = activation_dict[activation]
    model = model_class(
            n_nodes=arch,
            activation=activation,
            n_nets=n_nets,
            dropout_rate=None)
    trainer = pl.Trainer(
            max_epochs=epochs,
            deterministic=True,
            gpus=gpus,
            enable_checkpointing=False,
            default_root_dir=os.path.dirname(prefix))
    trainer.fit(
            model=model,
            train_dataloaders=loader)
    model.set_norm_info(
            x_mean=torch.tensor(x_mean),
            x_std=torch.tensor(x_std),
            y_mean=torch.tensor(y_mean),
            y_std=torch.tensor(y_std))
    model.eval()
    return model, trainer


def eval_model_real(model, x, y, n_samples, prefix):
    model.eval()
    y_dist = (
            model(torch.tensor(x), n_samples=n_samples)
            .detach().numpy())
    evaluate_real(
            y=y, y_dist=y_dist, model=model, x=x, n_samples=n_samples,
            outfile=f'{prefix}eval.txt')


def eval_model_synthetic(
        model, x_quantiles, y_quantiles, x_observed, y_observed,
        n_samples, prefix):

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
            model(torch.tensor(x_quantiles), n_samples=n_samples)
            .cpu().detach().numpy())
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
            default=100,
            help=(
                'Size of minibatches for training.'
                'Examples: 100 (default)'))
    parser.add_argument(
            '--epochs',
            type=int,
            default=40,
            help=(
                'Number of samples for model training.'
                'Examples: 40 (default)'))
    parser.add_argument(
            '--n-nets',
            type=int,
            default=4,
            help=(
                'Number of networks in ensemble.'
                'Examples: 5 (default)'))
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


def set_seed(seed):
    if seed is None:
        seed = np.random.choice(1000)
    pl.seed_everything(seed, workers=True)
    print(f'seed: {seed}')


def get_data(dataset, split):
    out = {}
    if is_synthetic(dataset):
        data = get_data_synthetic(
                dataset=dataset,
                split=split)
        x_quantiles, y_quantiles = data[0]
        x_train, y_train = data[1]
        x_test, y_test = data[2]
        x_test, y_test = x_test[::10], y_test[::10]
        idx = np.round(
                np.linspace(0, 1, 100)
                * (x_quantiles.shape[0]-1)).astype(int)
        x_quantiles, y_quantiles = x_quantiles[idx], y_quantiles[:, idx]
        out['x_quantiles'] = x_quantiles
        out['y_quantiles'] = y_quantiles
    else:
        data = get_data_real(dataset=dataset, split=split)
        (x_train, y_train), (x_test, y_test) = data
    out['x_train'] = x_train.astype(np.float32)
    out['y_train'] = y_train.astype(np.float32)
    out['x_test'] = x_test.astype(np.float32)
    out['y_test'] = y_test.astype(np.float32)

    print('n_features: ', out['x_train'].shape[1])
    print('n_train: ', out['x_train'].shape[0])
    print('n_test: ', out['x_test'].shape[0])

    return out


def get_model(
        model_class, x, y, n_layers, n_nodes, n_nets, batch_size, epochs,
        gpus, prefix):
    checkpoint_file = f'{prefix}model.ckpt'
    if os.path.exists(checkpoint_file):
        model = model_class.load_from_checkpoint(checkpoint_file)
        print(f'Model loaded from {checkpoint_file}')
    else:
        t0 = time()
        model, trainer = fit_model(
                model_class=model_class, x=x, y=y,
                n_layers=n_layers,
                n_nodes=n_nodes,
                activation='relu',
                learning_rate=0.1,
                batch_size=batch_size,
                epochs=epochs,
                n_nets=n_nets,
                gpus=gpus,
                prefix=prefix)
        print('runtime:', int(time() - t0), 'sec')
        trainer.save_checkpoint(checkpoint_file)
        print(f'Model saved to {checkpoint_file}')
    return model


def eval_model(model, data, prefix):
    n_samples = 10000 if data['x_test'].shape[0] < 10000 else 1000
    eval_model_real(
            model=model,
            x=data['x_test'], y=data['y_test'],
            n_samples=n_samples,
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
                n_samples=10000,
                prefix=prefix)


def get_gpus(device):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            gpus = 1
        else:
            device = 'cpu'
            gpus = None
    else:
        gpus = None
    print(f'device: {device}')
    return gpus


def is_synthetic(dataset):
    return dataset.startswith('features')


def main():
    args = get_args()
    set_seed(args.seed)
    gpus = get_gpus(args.device)
    dataset = args.dataset
    data = get_data(dataset=dataset, split=args.split)

    prefix = args.prefix
    if prefix is None:
        prefix = f'results/dnne/{dataset}/{args.split:02d}/'
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    model = get_model(
            model_class=MlpGaussianEnsemble,
            x=data['x_train'], y=data['y_train'],
            n_layers=args.n_layers,
            n_nodes=args.n_nodes,
            n_nets=args.n_nets,
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
