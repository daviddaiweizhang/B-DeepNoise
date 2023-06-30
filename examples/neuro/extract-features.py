import pickle
import os
from copy import deepcopy
import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dnne.dnne import Mlp
from make_splits import save_data, save_split_index


def split_data(x, y, prop_train):
    if prop_train is None:
        x_train = x
        y_train = y
        x_valid = None
        y_valid = None
    else:
        n_observations = x.shape[0]
        n_train = int(n_observations * prop_train)
        order = np.random.choice(n_observations, n_observations, replace=False)
        x_train = x[order[:n_train]]
        x_valid = x[order[n_train:]]
        y_train = y[order[:n_train]]
        y_valid = y[order[n_train:]]
    return (x_train, y_train), (x_valid, y_valid)


def plot_history(history, prefix, log=True):
    plt.figure(figsize=(16, 8))
    for metric in history[0].keys():
        hist = [e[metric] for e in history]
        plt.plot(hist, label=metric)
    if log:
        plt.yscale('log')
    plt.legend()
    outfile = f'{prefix}history.png'
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(outfile)


def set_seed(seed):
    seed = seed
    if seed is None:
        seed = np.random.choice(100)
    pl.seed_everything(seed, workers=True)
    print(f'seed: {seed}')


def resize_imgs(imgs):
    assert np.ndim(imgs) == 4
    shape_inp = imgs.shape[-3:]
    shape_out = 2**(np.round(np.log2(shape_inp)).astype(int))
    imgs = np.array([resize(im, shape_out) for im in imgs])
    return imgs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--dataset',
            type=str,
            required=True,
            help=(
                'Name of dataset.'
                'Examples: abide, abcd. '))
    parser.add_argument(
            '--prop-train',
            type=float,
            default=None,
            help=(
                'Proportion of validation samples.'
                'Examples: 0.8. '
                'Default: None (no validation). '))
    parser.add_argument(
            '--seed',
            type=int,
            default=None,
            help=(
                'Random seed for experiments.'
                'Examples: 0. '
                'Default: None (randomly generated). '))
    parser.add_argument(
            '--run-cnn',
            action='store_true',
            help='Run cnn experiments')
    parser.add_argument(
            '--run-ae',
            action='store_true',
            help='Run autoencoder-mlp experiments')
    parser.add_argument(
            '--run-pca',
            action='store_true',
            help='Run pca experiments')
    return parser.parse_args()


class ConvFeedForward(nn.Module):

    def __init__(
            self, n_channels, upsample=False,
            kernel_size=3, dropout_rate=None):
        super().__init__()
        layers = []
        n_layers = len(n_channels) - 2
        for i in range(n_layers+1):
            activation = nn.GELU()
            if dropout_rate is None:
                dropout = nn.Identity()
            else:
                dropout = nn.Dropout(dropout_rate)
            n_inp, n_out = n_channels[i:(i+2)]
            conv = nn.Conv3d(
                    in_channels=n_inp,
                    out_channels=n_out,
                    kernel_size=kernel_size,
                    padding='same')
            if upsample:
                pool = nn.Upsample(scale_factor=2)
            else:
                pool = nn.MaxPool3d(kernel_size=2)
            layers.append(conv)
            if i < n_layers:
                layers.append(dropout)
                layers.append(activation)
                layers.append(pool)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvEncoder(nn.Module):

    def __init__(
            self, channels_in, channels_out, n_layers,
            dropout_rate=None):
        super().__init__()

        channels_all = [channels_in] + [
                channels_out // 2**(n_layers - i)
                for i in range(n_layers+1)]
        self.net = ConvFeedForward(
                n_channels=channels_all,
                upsample=False,
                dropout_rate=dropout_rate)

    def forward(self, x):
        return self.net(x)


class ConvDecoder(nn.Module):

    def __init__(
            self, channels_in, channels_out, n_layers,
            dropout_rate=None):
        super().__init__()

        channels_all = [
                channels_in // 2**i
                for i in range(n_layers+1)] + [channels_out]
        self.net = ConvFeedForward(
                n_channels=channels_all,
                upsample=True,
                dropout_rate=dropout_rate)

    def forward(self, x):
        return self.net(x)


class ConvAutoencoder(pl.LightningModule):

    def __init__(
            self, img_depth, img_height, img_width,
            channels_in, channels_latent,
            n_layers, n_latent, dropout_rate,
            lr):
        super().__init__()

        latent_shape = (
                channels_latent,
                img_depth // 2**n_layers,
                img_height // 2**n_layers,
                img_width // 2**n_layers)

        encoder_conv = ConvEncoder(
                channels_in=channels_in,
                channels_out=channels_latent,
                n_layers=n_layers,
                dropout_rate=dropout_rate)
        encoder_reshape = nn.Flatten(start_dim=1, end_dim=-1)
        encoder_linear = nn.Linear(
                np.prod(latent_shape), n_latent)
        self.encoder = nn.Sequential(
                encoder_conv,
                encoder_reshape,
                encoder_linear)

        decoder_linear = nn.Linear(
                n_latent, np.prod(latent_shape))
        decoder_reshape = nn.Unflatten(
                dim=-1,
                unflattened_size=latent_shape)
        decoder_conv = ConvDecoder(
                channels_in=channels_latent,
                channels_out=channels_in,
                n_layers=n_layers,
                dropout_rate=dropout_rate)
        self.decoder = nn.Sequential(
                decoder_linear,
                decoder_reshape,
                decoder_conv)

        self.lr = lr
        self.save_hyperparameters()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs = batch
        reconstructed = self.forward(imgs)
        loss = F.mse_loss(reconstructed, imgs)
        self.log('loss_train', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch
        reconstructed = self.forward(imgs)
        loss = F.mse_loss(reconstructed, imgs)
        self.log('loss_valid', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class CNN(pl.LightningModule):

    def __init__(
            self, n_targets, img_depth, img_height, img_width,
            channels_in, channels_latent, n_latent,
            n_layers_conv, n_layers_mlp,
            dropout_rate, lr):
        super().__init__()

        latent_shape = (
                channels_latent,
                img_depth // 2**n_layers_conv,
                img_height // 2**n_layers_conv,
                img_width // 2**n_layers_conv)

        encoder_conv = ConvEncoder(
                channels_in=channels_in,
                channels_out=channels_latent,
                n_layers=n_layers_conv,
                dropout_rate=dropout_rate)
        encoder_reshape = nn.Flatten(start_dim=1, end_dim=-1)
        encoder_linear = nn.Linear(
                np.prod(latent_shape), n_latent)
        self.encoder = nn.Sequential(
                encoder_conv,
                encoder_reshape,
                encoder_linear)

        n_nodes = [n_latent] + [n_latent] * n_layers_mlp + [n_targets]
        self.decoder = Mlp(
                n_nodes=n_nodes, activation=nn.GELU(),
                dropout_rate=dropout_rate, lr=None)

        self.lr = lr
        self.save_hyperparameters()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log('loss_train', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        mse = F.mse_loss(y_pred, y)
        corr = torch.corrcoef(
                torch.stack([y_pred.flatten(), y.flatten()]))[0, 1]
        self.log('mse_valid', mse, prog_bar=True)
        self.log('corr_valid', corr, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MetricTracker(pl.Callback):

    def __init__(self):
        self.collection = []

    def on_train_epoch_end(self, trainer, module):
        metrics = deepcopy(trainer.logged_metrics)
        self.collection.append(metrics)

    def clean(self):
        keys = [set(e.keys()) for e in self.collection]
        keys = set().union(*keys)
        for elem in self.collection:
            for ke in keys:
                if isinstance(elem[ke], torch.Tensor):
                    elem[ke] = elem[ke].item()


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


class ImageDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        im = self.imgs[idx]
        return im


def get_data(dataset, linear_preprocess=False):
    imgs, mask, targets = load_data(dataset)
    if linear_preprocess:
        imgs = imgs[..., mask]
        imgs -= imgs.mean(0)
        imgs /= imgs.std(0)
    else:
        imgs = standardize_imgs(imgs, mask)
        imgs = imgs.swapaxes(-3, -1)
        imgs = resize_imgs(imgs)
        imgs = imgs[:, np.newaxis, :, :, :]
    targets = targets[:, np.newaxis]
    return imgs, targets


def load_data(dataset):
    infile = f'data/neuroimaging/{dataset}/raw.pickle'
    data = pickle.load(open(infile, 'rb'))
    imgs = data['imgs']
    mask = data['mask']
    target = data['target']
    return imgs, mask, target


def standardize_imgs(imgs, mask):

    imgs[..., ~mask] = np.nan
    imgs -= np.nanmin(imgs)
    imgs = np.log(imgs)
    imgs -= np.quantile(imgs[..., mask], 0.01)
    imgs /= np.quantile(imgs[..., mask], 0.99)
    imgs = np.clip(imgs, 0, 1)
    imgs[..., ~mask] = 0.0
    return imgs


def fit_model(
        model, loader, loader_valid,
        epochs, batch_size, lr,
        gpus, prefix):
    tracker = MetricTracker()
    trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[tracker],
            gpus=gpus,
            enable_checkpointing=False,
            default_root_dir=os.path.dirname(prefix))
    trainer.fit(
            model=model,
            train_dataloaders=loader,
            val_dataloaders=loader_valid)
    tracker.clean()
    history = tracker.collection
    return model, trainer, history


def fit_autoencoder(
        imgs, imgs_valid,
        epochs, batch_size, lr,
        gpus, prefix):
    dataset = ImageDataset(imgs.astype(np.float32))
    loader = DataLoader(dataset, batch_size=batch_size)
    print('n_train: ', len(dataset))
    if imgs_valid is not None:
        dataset_valid = ImageDataset(imgs_valid.astype(np.float32))
        n_valid = len(dataset_valid)
        loader_valid = DataLoader(
                dataset_valid, batch_size=n_valid)
        print('n_valid: ', n_valid)
    else:
        loader_valid = None
    model = ConvAutoencoder(
            img_depth=imgs.shape[-3],
            img_height=imgs.shape[-2],
            img_width=imgs.shape[-1],
            channels_in=imgs.shape[-4],
            channels_latent=32,
            n_layers=3,
            n_latent=256,
            dropout_rate=None,
            lr=lr)
    return fit_model(
        model, loader, loader_valid,
        epochs, batch_size, lr,
        gpus, prefix)


def fit_mlp(
        x, y, x_valid, y_valid,
        epochs, batch_size, lr,
        gpus, prefix):

    dataset = GenericDataset(
            x.astype(np.float32),
            y.astype(np.float32))
    loader = DataLoader(dataset, batch_size=batch_size)
    print('n_train: ', len(dataset))
    if (x_valid is not None) and (y_valid is not None):
        dataset_valid = GenericDataset(
                x_valid.astype(np.float32),
                y_valid.astype(np.float32))
        n_valid = len(dataset_valid)
        loader_valid = DataLoader(
                dataset_valid, batch_size=n_valid)
        print('n_valid: ', n_valid)
    else:
        loader_valid = None

    n_nodes = [x.shape[-1]] + [256] * 4 + [y.shape[-1]]
    model = Mlp(
            n_nodes=n_nodes, activation=nn.GELU(),
            dropout_rate=None, lr=1e-3)
    return fit_model(
        model, loader, loader_valid,
        epochs, batch_size, lr,
        gpus, prefix)


def fit_cnn(
        imgs, targets,
        imgs_valid, targets_valid,
        epochs, batch_size, lr,
        gpus, prefix):
    dataset = GenericDataset(
            imgs.astype(np.float32),
            targets.astype(np.float32))
    loader = DataLoader(dataset, batch_size=batch_size)
    print('n_train: ', len(dataset))
    if imgs_valid is not None:
        dataset_valid = GenericDataset(
                imgs_valid.astype(np.float32),
                targets_valid.astype(np.float32))
        n_valid = len(dataset_valid)
        loader_valid = DataLoader(
                dataset_valid, batch_size=n_valid)
        print('n_valid: ', n_valid)
    else:
        loader_valid = None
    model = CNN(
            n_targets=targets.shape[-1],
            img_depth=imgs.shape[-3],
            img_height=imgs.shape[-2],
            img_width=imgs.shape[-1],
            channels_in=imgs.shape[-4],
            channels_latent=32,
            n_layers_conv=3,
            n_layers_mlp=4,
            n_latent=256,
            dropout_rate=None,
            lr=lr)
    return fit_model(
        model, loader, loader_valid,
        epochs, batch_size, lr,
        gpus, prefix)


def eval_cnn(model, imgs, targets, prefix):
    targets_pred = model(torch.tensor(imgs)).detach().numpy()
    corr = np.corrcoef(targets_pred.flatten(), targets.flatten())[0, 1]
    print('corr: ', corr)
    outfile = f'{prefix}corr.png'
    plt.plot(targets_pred.flatten(), targets.flatten(), 'o')
    plt.savefig(outfile, dpi=300)
    plt.close()


def eval_autoencoder(model, imgs):
    reconstructed = model(torch.tensor(imgs)).detach().numpy()
    mask = np.abs(imgs).max(0) != 0
    corr = np.corrcoef(
            imgs[:, mask].flatten(),
            reconstructed[:, mask].flatten())[0, 1]
    print('corr: ', corr)


def eval_mlp(model, features, targets):
    targets_pred = model(
            torch.tensor(features)).detach().numpy()
    corr = np.corrcoef(
            targets_pred.flatten(),
            targets.flatten())[0, 1]
    print('corr', corr)


def run_experiment_cnn(dataset, data):
    (imgs, targets), (imgs_valid, targets_valid) = data
    prefix = f'data/neuroimaging/{dataset}/cnn/'
    checkpoint_file = f'{prefix}model.ckpt'
    if os.path.exists(checkpoint_file):
        model = CNN.load_from_checkpoint(checkpoint_file)
        print(f'Model loaded from {checkpoint_file}')
    else:
        epochs = 50 if imgs_valid is None else 100
        model, trainer, history = fit_cnn(
                imgs=imgs, targets=targets,
                imgs_valid=imgs_valid, targets_valid=targets_valid,
                epochs=epochs, batch_size=512, lr=1e-3,
                gpus=1, prefix=prefix)
        trainer.save_checkpoint(checkpoint_file)
        print(f'Model saved to {checkpoint_file}')
        plot_history(history, f'{prefix}')
    model.eval()
    eval_cnn(model, imgs, targets, f'{prefix}train-')
    if imgs_valid is not None and targets_valid is not None:
        eval_cnn(model, imgs_valid, targets_valid, f'{prefix}valid-')
        features, features_valid = [
                model.encoder(torch.tensor(im)).detach().numpy()
                for im in [imgs, imgs_valid]]
        run_experiment_mlp(
                data=(
                    (features, targets),
                    (features_valid, targets_valid)),
                prefix=f'data/neuroimaging/{dataset}/cnn/mlp/')
    if imgs_valid is None:
        features = model.encoder(torch.tensor(imgs)).detach().numpy()
        save_data(
                features, targets,
                f'data/neuroimaging/{dataset}/')
        save_split_index(
                n_observations=features.shape[0],
                prop_test=0.1, n_splits=20,
                prefix=f'data/neuroimaging/{dataset}/')


def run_experiment_ae(dataset, data):
    (imgs, targets), (imgs_valid, targets_valid) = data
    prefix = f'data/neuroimaging/{dataset}/ae/'
    checkpoint_file = f'{prefix}model.ckpt'
    if os.path.exists(checkpoint_file):
        model = ConvAutoencoder.load_from_checkpoint(checkpoint_file)
        print(f'Model loaded from {checkpoint_file}')
    else:
        model, trainer, history = fit_autoencoder(
                imgs=imgs, imgs_valid=imgs_valid,
                epochs=100, batch_size=128, lr=1e-3,
                gpus=1, prefix=prefix)
        trainer.save_checkpoint(checkpoint_file)
        print(f'Model saved to {checkpoint_file}')
        plot_history(history, f'{prefix}')
    eval_autoencoder(model, imgs_valid)
    features, features_valid = [
            model.encoder(torch.tensor(im)).detach().numpy()
            for im in [imgs, imgs_valid]]
    run_experiment_mlp(
            data=(
                (features, targets),
                (features_valid, targets_valid)),
            prefix=f'data/neuroimaging/{dataset}/ae/mlp/')


def run_experiment_mlp(data, prefix):
    (features, targets), (features_valid, targets_valid) = data
    model, trainer, history = fit_mlp(
            x=features, y=targets,
            x_valid=features_valid, y_valid=targets_valid,
            epochs=100, batch_size=128, lr=1e-3,
            gpus=1, prefix=prefix)
    plot_history(history, prefix)
    eval_mlp(model, features, targets)
    if features_valid is not None and targets_valid is not None:
        eval_mlp(model, features_valid, targets_valid)


def run_experiment_pca(dataset):
    imgs, targets = get_data(dataset, linear_preprocess=True)
    model = PCA(n_components=75)
    model.fit(imgs)
    features = model.fit_transform(imgs)
    save_data(
            features, targets,
            f'data/neuroimaging/{dataset}/')
    save_split_index(
            n_observations=features.shape[0],
            prop_test=0.1, n_splits=20,
            prefix=f'data/neuroimaging/{dataset}/')


def main():
    args = get_args()
    set_seed(args.seed)
    dataset = args.dataset
    if args.run_pca:
        run_experiment_pca(dataset)
    else:
        imgs, targets = get_data(dataset)
        data = split_data(imgs, targets, prop_train=args.prop_train)
        if args.run_cnn:
            run_experiment_cnn(dataset=dataset, data=data)
        elif args.run_ae:
            run_experiment_ae(dataset=dataset, data=data)


if __name__ == '__main__':
    main()
