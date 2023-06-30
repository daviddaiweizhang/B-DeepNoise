import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Mlp(pl.LightningModule):

    def __init__(self, n_nodes, activation=None, dropout_rate=None, lr=1e-3):
        super().__init__()

        if activation is None:
            activation = nn.LeakyReLU(0.1)

        layers = []
        n_layers = len(n_nodes) - 2
        for i in range(n_layers+1):
            if dropout_rate is None or i == 0:
                dropout_postact = nn.Identity()
            else:
                dropout_postact = nn.Dropout(dropout_rate)
            if dropout_rate is None or i == n_layers:
                dropout_preact = nn.Identity()
            else:
                dropout_preact = nn.Dropout(dropout_rate)
            if i == n_layers:
                act = nn.Identity()
            else:
                act = activation
            dim_inp, dim_out = n_nodes[i], n_nodes[i+1]
            linear = nn.Linear(dim_inp, dim_out)
            layers.append(dropout_postact)
            layers.append(linear)
            layers.append(dropout_preact)
            layers.append(act)
        self.net = nn.Sequential(*layers)

        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log('loss_train', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log('loss_valid', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MlpGaussian(nn.Module):

    def __init__(self, n_nodes, activation, dropout_rate=None):
        super().__init__()
        self.net_mean = Mlp(
                n_nodes=n_nodes, activation=activation,
                dropout_rate=dropout_rate)
        self.net_std = nn.Sequential(
                Mlp(
                    n_nodes=n_nodes, activation=activation,
                    dropout_rate=dropout_rate),
                nn.Softplus())

        n_features = n_nodes[0]
        n_targets = n_nodes[-1]
        self.x_mean = nn.Parameter(
                torch.zeros(n_features), requires_grad=False)
        self.x_std = nn.Parameter(
                torch.ones(n_features), requires_grad=False)
        self.y_mean = nn.Parameter(
                torch.zeros(n_targets), requires_grad=False)
        self.y_std = nn.Parameter(
                torch.ones(n_targets), requires_grad=False)

    def distribution(self, x):
        x = (x - self.x_mean) / self.x_std
        mean = self.net_mean(x)
        std = self.net_std(x)
        mean = mean * self.y_std + self.y_mean
        std = std * self.y_std
        return mean, std

    def forward(self, x, n_samples):
        mean, std = self.distribution(x)
        noise_shape = (n_samples,) + mean.shape
        noise = torch.randn(*noise_shape)
        return noise * std + mean

    def set_norm_info(self, x_mean, x_std, y_mean, y_std):
        self.x_mean[:] = x_mean
        self.x_std[:] = x_std
        self.y_mean[:] = y_mean
        self.y_std[:] = y_std


class MlpGaussianEnsemble(pl.LightningModule):

    def __init__(
            self, n_nodes, activation, n_nets,
            dropout_rate=None, lr=1e-3):
        super().__init__()
        self.nets = nn.ModuleList([
            MlpGaussian(
                n_nodes=n_nodes,
                activation=activation,
                dropout_rate=dropout_rate)
            for __ in range(n_nets)])

        self.lr = lr
        self.save_hyperparameters()

    def distribution(self, x, *args, **kwargs):
        out = [ne.distribution(x) for ne in self.nets]
        mean = torch.stack([ou[0] for ou in out], dim=0)
        std = torch.stack([ou[1] for ou in out], dim=0)
        return mean, std

    def forward(self, x, n_samples):
        n_nets = len(self.nets)
        weights = [1.0 / n_nets] * n_nets
        ns = np.random.multinomial(n_samples, weights)
        y = torch.cat(
                [ne.forward(x, n) for ne, n in zip(self.nets, ns)], dim=0)
        return y

    def set_norm_info(self, x_mean, x_std, y_mean, y_std):
        for ne in self.nets:
            ne.set_norm_info(x_mean, x_std, y_mean, y_std)

    def training_step(self, batch, batch_idx):
        x, y = batch
        mean, std = self.distribution(x)
        loss = 0.5 * (y - mean)**2 / std**2 + torch.log(std)
        loss = loss.mean()
        self.log('loss_train', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
