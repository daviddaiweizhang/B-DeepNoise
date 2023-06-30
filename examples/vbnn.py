import sys
import numpy as np
import torch
from torch import nn
import pyro
import tyxe
from torch.utils.data import DataLoader
from dnne_experiment import GenericDataset
from pyro.distributions import Normal
from dnne.dnne import Mlp
from dnne_experiment import get_data
from utils.evaluate import gaussian_loss


def gen_data(n=1000):
    x = np.random.randn(n, 1)
    y = x * 3 - 1
    noise = np.random.randn(*y.shape) * 0.1
    y = y + noise
    return x.astype(np.float32), y.astype(np.float32)


def create_model(n_nodes, activation, n_observations):
    net = Mlp(n_nodes=n_nodes, activation=activation)
    prior = tyxe.priors.IIDPrior(Normal(0, 1))
    likelihood = tyxe.likelihoods.HomoskedasticGaussian(
            dataset_size=n_observations, scale=0.001)
    inference = tyxe.guides.AutoNormal
    model = tyxe.VariationalBNN(net, prior, likelihood, inference)
    return model


def get_model(x, y, n_nodes, batch_size, epochs, lr):
    dataset = GenericDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model = create_model(
            n_nodes=n_nodes,
            activation=nn.ReLU(),
            n_observations=len(dataset))
    optim = pyro.optim.Adam({'lr': lr})
    model.fit(data_loader=data_loader, optim=optim, num_epochs=epochs)
    y_dist = model.predict(
            torch.tensor(x), num_predictions=100, aggregate=False)
    y_dist = y_dist.detach().numpy()
    noise_scale = np.square(y_dist - y).mean()**0.5
    return model, noise_scale


def eval_model(model, x, y, n_samples):
    y_mean, y_std = model.predict(
            torch.tensor(x),
            num_predictions=n_samples,
            aggregate=True)
    y_mean = y_mean.detach().numpy()
    y_std = y_std.detach().numpy()
    corr = np.corrcoef(y_mean.flatten(), y.flatten())[0, 1]
    rmse = np.square(y_mean - y).mean()**0.5
    nll = gaussian_loss(y_mean[np.newaxis], y_std[np.newaxis], y)
    # mse, ll = model.evaluate(
    #         torch.tensor(x), y=torch.tensor(y),
    #         num_predictions=n_samples,
    #         reduction='mean')
    print(corr, rmse, nll)


def main():

    dataset = sys.argv[1]  # e.g. yacht, energy
    split = 0
    n_layers = 4
    n_nodes = 50
    lr = 1e-3
    batch_size = 100
    epochs = 2000
    n_samples = 20

    data = get_data(dataset=dataset, split=split)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    # x, y = gen_data(n_observations=1000)

    n_features = x_train.shape[-1]
    n_targets = y_train.shape[-1]
    n_nodes = [n_features] + [n_nodes] * n_layers + [n_targets]
    model = get_model(
            x=x_train, y=y_train,
            n_nodes=n_nodes, lr=lr, batch_size=batch_size, epochs=epochs)
    eval_model(model, x=x_test, y=y_test, n_samples=n_samples)


if __name__ == '__main__':
    main()
