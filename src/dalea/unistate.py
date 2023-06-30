from time import time
import numpy as np
from scipy.stats import gaussian_kde, kendalltau
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import torch
from torch import nn
import matplotlib.pyplot as plt


def min_euclid_prob(model, x, n_samples):
    y_dist = model(x, n_samples=n_samples)
    n_classes = y_dist.shape[-1]
    eye = torch.eye(n_classes, device=x.device)
    distsq = ((y_dist[..., None, :] - eye)**2).mean(-1)
    y_pred = distsq.argmin(-1)[..., None]
    # y_pred = distsq.mean(0).argmin(-1)
    ran = torch.arange(n_classes, device=x.device)
    y_prob = (y_pred == ran).float().mean(0)
    return y_prob


def min_euclid_err(model, x, y, n_samples):
    y_prob = min_euclid_prob(model, x, n_samples)
    y_pred = y_prob.argmax(-1)
    err = (y_pred != y.argmax(-1))
    err = err.float().mean()
    return err


def nll_onehot(model, x, n_samples, n_classes):
    n_observations = x.shape[0]
    y_dummy = torch.tile(
            torch.eye(n_classes, device=x.device),
            (n_observations, 1, 1))
    x_expanded = x[..., np.newaxis, :]
    nll = nll_loss(
            model, x_expanded, y_dummy,
            n_samples=n_samples,
            reduction=False)
    nll = nll.sum(-1)
    return nll


def nll_onehot_err(model, x, y, n_samples, batch_size):
    nll = nll_onehot(model, x, n_samples, batch_size)
    y_pred = nll.argmin(-1)
    err = (y_pred != y.argmax(-1)).float().mean()
    return err


def rmse_loss(model, x, y, n_samples):
    y_pred = model(x, n_samples=n_samples).mean(0)
    return ((y_pred - y)**2).sum(-1).mean()**0.5


def plot_debug(*args, **kwargs):
    plot_debug_scatter(*args, **kwargs)
    plot_debug_histogram(*args, **kwargs)


def plot_debug_scatter(model, x, y, n_samples, prefix):
    nll = nll_loss(model, x, y, n_samples)
    yd = model(x, n_samples).cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    rmse = np.square(yd.mean(0) - y).mean()**0.5
    y = np.tile(y, (yd.shape[0], 1, 1))
    plt.figure(figsize=(8, 8))
    plt.plot(
            y.flatten()[::10], yd.flatten()[::10],
            'o', alpha=0.1)
    plt.axline([0, 0], [1, 1])
    plt.title(f'rmse: {rmse:.2f}, nll: {nll:.2f}')
    outfile = prefix + 'scatter.png'
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    print(outfile)
    plt.close()


def plot_debug_histogram(model, x, y, n_samples, prefix):
    plt.figure(figsize=(16, 8))
    for i, yu in enumerate(torch.unique(y)):
        isin = y[:, 0] == yu
        yd = model(x[isin], n_samples).cpu().detach().numpy()
        los = nll_loss(model, x[isin], y[isin], n_samples)
        plt.subplot(2, 5, 1+i)
        plt.hist(yd.flatten())
        plt.axvline(x=yu.item(), color='tab:orange')
        plt.title(f'class: {i:02d}, loss: {los:.2f}')
    outfile = prefix + 'histogram.png'
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    print(outfile)
    plt.close()


def loss_improvement_by_referral(loss, uncertainty):
    assert loss.ndim == 1
    assert uncertainty.ndim == 1
    assert loss.size == uncertainty.size
    loss = loss / loss.sum()
    order = uncertainty.argsort()[::-1]
    loss = loss[order]
    cumloss = np.cumsum(loss)
    return cumloss.mean()


def l2_improvement_by_referral(y_dist, y, alpha=0.05):
    mn = y_dist.mean(0)
    l2 = (mn - y)**2
    width = ci_width(y_dist, alpha/2, 1-alpha/2)
    return loss_improvement_by_referral(
            loss=l2.flatten(),
            uncertainty=width.flatten())


def rmse_certain(y_dist, y, lower, upper, alpha=0.05):
    assert 0 <= lower <= upper <= 1
    mn = y_dist.mean(0)
    l2 = (mn - y)**2
    width = ci_width(y_dist, alpha/2, 1-alpha/2)
    l2 = l2.flatten()
    width = width.flatten()
    order = width.argsort()
    n = l2.size
    start = min(int(n * lower), n-1)
    stop = max(int(n * upper), start+1)
    rmse = l2[order[start:stop]].mean()**0.5
    return rmse


def corr_uncertainty_accuracy(y_dist, y, alpha=0.05):
    assert y_dist.shape[1:] == y.shape
    mn = y_dist.mean(0)
    deviation = np.abs(mn - y)
    width = ci_width(y_dist, alpha/2, 1-alpha/2)
    return kendalltau(width.flatten(), deviation.flatten())[0]


def cdf_loss(y_dist, y, abs_diff=True):
    assert y_dist.shape[1:] == y.shape
    y_dist = np.sort(y_dist, axis=0)
    n_samples = y_dist.shape[0]
    loss = []
    for i in range(n_samples):
        empirical = (y < y_dist[i]).mean()
        theoretical = i / n_samples + 0.5
        deviation = empirical - theoretical
        if abs_diff:
            deviation = np.abs(deviation)
        loss.append(deviation)
    loss = np.mean(loss)
    return loss


def kde_gaussian(y_dist, y):
    assert y_dist.ndim == 2
    assert y.ndim == 1
    return np.mean([
        gaussian_kde(y_dist[:, i]).logpdf(y[i])[0]
        for i in range(y.shape[0])])


def kde_exponential(y_dist, y, cv=False):
    assert y_dist.ndim == 2
    assert y.ndim == 1
    if cv:
        kde = kde_exponential_single_cv
    else:
        kde = kde_exponential_single
    logprob = np.mean([kde(y_dist[:, i], y[i]) for i in range(y.shape[0])])
    return logprob


def kde_exponential_single_cv(y_dist, y):
    assert np.ndim(y_dist) == 1
    assert np.size(y) == 1
    params = {
            'bandwidth': np.logspace(-2, 2, 20),
            'kernel': ['exponential']}
    grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)
    grid.fit(y_dist.reshape(-1, 1))
    kde = grid.best_estimator_
    logprob = kde.score([[y]])
    return logprob


def kde_exponential_single(y_dist, y):
    assert np.ndim(y_dist) == 1
    assert np.size(y) == 1
    n = np.size(y_dist)
    bandwidth = n**(-0.2)  # Scott's Rule
    kde = KernelDensity(bandwidth=bandwidth, kernel='exponential')
    kde.fit(y_dist.reshape(n, 1))
    logprob = kde.score([[y]])
    return logprob


def ci_coverage_rate(y, y_dist, lower, upper):
    assert y_dist.shape[1:] == y.shape
    assert 0 <= lower < upper <= 1
    ci = np.quantile(y_dist, (lower, upper), axis=0)
    return ((y > ci[0]) * (y < ci[1])).mean()


def ci_width(y_dist, lower, upper):
    assert 0 <= lower < upper <= 1
    ci = np.quantile(y_dist, (lower, upper), axis=0)
    width = ci[1] - ci[0]
    return width


def required_interval_width(y_dist, y, alpha):
    increment = 1e-4
    theoretical = -increment
    empirical = 0.0
    while empirical <= alpha:
        theoretical_last = theoretical
        theoretical += increment
        empirical = 1 - ci_coverage_rate(
                y=y, y_dist=y_dist,
                lower=theoretical/2,
                upper=1-theoretical/2)
    if theoretical_last >= 0:
        width = ci_width(
                y_dist=y_dist,
                lower=theoretical_last/2,
                upper=1-theoretical_last/2)
        width = np.median(width)
    else:
        width = np.inf
    return width


class GaussianNoise(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        noise = torch.randn(*x.shape, device=x.device)
        scale = torch.exp(self.log_scale)
        return x + noise * scale


class DaleaUniStateEnsemble(nn.Module):

    def __init__(self, models, loss_logits=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if loss_logits is None:
            loss_logits = np.zeros(len(models))
        weights = np.exp(-loss_logits)
        self.weights = weights / weights.sum()

    def forward(self, x, n_samples):
        ns = np.random.multinomial(n_samples, self.weights)
        y = torch.cat(
                [m.forward(x, n) for m, n in zip(self.models, ns)], dim=0)
        return y

    def distribution(self, x, n_samples):
        ns = np.random.multinomial(n_samples, self.weights)
        out = [m.distribution(x, n) for m, n in zip(self.models, ns)]
        mean = torch.cat([mn for mn, __ in out], dim=0)
        std = torch.cat([sd for __, sd in out], dim=0)
        return mean, std


class DaleaUniState(nn.Module):

    def __init__(
            self, n_features, n_targets,
            n_layers, n_nodes, activation):
        super().__init__()

        layers = []
        for i in range(n_layers+1):
            if i == 0:
                dim_inp = n_features
                noise_postact = nn.Identity()
            else:
                dim_inp = n_nodes
                noise_postact = GaussianNoise(dim_inp)
            if i == n_layers:
                dim_out = n_targets
                activation_layer = nn.Identity()
            else:
                dim_out = n_nodes
                activation_layer = activation
            linear = nn.Linear(dim_inp, dim_out)
            noise_preact = GaussianNoise(dim_out)
            lyr = nn.Sequential()
            lyr.add_module('noise_postact', noise_postact)
            lyr.add_module('linear', linear)
            lyr.add_module('noise_preact', noise_preact)
            lyr.add_module('activation', activation_layer)
            layers.append(lyr)
        self.net = nn.Sequential(*layers)

        self.x_mean = nn.Parameter(
                torch.zeros(n_features), requires_grad=False)
        self.x_std = nn.Parameter(
                torch.ones(n_features), requires_grad=False)
        self.y_mean = nn.Parameter(
                torch.zeros(n_targets), requires_grad=False)
        self.y_std = nn.Parameter(
                torch.ones(n_targets), requires_grad=False)

    def set_norm_info(self, x_mean, x_std, y_mean, y_std):
        self.x_mean[:] = x_mean
        self.x_std[:] = x_std
        self.y_mean[:] = y_mean
        self.y_std[:] = y_std

    def forward(self, x, n_samples):
        x = (x - self.x_mean) / self.x_std
        x = torch.tile(x, (n_samples,) + (1,) * x.ndim)
        x = self.net(x)
        return x * self.y_std + self.y_mean

    def distribution(self, x, n_samples):
        ndim = x.ndim
        x = (x - self.x_mean) / self.x_std
        x = torch.tile(x, (n_samples,) + (1,) * ndim)
        x = self.net[:-1](x)

        mean = self.net[-1].linear(x)
        variance = torch.sum(
                torch.exp(2 * self.net[-1].noise_postact.log_scale)
                * self.net[-1].linear.weight**2,
                dim=-1)
        variance += torch.exp(2 * self.net[-1].noise_preact.log_scale)
        std = variance**0.5

        # x = self.net[-1].noise_postact(x)
        # x = self.net[-1].linear(x)
        # mean = x
        # std = torch.exp(self.net[-1].noise_preact.log_scale)

        std = torch.tile(std, (n_samples,) + (1,) * ndim)
        mean = mean * self.y_std + self.y_mean
        std = std * self.y_std
        return mean, std

    def weight_regularization_loss(self, l2, reduction):
        assert reduction in ['mean', 'sum']
        loss = 0.0
        for layer in self.net:
            loss_onelayer = l2 * ((layer.linear.weight)**2)
            if reduction == 'mean':
                loss_onelayer = loss_onelayer.mean()
            elif reduction == 'sum':
                loss_onelayer = loss_onelayer.sum()
            loss += loss_onelayer
        return loss

    def regularization_loss(self, l2, reduction='sum'):
        return self.weight_regularization_loss(
                l2=l2, reduction=reduction)


def smoothen(x, bandwidth_factor=0, bandwidth_min=10):
    bandwidth = max(
            bandwidth_min,
            int(len(x) * bandwidth_factor))
    start = max(0, len(x) - bandwidth)
    return np.quantile(x[start:], 0.9)


def nll_loss(model, x, y, n_samples, reduction=True):
    mean, std = model.distribution(x, n_samples=n_samples)
    n = mean.shape[0]
    logprob = (
            (-0.5) * torch.log(2 * torch.pi * std**2)
            + (-0.5) * (y - mean)**2 / std**2)
    logprob = torch.logsumexp(logprob, dim=0) - np.log(n)
    nlp = (-1) * logprob
    if reduction:
        nlp = nlp.mean(-1).mean(-1)
    return nlp


def metrics(model, x, y, n_samples=1000):
    tot_loss, regu_loss, y_dist = quantile_loss(
            model, x, y, regularization=None,
            n_samples=n_samples, return_extras=True)
    y_dist = y_dist.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    cover95 = ci_coverage_rate(y, y_dist, 0.025, 0.975)
    cover50 = ci_coverage_rate(y, y_dist, 0.25, 0.75)
    rmse = ((y_dist.mean(0) - y)**2).mean()**0.5
    msg = (
            f'rmse: {rmse:.3f}, qtloss: {tot_loss:.3f} '
            f'cover50: {cover50:.3f}, cover95: {cover95:.3f}')
    return msg


def quantile_loss(
        model, x, y,
        regularization=None,
        n_samples=1000,
        reduction='mean',
        return_extras=False):
    y_dist = model(x, n_samples=n_samples)
    rank = y_dist.argsort(0).argsort(0)
    qt = (rank + 0.5) / n_samples
    err = y - y_dist
    loss = torch.max((qt - 1) * err, qt * err)

    if reduction == 'mean':
        loss_tot = loss.mean()
    elif reduction == 'sum':
        loss_tot = loss.mean(0).sum()

    if regularization is None:
        regularization_loss = 0.0
    else:
        assert regularization >= 0
        regularization_loss = model.regularization_loss(l2=regularization)
    loss_tot += regularization_loss

    if return_extras:
        out = (loss_tot, regularization_loss, y_dist)
    else:
        out = loss_tot
    return out


def eval_loss(
        model, x, y, criterion, train,
        optimizer=None, batch_size=None,
        criterion_kwargs={}, reduce=True):
    n_observations = x.shape[0]
    if batch_size is None:
        n_batches = 1
    else:
        n_batches = (n_observations + batch_size - 1) // batch_size
        if reduce:
            idx_all = torch.randperm(n_observations, device=x.device)
        else:
            idx_all = torch.arange(n_observations, device=x.device)
    if reduce:
        loss_tot = 0.0
    else:
        loss_tot = []
    if train:
        model.train()
    else:
        model.eval()
    for i in range(n_batches):
        if batch_size is None:
            xi, yi = x, y
            weight = 1.0
        else:
            i_start = i * batch_size
            i_stop = min((i+1) * batch_size, n_observations)
            idx = idx_all[i_start:i_stop]
            xi, yi = x[idx], y[idx]
            weight = len(idx) / n_observations
        if train:
            optimizer.zero_grad()
            loss = criterion(model, xi, yi, **criterion_kwargs)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = criterion(model, xi, yi, **criterion_kwargs)
        if loss.nelement() == 1:
            loss = loss.item()
        if reduce:
            loss_tot = loss_tot + loss * weight
        else:
            loss_tot.append(loss * weight)
    return loss_tot


def train_model(
        model, x, y, criterion,
        learning_rate, epochs, batch_size,
        x_valid=None, y_valid=None, patience=None,
        criterion_stopping=None,
        criterion_kwargs={},
        criterion_stopping_kwargs={},
        verbose=True, metrics=metrics):

    validate = (x_valid is not None) and (y_valid is not None)
    optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate)
    history = {'train': []}
    # history['rmse_train'] = []
    if validate:
        history['valid'] = []
        history['smooth'] = []
        loss_best = np.inf

    loss_train = eval_loss(
            model=model, x=x, y=y, train=False,
            criterion=criterion_stopping,
            criterion_kwargs=criterion_stopping_kwargs,
            batch_size=batch_size)
    history['train'].append(loss_train)
    # rmse_train = eval_loss(
    #         model, x=x, y=y, train=False,
    #         criterion=rmse_loss,
    #         criterion_kwargs=criterion_stopping_kwargs,
    #         batch_size=batch_size)
    # history['rmse_train'].append(rmse_train)
    time_start = time()
    breaknow = False
    for epoch in range(epochs):

        if validate:
            loss_valid = eval_loss(
                    model=model, x=x_valid, y=y_valid, train=False,
                    criterion=criterion_stopping,
                    criterion_kwargs=criterion_stopping_kwargs,
                    batch_size=batch_size)
            history['valid'].append(loss_valid)
            loss_smooth = smoothen(history['valid'])
            history['smooth'].append(loss_smooth)
            if loss_smooth < loss_best:
                epoch_best = epoch
                loss_best = loss_smooth
                param_best = model.state_dict()
            since_best = epoch - epoch_best
            if (patience is not None) and (since_best >= patience):
                if loss_smooth > loss_best:
                    breaknow = True

        if (verbose and epoch % 10 == 0) or breaknow:
            runtime = time() - time_start
            time_start = time()
            msg = (
                    f'epoch {epoch:04d}, '
                    f'time {runtime:07.3f}, '
                    f'loss_train {loss_train:07.3f}'
                    )
            if validate:
                msg += (
                        f', loss_valid {loss_valid:07.3f}'
                        f', loss_smooth {loss_smooth:07.3f}'
                        f', loss_best {loss_best:07.3f}'
                        f', since_best {since_best:04d}'
                        )
            print(msg)
            # if epoch % 100 == 0:
            #     plot_debug(
            #             model, x, y,
            #             n_samples=criterion_kwargs['n_samples'],
            #             prefix=f'tmp/debug/{epoch:06d}-')
            # if metrics is not None:
            #     mets_train = eval_loss(
            #         model, x, y, metrics)
            #     print('metrics train:', mets_train)
            # if validate and metrics is not None:
            #     mets_valid = eval_loss(
            #         model, x_valid, y_valid, metrics)
            #     print('metrics valid:', mets_valid)

        if breaknow:
            break

        eval_loss(
                model=model, x=x, y=y,
                train=True, optimizer=optimizer,
                criterion=criterion,
                criterion_kwargs=criterion_kwargs,
                batch_size=batch_size)
        loss_train = eval_loss(
                model=model, x=x, y=y, train=False,
                criterion=criterion_stopping,
                criterion_kwargs=criterion_stopping_kwargs,
                batch_size=batch_size)
        history['train'].append(loss_train)
        # rmse_train = eval_loss(
        #         model, x=x, y=y, train=False,
        #         criterion=rmse_loss,
        #         criterion_kwargs=criterion_stopping_kwargs,
        #         batch_size=batch_size)
        # history['rmse_train'].append(rmse_train)

    if validate:
        model.load_state_dict(param_best)
        history['epoch_best'] = epoch_best
    print('Finished Training')
    return history
