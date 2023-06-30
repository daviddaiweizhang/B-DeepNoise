import numpy as np
from scipy import stats
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


from dalea.unistate import (
        eval_loss, nll_loss, ci_coverage_rate, ci_width,
        rmse_certain, required_interval_width,
        corr_uncertainty_accuracy, cdf_loss,
        l2_improvement_by_referral,
        nll_onehot, min_euclid_prob)
from utils.utils import save_qq, gaussian_loss
from utils.visual import plot_quantiles, plot_qq


def fpr(y_true, y_score, tpr_threshold):
    fpr, tpr, __ = roc_curve(y_true, y_score)
    return fpr[tpr >= tpr_threshold].min()


def violin_plot(x, labels):
    df = pd.DataFrame(x)
    stub = 'y'
    df.columns = [f'{stub}{e}' for e in labels]
    df['id'] = df.index
    df = pd.wide_to_long(df, stub, i='id', j='x')
    df = df.reset_index()
    sns.violinplot(data=df, x='x', y='y', color='lightgray')


def rhat(x):
    # return float(tfp.mcmc.potential_scale_reduction(
    #     x, split_chains=False))
    pass


def gof(y_dist, y):
    y_mean = y_dist.mean(0)
    y_std = y_dist.std(0)
    return (
            (y - y_mean)**2 / y_std**2
            + 2 * np.log(y_std)).mean((-1, -2))


def cinterval(x, falserate):
    return np.quantile(x, [falserate/2, 1-falserate/2], 0)


def flatten_dist(x):
    return x.reshape(-1, *x.shape[-2:])


def arr_to_str(x, sep=' '):
    assert np.ndim(x) <= 1
    return sep.join(map(str, x))


def stratify(x, y, n_strata, reduction_func=np.median):
    assert x.shape == y.shape
    strata_border = np.quantile(
            x, np.linspace(0, 1, n_strata+1))
    strata_border[0] -= 0.1
    strata_border[-1] += 0.1
    x_strata = []
    y_strata = []
    for i in range(n_strata):
        idx_stratum = np.logical_and(
            x > strata_border[i],
            x <= strata_border[i+1])
        x_stat = reduction_func(x[idx_stratum])
        y_stat = reduction_func(y[idx_stratum])
        x_strata.append(x_stat)
        y_strata.append(y_stat)
    x_strata = np.stack(x_strata)
    y_strata = np.stack(y_strata)
    return y_strata, x_strata


def evaluate_samples(
        y_dist_train=None, y_dist_test=None, y_train=None,
        y_mean_truth_test=None, y_test=None,
        n_strata=5, falserate=0.1, outfile='evaluation.txt'):
    """ Evaluate posterior samples.
        Args:
            y_dist_train: Predictive samples of y
                in the training set.
                Has shape
                (n_realizations, n_states, n_chains,
                n_observations_train, n_targets)
            y_dist_test: Predictive samples of y
                in the testing set.
                Has shape
                (n_realizations, n_states, n_chains,
                n_observations_train, n_targets)
            y_train: Observed values of y in the training set.
                Has shape (n_observations_train, n_targets).
            y_mean_truth_test: True mean values of y
                in the testing set.
                If missing, use `y_test` instead.
                Has shape (n_observations_test, n_targets).
            y_test: Observed values of y in the testing set.
                Required if `y_mean_truth` is not provided
                Has shape (n_observations_test, n_targets).
            n_strata: Number of strata to use to
                stratify the deviation from mean
                by the radius of CI
            falserate: Controls the radius of CI.
                Example: 0.95 CI --> falserate == 0.025
            outfile: File for saving the evaluation results.
    """

    # check convergence
    if y_dist_train is not None and y_train is not None:
        trace_gof = gof(y_dist_train, y_train)
        rhat_gof = rhat(trace_gof)
    else:
        rhat_gof = None

    if y_mean_truth_test is not None:
        # ground truth is known
        y_ref = y_mean_truth_test
        y_dist = flatten_dist(y_dist_test.mean(0))
    else:
        # ground truth is unknown
        y_ref = y_test
        y_dist = flatten_dist(y_dist_test)

    y_mean = y_dist.mean(-3)
    devia = np.abs(y_mean - y_ref)
    yfit_corr, yfit_pval = stats.kendalltau(
            y_mean.flatten(), y_ref.flatten())
    # get MSE
    mse = (devia**2).mean()
    # get correlation between CI radius and deviation
    ci = cinterval(y_dist, falserate)
    cirad = (ci[1] - ci[0]) / 2
    cilen = (cirad * 2).mean()
    cide_corr, cide_pval = stats.kendalltau(
            cirad.flatten(), devia.flatten())
    # stratify error by CI radius
    devia_strata, cirad_strata, = stratify(
            x=cirad, y=devia,
            n_strata=n_strata, reduction_func=np.mean)
    # stratify error quantile by CI radius
    n_observations = devia.shape[-2]
    deviq = devia.argsort(-2).argsort(-2) / n_observations
    deviq_strata, __, = stratify(
            x=cirad, y=deviq,
            n_strata=n_strata, reduction_func=np.median)
    # stratify coverage by CI radius
    is_covered = np.logical_and(y_ref > ci[0], y_ref < ci[1])
    cover_rate = is_covered.mean()
    cover_strata, __ = stratify(
            x=cirad, y=is_covered,
            n_strata=n_strata, reduction_func=np.mean)

    with open(outfile, 'w') as file:
        print('rhat:', rhat_gof, file=file)
        print('mse:', mse, file=file)
        print('yfit_corr:', yfit_corr, file=file)
        print('yfit_pval:', yfit_pval, file=file)
        print('cover_rate:', cover_rate, file=file)
        print('cilen:', cilen, file=file)
        print('cide_corr:', cide_corr, file=file)
        print('cide_pval:', cide_pval, file=file)
        print(
                'cirad_strata:',
                arr_to_str(cirad_strata), file=file)
        print(
                'devia_strata:',
                arr_to_str(devia_strata), file=file)
        print(
                'deviq_strata:',
                arr_to_str(deviq_strata), file=file)
        print(
                'cover_strata:',
                arr_to_str(cover_strata), file=file)
    print('Evaluation saved to', outfile)


def evaluate_real(y, y_dist, outfile=None, batch_size=None, **kwargs):
    """ Evaluate predictive samples and save evaluation to file.
        Args:
            y: Observed values of y.
                Has shape (n_observations, n_targets)
            y_dist: Predictive samples of y.
                Has shape (n_realizations, n_observations, n_targets)
            outfile: File for saving the evaluation.
    """

    assert y_dist.ndim == 3
    assert y.shape == y_dist.shape[1:]

    metrics = {}
    if all([e in kwargs.keys() for e in ['model', 'x', 'n_samples']]):
        # import torch
        # metrics['nll'] = nll_loss(
        #         kwargs['model'],
        #         torch.tensor(kwargs['x'], dtype=torch.float),
        #         torch.tensor(y, dtype=torch.float),
        #         kwargs['n_samples']).item()
        metrics['nll'] = eval_loss(
                model=kwargs['model'],
                x=torch.tensor(kwargs['x'], dtype=torch.float),
                y=torch.tensor(y, dtype=torch.float),
                train=False,
                criterion=nll_loss,
                criterion_kwargs={'n_samples': kwargs['n_samples']},
                batch_size=batch_size)
    elif all([e in kwargs.keys() for e in ['mean', 'std']]):
        metrics['nll'] = gaussian_loss(kwargs['mean'], kwargs['std'], y)

    metrics['rmse'] = np.sqrt(np.square(y_dist.mean(0) - y).mean())
    metrics['ci95_coverage'] = ci_coverage_rate(y, y_dist, 0.025, 0.975)
    metrics['ri95_width'] = required_interval_width(y_dist, y, 0.05)
    metrics['rmse_certain_0'] = rmse_certain(y_dist, y, 0.0, 0.2)
    metrics['rmse_certain_1'] = rmse_certain(y_dist, y, 0.2, 0.4)
    metrics['rmse_certain_2'] = rmse_certain(y_dist, y, 0.4, 0.6)
    metrics['rmse_certain_3'] = rmse_certain(y_dist, y, 0.6, 0.8)
    metrics['rmse_certain_4'] = rmse_certain(y_dist, y, 0.8, 1.0)
    metrics['rmse_certain50'] = rmse_certain(y_dist, y, 0.0, 0.5)
    metrics['cua'] = corr_uncertainty_accuracy(y_dist, y)
    metrics['correlation'] = stats.kendalltau(y_dist.mean(0), y)[0]
    metrics['lir'] = l2_improvement_by_referral(y_dist, y)
    metrics['cdfl'] = cdf_loss(y_dist, y)
    metrics['ci95_width'] = np.median(ci_width(y_dist, 0.025, 0.975))
    if outfile is not None:
        with open(outfile, 'w') as file:
            for key, value in metrics.items():
                print(f'{key}: {value}', file=file)
        print(outfile)
    return metrics


def ks_dist(x, y):
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[-1] == y.shape[-1]
    ks = np.array([stats.ks_2samp(xi, yi)[0] for xi, yi in zip(x.T, y.T)])
    return ks


def qq_dist_single(x, y, n=100):
    assert x.ndim == 1
    assert y.ndim == 1
    q = (np.arange(n) + 0.5) / n
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    dist = np.abs(xq - yq).mean()
    return dist


def qq_dist(x, y):
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[-1] == y.shape[-1]
    dist = np.array([qq_dist_single(xi, yi) for xi, yi in zip(x.T, y.T)])
    return dist


def evaluate_synthetic(
        x_quantiles, y_quantiles,
        x_observed, y_observed,
        y_dist, prefix,
        make_quantile_plots=False,
        make_qq_plots=False):

    y_dist_2d = y_dist.reshape(
                y_dist.shape[0],
                np.prod(y_dist.shape[1:]))
    y_quantiles_2d = y_quantiles.reshape(
        y_quantiles.shape[0],
        np.prod(y_quantiles.shape[1:]))

    ks = ks_dist(y_dist_2d, y_quantiles_2d).mean()
    qq = qq_dist(y_dist_2d, y_quantiles_2d).mean()

    outfile = prefix + 'eval-synthetic.txt'
    with open(outfile, 'w') as file:
        print('ks', ks, file=file)
        print('qq', qq, file=file)
    print(outfile)

    if make_quantile_plots:
        y_dist = np.sort(y_dist, 0)
        plot_quantiles(
                x_quantiles=x_quantiles[..., 0],
                y_quantiles=y_dist[..., 0],
                x_observed=x_observed,
                y_observed=y_observed,
                quantiles=np.linspace(0, 1, 20+2)[1:-1],
                outfile=f'{prefix}quantiles.png')

    if make_qq_plots:
        for x_point in [0.1, 0.5]:
            label = f'{x_point:.2f}'
            idx = np.abs(x_quantiles[..., 0] - x_point).argmin()
            plot_qq(
                    expected=y_quantiles[..., idx, 0],
                    observed=y_dist[..., idx, 0],
                    outfile=f'{prefix}qq-{label}.png')
            save_qq(
                    expected=y_quantiles[..., idx, 0],
                    observed=y_dist[..., idx, 0],
                    label=label,
                    outfile=f'{prefix}qq-{label}.tsv')


def print_eval_by_state(mean, std, y):

    n_show = 10

    n_states = mean.shape[1]
    state_stride = (n_states + n_show - 1) // n_show

    rmse_chainstates = np.square(
            mean.mean(0) - y).mean((-2, -1))**0.5
    print(f'rmse_chainstates (per {state_stride} states)')
    print(rmse_chainstates[::state_stride])

    rmse_states = np.square(
            mean.mean((0, 2)) - y).mean((-2, -1))**0.5
    print(f'rmse_chainstates (per {state_stride} states)')
    print(rmse_states[::state_stride])

    rmse_chains = np.square(
            mean.mean((0, 1)) - y).mean((-2, -1))**0.5
    print('rmse_chains')
    print(rmse_chains)

    rmse = np.square(
            mean.mean((0, 1, 2)) - y).mean((-2, -1))**0.5
    print('rmse')
    print(rmse)

    nll_chainstates = gaussian_loss(mean, std, y)
    print(f'nll_chainstates (per {state_stride} states)')
    print(nll_chainstates[::state_stride])

    mean_tp = mean.swapaxes(1, 2)
    std_tp = std.swapaxes(1, 2)
    nll_states = gaussian_loss(
            mean_tp.reshape((-1,) + mean_tp.shape[-3:]),
            std_tp.reshape((-1,) + std_tp.shape[-3:]),
            y)
    print('nll_states')
    print(nll_states[::state_stride])

    nll_chains = gaussian_loss(
            mean.reshape((-1,) + mean.shape[-3:]),
            std.reshape((-1,) + std.shape[-3:]),
            y)
    print('nll_chains')
    print(nll_chains)

    nll = gaussian_loss(
            mean.reshape((-1,) + mean.shape[-2:]),
            std.reshape((-1,) + std.shape[-2:]),
            y)
    print('nll')
    print(nll)


def evaluate_influence(model, x, method, outfile, n_samples=100):
    assert x.ndim == 2
    model.eval()
    model.zero_grad()
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y_dist = model(x, n_samples=n_samples)
    if method == 'mean':
        y_stat = y_dist.mean(0)
    elif method == 'variance':
        y_stat = y_dist.var(0)
    else:
        raise ValueError('Method not recognized. ')
    y_stat.sum().backward()
    influence = ((x.grad**2).mean(0)**0.5) / x.std(0)
    influence = influence.detach().numpy()
    np.savetxt(outfile, influence.reshape(-1, 1), fmt='%.4f')
    print(outfile)


def evaluate_individual(model, x, y, prefix, n_samples=1000):
    assert x.ndim == 2
    assert y.ndim == 2
    model.eval()

    order = y[:, 0].argsort()
    y = y[order]
    x = x[order]
    y_dist = model(
            torch.tensor(x, dtype=torch.float32),
            n_samples=n_samples)
    y_dist = y_dist.detach().numpy()

    n_observations = x.shape[0]
    qt = np.linspace(0, 1, 20+1)[1:-1]
    i = np.round(n_observations * qt).astype(int)

    # title = (
    #         'Observed values and '
    #         'estimated predictive distributions of g-scores ')
    show_y = True
    margin_kwargs = dict(
            left=0.10, right=0.97, bottom=0.12, top=0.97)
    figsize = (16, 8)
    # if x.shape[-1] <= 11:
    #     # title += ' (imaging predictors excluded)'
    #     # show_y = False
    #     # margin_kwargs = dict(
    #     #         left=0.05, right=0.97, bottom=0.12, top=0.97)
    #     # figsize = (14.5, 8)
    # else:
    #     # title += ' (imaging predictors included)'

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    violin_plot(
            x=y_dist[:, i, 0],
            labels=np.round(qt * 100).astype(int))

    plt.plot(
            np.arange(qt.size), y[i, 0], 'X', label='g-score',
            markerfacecolor='white',
            markeredgecolor='black',
            markeredgewidth=2,
            markersize=15)

    # plt.title(title, fontsize=28)
    plt.xlabel('g-score quantile (observed)', fontsize=25)
    plt.ylabel('g-score value (observed and predicted)', fontsize=25)
    plt.ylim(-1.5, 1.5)

    # set tick font size
    ax.tick_params(axis='both', which='major', labelsize=25)
    # Turn off y axis tick and labels
    ax.axes.yaxis.set_visible(show_y)
    # set margin
    plt.subplots_adjust(**margin_kwargs)

    outfile = prefix + 'violin.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(outfile)


def error_stratified_by_quantile(error, uncertainty, n_strata):
    n_observations = error.size
    n_observations -= n_observations % n_strata
    error, uncertainty = error[:n_observations], uncertainty[:n_observations]
    order = uncertainty.argsort()
    error = error[order]
    uncertainty = uncertainty[order]
    error_strata = error.reshape(n_strata, -1).mean(-1)
    uncertainty_strata = uncertainty.reshape(n_strata, -1).mean(-1)
    return error_strata, uncertainty_strata


def error_stratified_by_prob(error, prob, n_strata):
    assert np.min(prob) >= 0
    assert np.max(prob) <= 1
    bins = np.floor(prob * n_strata).astype(int)
    bins = np.clip(bins, 0, n_strata-1)
    err_strata = np.array([
        error[bins == b].mean()
        if (bins == b).any()
        else np.nan
        for b in range(n_strata)])
    weight = (bins[..., np.newaxis] == np.arange(n_strata)).mean(0)
    prob_strata = (np.arange(n_strata) + 0.5) / n_strata
    return err_strata, prob_strata, weight


def plot_binary(grid, prob):
    im = plt.pcolormesh(
            grid[..., 0], grid[..., 1], prob,
            shading='gouraud', cmap='RdBu_r', vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)


def plot_multiclass(grid, prob):
    classes = prob.argmax(-1)
    plt.pcolormesh(
            grid[..., 0], grid[..., 1], classes,
            shading='nearest', cmap='tab10', vmin=0, vmax=10, alpha=0.7)


def plot_categorical(y_label, x, grid, prob, outfile):
    # TODO: figure out why figsize is not working
    n_classes = y_label.max() + 1
    if n_classes <= 2:
        plt.figure(figsize=(8, 8))
        plot_binary(grid=grid, prob=prob[..., 1])
        cmap = ListedColormap(["#88CCFF", "#FF8844"])
        y_label = y_label * 2.0 - 1.0
    else:
        plt.figure(figsize=(12, 8))
        plot_multiclass(grid=grid, prob=prob)
        cmap = plt.get_cmap('tab10')
    plt.scatter(
            x[:, 0], x[:, 1], c=cmap(y_label),
            edgecolors='black')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(outfile)


def plot_ood(is_in, score_in, outfile):
    plt.hist(score_in[~is_in], label='out', bins=100, alpha=0.5)
    plt.hist(score_in[is_in], label='in', bins=100, alpha=0.5)
    plt.xlabel('confidence')
    plt.title(
            f'fpr80: {fpr(is_in, score_in, 0.80):.3f}, '
            f'fpr95: {fpr(is_in, score_in, 0.95):.3f}, '
            f'rocauc: {roc_auc_score(is_in, score_in):.3f}, '
            f'aupr: {average_precision_score(is_in, score_in):.3f}')
    plt.legend()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(outfile)


def get_overconfidence(observed, expected, weight):
    isin = np.isfinite(np.stack([observed, expected])).all(0)
    obs, exp, wt = observed[isin], expected[isin], weight[isin]
    over = np.clip(obs - exp, 0, None)
    over_avg = (over * wt).sum()
    return over_avg


def evaluate_categorical(
        y, y_dist, model, x,
        n_samples, batch_size, prefix):
    assert y_dist.ndim == 3
    assert y.shape == y_dist.shape[1:]

    model.cuda()
    model.eval()
    n_observations = x.shape[0]
    n_classes = y.shape[-1]
    n_batches = (n_observations + batch_size-1) // batch_size
    x_batches = np.array_split(x, n_batches)
    y_label = y.argmax(-1)

    metrics = {}

    nll = np.concatenate([
            nll_onehot(
                model=model,
                x=torch.tensor(xb, device='cuda'),
                n_samples=n_samples,
                n_classes=n_classes)
            .cpu().detach().numpy()
            for xb in x_batches])
    y_pred_nll = nll.argmin(-1)
    metrics['err_nll'] = (y_pred_nll != y_label).mean()

    prob_euc = np.concatenate([
            min_euclid_prob(
                model,
                torch.tensor(xb, device='cuda'),
                n_samples)
            .cpu().detach().numpy()
            for xb in x_batches])
    y_pred_euc = prob_euc.argmax(-1)
    metrics['err_euc'] = (y_pred_euc != y_label).mean()

    is_wrong = (y_pred_nll != y_label)
    uncertainty = nll.min(-1)
    err_strata_quantile, uncertainty_strata_quantile = (
            error_stratified_by_quantile(
                error=is_wrong.astype(float),
                uncertainty=uncertainty,
                n_strata=100))
    metrics['err_uncert_corr'] = spearmanr(
            err_strata_quantile, uncertainty_strata_quantile)[0]

    prob = softmax(-nll, axis=-1)
    prob_wrong = 1 - prob.max(-1)
    strata_outputs = error_stratified_by_prob(
            error=is_wrong.astype(float),
            prob=prob_wrong, n_strata=100)
    err_observed, err_expected, bin_weight, = strata_outputs
    metrics['overconfidence'] = get_overconfidence(
            err_observed, err_expected, bin_weight)

    is_in = y.sum(-1) > 0
    if not is_in.all():
        score_out = (uncertainty.argsort().argsort() + 0.5) / uncertainty.size
        score_in = 1 - score_out
        metrics['ood_fpr80'] = fpr(is_in, score_in, 0.80)
        metrics['ood_fpr95'] = fpr(is_in, score_in, 0.95)
        metrics['ood_rocauc'] = roc_auc_score(is_in, score_in)
        metrics['ood_aupr'] = average_precision_score(is_in, score_in)
        plot_ood(is_in, score_in, outfile=prefix+'ood.png')

    n_features = x.shape[-1]
    if n_features == 2:
        n_ticks = 100
        extent = (
                np.quantile(x, (0, 1)).T
                + (-0.1, 0.1) * x.std(0, keepdims=True).T)
        x_grid = np.meshgrid(
                np.linspace(extent[0][0], extent[0][1], n_ticks),
                np.linspace(extent[1][0], extent[1][1], n_ticks),
                indexing='ij')
        x_grid = np.stack(x_grid, axis=-1).astype(np.float32)
        x_grid = x_grid.reshape(-1, x_grid.shape[-1])
        nll_grid = np.concatenate([
                nll_onehot(
                    model=model,
                    x=torch.tensor(xb, device='cuda'),
                    n_samples=n_samples,
                    n_classes=n_classes)
                .cpu().detach().numpy()
                for xb in np.array_split(x_grid, n_batches)])
        x_grid = x_grid.reshape(n_ticks, n_ticks, -1)
        nll_grid = nll_grid.reshape(n_ticks, n_ticks, -1)
        prob_grid = softmax(-nll_grid, axis=-1)
        plot_categorical(
                y_label=y_label, x=x,
                grid=x_grid, prob=prob_grid,
                outfile=prefix+'scatter.png')

    outfile = prefix + 'eval-categorical.txt'
    with open(outfile, 'w') as file:
        for key, value in metrics.items():
            print(f'{key}: {value}', file=file)
    print(outfile)
    return metrics


def get_nll_trace(mean, std, y_quantiles, batch_size=10):
    '''
    Compute negative log-lilihood across states and chains.
    Args:
        mean: Mean of the predictive distributions. Has shape
            (n_realizations, n_states, n_chains, n_observations, n_targets)
        std: standard deviation of the predictive distributions. Has shape
            (n_realizations, n_states, n_chains, n_observations, n_targets)
        y_quantiles: Evenly spaced quantiles of the target
            for every observation. Has shape
            (n_quantiles, n_observations, n_targets)
    Returns:
        nll: Has shape (n_states, n_chains).
    '''
    assert mean.ndim == 5
    assert std.shape == mean.shape
    assert y_quantiles.shape[-2:] == mean.shape[-2:]
    assert y_quantiles.ndim == 3

    n_quants = y_quantiles.shape[0]
    mean = np.tile(np.expand_dims(mean, -3), (n_quants, 1, 1))
    std = np.tile(np.expand_dims(std, -3), (n_quants, 1, 1))
    mean = mean.reshape(mean.shape[:-3] + (-1,) + mean.shape[-1:])
    std = std.reshape(std.shape[:-3] + (-1,) + std.shape[-1:])
    y_quantiles = y_quantiles.reshape(-1, y_quantiles.shape[-1])
    n_observations = y_quantiles.shape[0]
    n_batches = (n_observations + batch_size - 1) // batch_size
    n_states = mean.shape[1]
    idx_list = np.array_split(np.arange(n_states), n_batches)
    nll = np.concatenate([
            gaussian_loss(mean[:, idx], std[:, idx], y_quantiles)
            for idx in idx_list])
    return nll
