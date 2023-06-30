import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def variance_vs_residsq(
        y_obsr_raw, y_dist_raw,
        name='', corr_method='pearson'):
    assert y_dist_raw.ndim == y_obsr_raw.ndim + 1
    assert y_dist_raw.shape[1:] == y_obsr_raw.shape
    y_obsr = y_obsr_raw.flatten()
    y_dist = y_dist_raw.reshape(y_dist_raw.shape[-3], -1)
    y_mean = y_dist.mean(0)
    y_var = y_dist.var(0)
    y_resid = y_obsr - y_mean
    y_residsq = y_resid**2

    corr_pearson, pval_pearson = stats.pearsonr(y_var, y_residsq)
    corr_pearlog, pval_pearlog = stats.pearsonr(
            np.log(y_var), np.log(y_residsq))
    corr_spearman, pval_spearman = stats.spearmanr(y_var, y_residsq)
    corr_kendall, pval_kendall = stats.kendalltau(y_var, y_residsq)
    print(f'{name} corr_variaresid_pearson: {corr_pearson}')
    print(f'{name} pval_variaresid_pearson: {pval_pearson}')
    print(f'{name} corr_variaresid_pearlog: {corr_pearlog}')
    print(f'{name} pval_variaresid_pearlog: {pval_pearlog}')
    print(f'{name} corr_variaresid_spearman: {corr_spearman}')
    print(f'{name} pval_variaresid_spearman: {pval_spearman}')
    print(f'{name} corr_variaresid_kendall: {corr_kendall}')
    print(f'{name} pval_variaresid_kendall: {pval_kendall}')

    plt.axline([0, 0], [1, 1])
    plt.plot(
            y_var, y_residsq,
            'o', alpha=0.5, color='tab:blue')
    if corr_method == 'kendall':
        corr = stats.kendalltau
    elif corr_method == 'pearson':
        corr = stats.pearsonr
    elif corr_method == 'spearman':
        corr = stats.spearmanr
    else:
        raise ValueError('`corr_method` not recognized')
    corr, pval = corr(y_var, y_residsq)
    rsq = corr**2
    plt.title(
            f'{name} ('
            f'{corr_method} rsq: {rsq:.3f}, '
            f'pval: {pval:.3f})')
    plt.ylabel('residual squared')
    plt.xlabel('predictive variance')
    plt.xscale('log')
    plt.yscale('log')


def show_performance(
        y_pred_raw, y_obsr_raw,
        y_dist_raw=None, ci_falserate=None,
        name='', corr_method='pearson'):
    assert y_pred_raw.shape == y_obsr_raw.shape
    y_pred_flat = y_pred_raw.flatten()
    y_obsr_flat = y_obsr_raw.flatten()
    order = y_obsr_flat.argsort()
    y_pred = y_pred_flat[order]
    y_obsr = y_obsr_flat[order]
    if (y_dist_raw is not None) and (ci_falserate is not None):
        assert y_dist_raw.ndim == y_obsr_raw.ndim + 1
        assert y_dist_raw.shape[1:] == y_obsr_raw.shape
        y_ci_raw = np.quantile(
                y_dist_raw,
                [ci_falserate/2, 1-ci_falserate/2], 0)
        y_ci_flat = y_ci_raw.reshape(2, -1)
        y_ci = y_ci_flat[:, order]

    plt.plot(
            y_obsr, y_obsr, 'o-', alpha=0.5,
            label='perfect fit',
            color='tab:blue')
    mse_null = y_obsr.var()
    plt.axhline(
            y_obsr.mean(), linestyle='--',
            label=f'null model (mse: {mse_null:02.2f})',
            color='tab:green')

    if (y_dist_raw is not None) and (ci_falserate is not None):
        coverage = (
                (y_obsr >= y_ci[0])
                * (y_obsr <= y_ci[1])).mean()
        plt.fill_between(
                y_obsr,
                y_ci[0], y_ci[1],
                label=(
                    f'predictive {1-ci_falserate:.2f} interval '
                    f'(coverage: {coverage:.2f})'),
                color='tab:orange',
                alpha=0.2)

    corr_pearson, pval_pearson = stats.pearsonr(y_pred, y_obsr)
    corr_spearman, pval_spearman = stats.spearmanr(y_pred, y_obsr)
    corr_kendall, pval_kendall = stats.kendalltau(y_pred, y_obsr)
    print(f'{name} corr_performance_pearson: {corr_pearson}')
    print(f'{name} pval_performance_pearson: {pval_pearson}')
    print(f'{name} corr_performance_spearman: {corr_spearman}')
    print(f'{name} pval_performance_spearman: {pval_spearman}')
    print(f'{name} corr_performance_kendall: {corr_kendall}')
    print(f'{name} pval_performance_kendall: {pval_kendall}')

    mse = np.square(y_pred - y_obsr).mean()
    if corr_method == 'kendall':
        corr = stats.kendalltau
    elif corr_method == 'pearson':
        corr = stats.pearsonr
    elif corr_method == 'spearman':
        corr = stats.spearmanr
    else:
        raise ValueError('`corr_method` not recognized')
    corr, pval = corr(y_pred, y_obsr)
    rsq = corr**2
    if pval > 0:
        neglogpval = -np.log10(pval)
    else:
        neglogpval = np.inf
    plt.plot(
            y_obsr, y_pred,
            label=(
                f'prdictive mean ('
                f'mse: {mse:02.2f}, '
                f'{corr_method} rsq: {rsq:.3f}, '
                f'-log10(p): {neglogpval:03.0f})'),
            color='tab:orange')

    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.legend()


def flatten_dist(x):
    return x.reshape(-1, *x.shape[-2:])


def plot_curve(y_dist, x, y, y_mean_truth, ci_falserate):
    if y_dist.ndim >= 3:
        y_dist_obsr = flatten_dist(y_dist)
        y_dist_mean = flatten_dist(y_dist.mean(-5))
        y_mean = y_dist_mean.mean(0)
        y_ci_obsr = np.quantile(
                y_dist_obsr, [ci_falserate/2, 1-ci_falserate/2], 0)
        y_ci_mean = np.quantile(
                y_dist_mean, [ci_falserate/2, 1-ci_falserate/2], 0)
    elif y_dist.ndim == 2:
        y_mean = y_dist
        y_ci_obsr = None
        y_ci_mean = None

    markersize = 3
    linewidth = markersize * 0.5
    if y_mean_truth is not None:
        plt.plot(
                x[..., 0], y[..., 0], 'o', alpha=0.2,
                label='testing samples',
                color='tab:blue', markersize=markersize)
        plt.plot(
                x[..., 0], y_mean_truth[..., 0],
                label='true mean',
                linewidth=linewidth, color='tab:red')
        plt.plot(
                x[..., 0], y_mean[..., 0], '--',
                label='posterior mean',
                linewidth=linewidth, color='tab:orange')
        if y_ci_mean is not None:
            plt.fill_between(
                    x[..., 0], y_ci_mean[0, ..., 0], y_ci_mean[1, ..., 0],
                    label='posterior CI of mean',
                    color='tab:purple',
                    alpha=0.8)
        if y_ci_obsr is not None:
            plt.fill_between(
                    x[..., 0], y_ci_obsr[0, ..., 0], y_ci_obsr[1, ..., 0],
                    label='posterior CI of predictions',
                    color='tab:blue',
                    alpha=0.2)
        # plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper left', markerscale=3)
