import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from dalea.model import DaleaGibbs, hardmax
from dalea.make_data import make_one_node


def plot_conf_int(param_raw, states_raw, alpha):
    param = np.array(param_raw)
    states = np.array(states_raw)
    assert param.shape == states.shape[1:]
    ci = np.quantile(states, [alpha/2, 1 - alpha/2], axis=0)
    param_flat = param.flatten()
    ci_lower = ci[0].flatten()
    ci_upper = ci[1].flatten()
    ci_lower_sorted = ci_lower[param_flat.argsort()]
    ci_upper_sorted = ci_upper[param_flat.argsort()]
    param_flat.sort()
    plt.fill_between(
            param_flat, ci_lower_sorted, ci_upper_sorted,
            alpha=0.3, color='tab:orange')
    plt.axline([0, 0], [1, 1])
    return plt


def hardmax_median_corr_pvalue(param, states, prob_null):
    post_cat = hardmax(np.median(states, 0))
    is_correct = post_cat == param
    pvalue = stats.binom_test(
            x=is_correct.sum(),
            n=is_correct.size,
            p=prob_null,
            alternative='greater')
    return pvalue


def median_corr_pvalue(param, states):
    states_median = np.median(states, 0)
    linregress_results = stats.linregress(
            param.flatten(),
            states_median.flatten())
    return linregress_results.pvalue


def get_params_model(n_observations=100, target_type='continuous'):

    n_features = 1
    n_targets = 1
    n_hidden_nodes = 1

    model = DaleaGibbs(
            n_features,
            n_targets,
            n_hidden_nodes,
            target_type=target_type,
            save_all_params=True)
    activation_fn = model.activation_fn
    x, y, params = make_one_node(
            n_observations, activation_fn, target_type)[:3]
    model.set_data(x, y)
    model.reset_params(
            reset_beta_gamma='random',
            reset_u_v='deterministic',
            n_observations=x.shape[0],
            x=x)

    return params, model


def is_in_conf_int(param, param_states, alpha):
    conf_int = np.quantile(
            np.squeeze(param_states),
            [alpha/2, 1-alpha/2])
    return (
            conf_int[0]
            < np.squeeze(param)
            < conf_int[1])


def most_in_conf_int(param, param_states, alpha_conf_int, alpha_binom):
    conf_int = np.quantile(
            param_states,
            [alpha_conf_int/2, 1 - alpha_conf_int/2],
            axis=0)
    is_in = np.logical_and(
            param > conf_int[0],
            param < conf_int[1])
    n_is_in = is_in.sum()
    n_is_in_min, n_is_in_max = stats.binom.ppf(
            [alpha_binom/2, 1 - alpha_binom/2],
            is_in.size,
            1 - alpha_conf_int)
    return n_is_in_min <= n_is_in <= n_is_in_max


def fit_slope(x, y):
    return (x * y).sum() / (x * x).sum()
