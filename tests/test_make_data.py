import warnings

import pytest
import numpy as np
from scipy import stats
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import statsmodels.api as sm

from dalea.make_data import make_one_node
from dalea.model import are_str_and_eq


def bounded_relu(x):
    return np.clip(x, 0.0, 1.0)


@pytest.mark.parametrize('layer', [0, 1])
def test_make_one_node_beta_gamma(layer):
    n_observations = 10000
    alpha = 0.01

    activation_fn = bounded_relu
    x, y, params = make_one_node(n_observations, activation_fn)

    u = params['u'][layer]
    v = params['v'][layer]
    if are_str_and_eq(u, 'x'):
        u = x
    if are_str_and_eq(v, 'y'):
        v = y
    ols = sm.OLS(v.flatten(), sm.add_constant(u.flatten()))
    ols_result = ols.fit()
    conf_int = ols_result.conf_int(alpha)
    assert conf_int[0][0] < params['gamma'][layer] < conf_int[0][1]
    assert conf_int[1][0] < params['beta'][layer] < conf_int[1][1]


@pytest.mark.parametrize('layer', [0, 1])
def test_make_one_node_v(layer):
    n_observations = 10000
    alpha = 0.01
    activation_fn = bounded_relu
    x, y, params = make_one_node(n_observations, activation_fn)

    u = params['u'][layer]
    v = params['v'][layer]
    if are_str_and_eq(u, 'x'):
        u = x
    if are_str_and_eq(v, 'y'):
        v = y
    epsilon = (
            v
            - u @ params['beta'][layer]
            - params['gamma'][layer])
    assert stats.kstest(
            (epsilon / params['tau'][layer]).flatten(),
            'norm').pvalue > alpha


def test_make_one_node_u():
    n_observations = 10000
    alpha = 0.01
    layer = 1

    activation_fn = bounded_relu
    x, y, params = make_one_node(n_observations, activation_fn)

    delta = params['u'][layer] - activation_fn(params['v'][layer-1])
    assert stats.kstest(
            (delta / params['sigma'][layer]).flatten(),
            'norm').pvalue > alpha
