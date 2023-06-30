import pytest
import numpy as np

from utils import (
        get_params_model,
        is_in_conf_int,
        median_corr_pvalue,
        most_in_conf_int)


@pytest.mark.parametrize('layer', [0, 1])
@pytest.mark.parametrize('name', ['beta', 'gamma'])
@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
def test_conditional_distribution_beta_gamma(name, layer, target_type):

    n_states = 500
    n_burnin = n_states // 5
    alpha = 0.01
    params, model = get_params_model(target_type=target_type)

    model.set_param('tau', layer, params['tau'][layer])
    if target_type == 'continuous':
        v_layer_upper = model.n_hidden_layers
    elif target_type == 'categorical':
        v_layer_upper = model.n_hidden_layers + 1
    if layer < v_layer_upper:
        model.set_param('v', layer, params['v'][layer])
    if layer > 0:
        model.set_param('u', layer, params['u'][layer])

    param_states_list = []
    for i in range(n_states):
        model.update_beta_gamma(layer)
        if i >= n_burnin:
            param_states_list.append(model.get_param(name, layer))
    assert is_in_conf_int(params[name][layer], param_states_list, alpha)


@pytest.mark.parametrize('layer', [0, 1])
@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
def test_conditional_distribution_tau(layer, target_type):

    n_observations = 1000
    n_states = 500
    n_burnin = n_states // 5
    alpha = 0.01
    name = 'tau'
    params, model = get_params_model(n_observations, target_type)

    model.set_param('beta', layer, params['beta'][layer])
    model.set_param('gamma', layer, params['gamma'][layer])
    if target_type == 'continuous':
        v_layer_upper = model.n_hidden_layers
    elif target_type == 'categorical':
        v_layer_upper = model.n_hidden_layers + 1
    if layer < v_layer_upper:
        model.set_param('v', layer, params['v'][layer])
    if layer > 0:
        model.set_param('u', layer, params['u'][layer])

    param_states_list = []
    for i in range(n_states):
        model.update_tau(layer)
        if i >= n_burnin:
            param_states_list.append(model.get_param(name, layer))
    assert is_in_conf_int(params[name][layer], param_states_list, alpha)


@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
def test_conditional_distribution_sigma(target_type):

    n_states = 500
    n_burnin = n_states // 5
    alpha = 0.01
    name = 'sigma'
    params, model = get_params_model(target_type=target_type)

    for layer in range(1, model.n_hidden_layers+1):
        model.set_param('u', layer, params['u'][layer])
        model.set_param('v', layer-1, params['v'][layer-1])

        assert model.priors['a_sigma'][layer] / params['u'][layer].size < 0.01
        sum_of_squares = np.square(
                params['u'][layer]
                -
                model.activation_fn(params['v'][layer-1])).sum()
        assert model.priors['b_sigma'][layer] / sum_of_squares < 0.01

        param_states_list = []
        for i in range(n_states):
            model.update_sigma(layer)
            if i >= n_burnin:
                param_states_list.append(model.get_param(name, layer))
        assert is_in_conf_int(params[name][layer], param_states_list, alpha)


@pytest.mark.parametrize('target_type, layer', [
    ('continuous', 0),
    ('categorical', 0),
    ('categorical', 1)])
def test_conditional_distribution_v(layer, target_type):

    n_observations = 100
    n_states = 500
    n_burnin = n_states // 5
    alpha = 0.01
    name = 'v'
    n_hidden_layers = 1
    params, model = get_params_model(n_observations, target_type)

    model.set_param('beta', layer, params['beta'][layer])
    model.set_param('gamma', layer, params['gamma'][layer])
    model.set_param('tau', layer, params['tau'][layer])
    if layer < n_hidden_layers:
        model.set_param('u', layer+1, params['u'][layer+1])
        model.set_param('sigma', layer+1, params['sigma'][layer+1])
    if layer > 0:
        model.set_param('u', layer, params['u'][layer])
        model.set_param('sigma', layer, params['sigma'][layer])

    param_states_list = []
    for i in range(n_states):
        model.update_v(layer)
        if i >= n_burnin:
            param_states_list.append(model.get_param(name, layer))

    assert median_corr_pvalue(params[name][layer], param_states_list) < alpha


@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
def test_conditional_distribution_u(target_type):

    n_states = 1000
    n_burnin = n_states // 5
    alpha_conf_int = 0.1
    alpha_binom = 0.01
    name = 'u'
    params, model = get_params_model(target_type=target_type)

    for layer in range(1, model.n_hidden_layers+1):
        model.set_param('beta', layer, params['beta'][layer])
        model.set_param('gamma', layer, params['gamma'][layer])
        model.set_param('tau', layer, params['tau'][layer])
        model.set_param('sigma', layer, params['sigma'][layer])
        model.set_param('v', layer-1, params['v'][layer-1])
        if (
                layer < model.n_hidden_layers
                or target_type == 'categorical'):
            model.set_param('v', layer, params['v'][layer])

        for i in range(n_states):
            model.update_u(layer)
            if i >= n_burnin:
                model.save_params()
        param_states_list = model.get_states_single(name, layer)
        assert most_in_conf_int(
                params[name][layer],
                param_states_list,
                alpha_conf_int,
                alpha_binom)
