import pytest
import numpy as np

from utils import (
        get_params_model,
        is_in_conf_int,
        median_corr_pvalue,
        hardmax_median_corr_pvalue)


@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
def test_conditional_distributions_beta_gamma_tau(target_type):

    n_observations = 1000
    n_states = 500
    n_burnin = n_states // 5
    alpha = 0.01
    params, model = get_params_model(n_observations, target_type)

    for layer in range(model.n_hidden_layers+1):
        if (
                (
                    target_type == 'continuous'
                    and layer < model.n_hidden_layers)
                or target_type == 'categorical'):
            model.set_param('v', layer, params['v'][layer])
        if layer > 0:
            model.set_param('u', layer, params['u'][layer])
            model.set_param('sigma', layer, params['sigma'][layer])

    for i in range(n_states):
        for layer in range(model.n_hidden_layers+1):
            model.update_beta_gamma(layer)
            model.update_rho(layer)
            model.update_xi(layer)
            model.update_tau(layer)
        if i >= n_burnin:
            model.save_params()
    states = model.get_states()

    for layer in range(model.n_hidden_layers+1):
        for name in ['beta', 'gamma', 'tau']:
            assert is_in_conf_int(
                    params[name][layer],
                    states[name][layer],
                    alpha)


@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
@pytest.mark.parametrize('correct_init_sigma', [True, False])
def test_conditional_distributions_u_sigma(correct_init_sigma, target_type):

    n_observations = 1000
    n_states = 1000
    n_burnin = n_states // 5
    alpha = 0.01
    params, model = get_params_model(n_observations, target_type)

    for layer in range(model.n_hidden_layers+1):
        model.set_param('beta', layer, params['beta'][layer])
        model.set_param('gamma', layer, params['gamma'][layer])
        model.set_param('tau', layer, params['tau'][layer])
        if (
                (
                    target_type == 'continuous'
                    and layer < model.n_hidden_layers)
                or target_type == 'categorical'):
            model.set_param('v', layer, params['v'][layer])
        if layer > 0 and correct_init_sigma:
            model.set_param('sigma', layer, params['sigma'][layer])

    for i in range(n_states):
        for layer in range(1, model.n_hidden_layers+1):
            model.update_u(layer)
            model.update_sigma(layer)
        if i >= n_burnin:
            model.save_params()
    states = model.get_states()

    for layer in range(1, model.n_hidden_layers+1):

        assert median_corr_pvalue(
                params['u'][layer],
                states['u'][layer]) < alpha
        assert is_in_conf_int(
                params['sigma'][layer],
                states['sigma'][layer],
                alpha)


@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
def test_conditional_distributions_u_v(target_type):

    n_observations = 1000
    n_states = 1000
    n_burnin = n_states // 5
    alpha = 0.01
    params, model = get_params_model(n_observations, target_type)

    for layer in range(model.n_hidden_layers+1):
        model.set_param('beta', layer, params['beta'][layer])
        model.set_param('gamma', layer, params['gamma'][layer])
        model.set_param('tau', layer, params['tau'][layer])
        if layer > 0:
            model.set_param('sigma', layer, params['sigma'][layer])

    for i in range(n_states):
        for layer in range(model.n_hidden_layers+1):
            if (
                    (
                        target_type == 'continuous'
                        and layer < model.n_hidden_layers)
                    or target_type == 'categorical'):
                model.update_v(layer)
            if layer > 0:
                model.update_u(layer)
        if i >= n_burnin:
            model.save_params()
    states = model.get_states()

    for layer in range(model.n_hidden_layers+1):

        if (
                (
                    target_type == 'continuous'
                    and layer < model.n_hidden_layers)
                or target_type == 'categorical'):
            assert median_corr_pvalue(
                    params['v'][layer],
                    states['v'][layer]) < alpha

        if layer > 0:
            assert median_corr_pvalue(
                    params['u'][layer],
                    states['u'][layer]) < alpha


@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
def test_conditional_distributions_u_v_sigma(target_type):

    n_observations = 1000
    n_states = 1000
    n_burnin = n_states // 5
    alpha = 0.01
    tolerance = 3
    params, model = get_params_model(n_observations, target_type)

    for layer in range(model.n_hidden_layers+1):
        model.set_param('beta', layer, params['beta'][layer])
        model.set_param('gamma', layer, params['gamma'][layer])
        model.set_param('tau', layer, params['tau'][layer])

    for i in range(n_states):
        for layer in range(model.n_hidden_layers+1):
            if (
                    (
                        target_type == 'continuous'
                        and layer < model.n_hidden_layers)
                    or target_type == 'categorical'):
                model.update_v(layer)
            if layer > 0:
                model.update_u(layer)
                model.update_sigma(layer)
        if i >= n_burnin:
            model.save_params()
    states = model.get_states()

    for layer in range(model.n_hidden_layers+1):

        if (
                (
                    target_type == 'continuous'
                    and layer < model.n_hidden_layers)
                or target_type == 'categorical'):
            assert median_corr_pvalue(
                    params['v'][layer],
                    states['v'][layer]) < alpha

        if layer > 0:
            assert median_corr_pvalue(
                    params['u'][layer],
                    states['u'][layer]) < alpha
            sigma_empirical = np.sqrt(np.mean(np.square(
                    states['u'][layer]
                    - model.activation_fn(states['v'][layer-1]))))
            assert is_in_conf_int(
                    sigma_empirical,
                    states['sigma'][layer],
                    alpha)
            assert np.quantile(
                    states['sigma'][layer],
                    1-alpha,
                    axis=0) < params['sigma'][layer] * tolerance


@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
def test_conditional_distributions_beta_gamma_u(target_type):

    n_observations = 100
    n_states = 1000
    n_burnin = n_states // 5
    alpha = 0.01
    params, model = get_params_model(n_observations, target_type)

    for layer in range(model.n_hidden_layers+1):
        model.set_param('tau', layer, params['tau'][layer])
        if (
                (
                    target_type == 'continuous'
                    and layer < model.n_hidden_layers)
                or target_type == 'categorical'):
            model.set_param('v', layer, params['v'][layer])
        if layer > 0:
            model.set_param('sigma', layer, params['sigma'][layer])

    for i in range(n_states):
        for layer in range(model.n_hidden_layers+1):
            model.update_beta_gamma(layer)
            model.update_rho(layer)
            model.update_xi(layer)
            if layer > 0:
                model.update_u(layer)
        if i >= n_burnin:
            model.save_params()
    states = model.get_states()

    for layer in range(model.n_hidden_layers+1):

        for name in ['beta', 'gamma']:
            assert is_in_conf_int(
                    params[name][layer],
                    states[name][layer],
                    alpha)

        if layer > 0:
            assert median_corr_pvalue(
                    params['u'][layer],
                    states['u'][layer]) < alpha


@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
def test_conditional_distributions_beta_gamma_v(target_type):

    n_observations = 100
    n_states = 1000
    n_burnin = n_states // 5
    alpha = 0.01
    params, model = get_params_model(n_observations, target_type)

    for layer in range(model.n_hidden_layers+1):
        model.set_param('tau', layer, params['tau'][layer])
        if layer > 0:
            model.set_param('u', layer, params['u'][layer])
            model.set_param('sigma', layer, params['sigma'][layer])

    for i in range(n_states):
        for layer in range(model.n_hidden_layers+1):
            model.update_beta_gamma(layer)
            model.update_rho(layer)
            model.update_xi(layer)
            if (
                    (
                        target_type == 'continuous'
                        and layer < model.n_hidden_layers)
                    or target_type == 'categorical'):
                model.update_v(layer)
        if i >= n_burnin:
            model.save_params()
    states = model.get_states()

    for layer in range(model.n_hidden_layers+1):

        for name in ['beta', 'gamma']:
            assert is_in_conf_int(
                    params[name][layer],
                    states[name][layer],
                    alpha)

        if (
                (
                    target_type == 'continuous'
                    and layer < model.n_hidden_layers)
                or target_type == 'categorical'):
            assert median_corr_pvalue(
                    params['v'][layer],
                    states['v'][layer]) < alpha


@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
def test_conditional_distributions_u_v_beta_gamma_last(target_type):

    n_observations = 100
    n_states = 1000
    n_burnin = n_states // 5
    alpha = 0.01
    params, model = get_params_model(n_observations, target_type)

    for layer in range(model.n_hidden_layers+1):
        model.set_param('tau', layer, params['tau'][layer])
        if layer < model.n_hidden_layers:
            model.set_param('beta', layer, params['beta'][layer])
            model.set_param('gamma', layer, params['gamma'][layer])
        if layer > 0:
            model.set_param('sigma', layer, params['sigma'][layer])

    for i in range(n_states):
        for layer in range(model.n_hidden_layers+1):
            if layer > 0:
                model.update_beta_gamma(layer)
                model.update_rho(layer)
                model.update_xi(layer)
            if (
                    (
                        target_type == 'continuous'
                        and layer < model.n_hidden_layers)
                    or target_type == 'categorical'):
                model.update_v(layer)
            if layer > 0:
                model.update_u(layer)
        if i >= n_burnin:
            model.save_params()
    states = model.get_states()

    for layer in range(model.n_hidden_layers+1):

        if (
                (
                    target_type == 'continuous'
                    and layer < model.n_hidden_layers)
                or target_type == 'categorical'):
            assert median_corr_pvalue(
                    params['v'][layer],
                    states['v'][layer]) < alpha

        if layer > 0:
            for name in ['beta', 'gamma']:
                assert is_in_conf_int(
                        params[name][layer],
                        states[name][layer],
                        alpha)
            assert median_corr_pvalue(
                    params['u'][layer],
                    states['u'][layer]) < alpha


@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
@pytest.mark.parametrize('test_all_params', [False, True])
def test_conditional_distributions_beta_gamma_u_v(
        target_type, test_all_params):

    n_observations = 500
    n_states = 1000
    n_burnin = 0
    alpha = 0.01
    n_realizations = 50

    params, model = get_params_model(n_observations, target_type)
    data = model.data

    if target_type == 'categorical':
        n_features = data['y'].shape[1]
        prob_null = 1/(n_features + 1)

    if test_all_params:
        # sample all parameters
        model.sample_states(
                data['x'], data['y'],
                n_states, n_burnin=n_burnin, reset='random')
    else:
        # set variance parameters to true values
        for layer in range(model.n_hidden_layers+1):
            model.set_param('tau', layer, params['tau'][layer])
            if layer > 0:
                model.set_param('sigma', layer, params['sigma'][layer])

        # sample non-variance parameters
        for i in range(n_states):
            for layer in range(model.n_hidden_layers+1):
                model.update_beta_gamma(layer)
                model.update_rho(layer)
                model.update_xi(layer)
                if (
                        (
                            target_type == 'continuous'
                            and layer < model.n_hidden_layers)
                        or target_type == 'categorical'):
                    model.update_v(layer)
                if layer > 0:
                    model.update_u(layer)
            if i >= n_burnin:
                model.save_params()

    states = model.get_states()

    # get point estimates of y from v[1] (categorical only)
    if target_type == 'categorical':
        y_pred_states_from_v1 = states['v'][1]
        assert hardmax_median_corr_pvalue(
                data['y'],
                y_pred_states_from_v1,
                prob_null=prob_null) < alpha

    # get point estimates of y from u[1] without random effects
    y_pred_states_from_u1 = (
            states['u'][1]
            @ states['beta'][1]
            + states['gamma'][1])
    if target_type == 'continuous':
        assert median_corr_pvalue(
                data['y'],
                y_pred_states_from_u1) < alpha
    elif target_type == 'categorical':
        assert hardmax_median_corr_pvalue(
                data['y'],
                y_pred_states_from_u1,
                prob_null=prob_null) < alpha

    # get point estimates of y from v[0] without random effects
    y_pred_states_from_v0 = (
            model.activation_fn(states['v'][0])
            @ states['beta'][1]
            + states['gamma'][1])
    if target_type == 'continuous':
        assert median_corr_pvalue(
                data['y'],
                y_pred_states_from_v0) < alpha
    elif target_type == 'categorical':
        assert hardmax_median_corr_pvalue(
                data['y'],
                y_pred_states_from_v0,
                prob_null=prob_null) < alpha

    # get point estimates of y from x without random effects
    y_pred_states_from_x = (
            model.activation_fn(
                data['x']
                @ states['beta'][0]
                + states['gamma'][0])
            @ states['beta'][1]
            + states['gamma'][1])
    if target_type == 'continuous':
        assert median_corr_pvalue(
                data['y'],
                y_pred_states_from_x) < alpha
    elif target_type == 'categorical':
        assert hardmax_median_corr_pvalue(
                data['y'],
                y_pred_states_from_x,
                prob_null=prob_null) < alpha

    # get distribution of y from x and summarize into point est
    y_realizations = model.predict(
            data['x'], (n_realizations,), return_logits=True)
    y_estimations = np.median(y_realizations, 0)
    if target_type == 'continuous':
        assert median_corr_pvalue(
                data['y'], y_estimations) < alpha
    if target_type == 'categorical':
        assert hardmax_median_corr_pvalue(
                data['y'],
                y_estimations,
                prob_null=prob_null) < alpha
