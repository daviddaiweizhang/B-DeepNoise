import pytest
import numpy as np
from scipy import stats

from dalea.model import DaleaGibbs, softmax


def predict_onelayer(
        x, states, activation_fn, target_type='continuous',
        realization_shape=(), use_random_effects=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    u_0 = x
    v_0_mean = u_0 @ states['beta'][0] + states['gamma'][0]
    if use_random_effects:
        tau_0 = np.expand_dims(states['tau'][0], [-1, -2])
    else:
        tau_0 = 0.0
    epsilon_0 = (
            np.random.randn(*(realization_shape + v_0_mean.shape))
            * tau_0)
    v_0 = v_0_mean + epsilon_0
    u_1_mean = activation_fn(v_0)
    if use_random_effects:
        sigma_1 = np.expand_dims(states['sigma'][1], [-1, -2])
    else:
        sigma_1 = 0.0
    delta_1 = np.random.randn(*u_1_mean.shape) * sigma_1
    u_1 = u_1_mean + delta_1
    v_1_mean = u_1 @ states['beta'][1] + states['gamma'][1]
    if use_random_effects:
        tau_1 = np.expand_dims(states['tau'][1], [-1, -2])
    else:
        tau_1 = 0.0
    del states
    epsilon_1 = np.random.randn(*v_1_mean.shape) * tau_1
    v_1 = v_1_mean + epsilon_1
    if target_type == 'continuous':
        y = v_1
    elif target_type == 'categorical':
        y = softmax(v_1)
    return y


def get_ols(x, y):
    return np.linalg.inv(x.T @ x) @ (x.T @ y)


def test_burnin_thinning():
    n_burnin = 3
    n_thinning = 2
    n_states = 9
    n_features = 4
    n_targets = 2
    n_observations = 5
    n_hidden_nodes = 3

    x = np.random.randn(n_observations, n_features)
    y = np.random.randn(n_observations, n_targets)

    dalea = DaleaGibbs(n_features, n_targets, n_hidden_nodes)
    dalea.sample_states(
            x, y, n_states=n_states,
            n_burnin=n_burnin, n_thinning=n_thinning)
    states = dalea.get_states()
    assert states['beta'][0].shape[0] == n_states


def test_get_param():
    n_observations = 10
    dalea = DaleaGibbs(
            n_features=1, n_targets=1, n_hidden_nodes=1)
    x = np.random.randn(n_observations, 3)
    y = np.random.randn(n_observations, 2)
    dalea.set_data(x, y)
    assert np.allclose(dalea.get_param('u', 0), x)
    assert np.allclose(dalea.get_param('v', 1), y)


def test_set_data_continuous():
    n_observations = 10
    n_targets = 1
    n_features = 3
    dalea = DaleaGibbs(
            n_features=1, n_hidden_nodes=1,
            n_targets=n_targets, target_type='continuous')
    dalea.set_data(
            x=np.random.randn(n_observations, n_features),
            y=np.random.randn(n_observations, 1))
    assert dalea.data['x'] is not None
    assert dalea.data['y'] is not None


def test_set_data_categorical():
    n_observations = 10
    n_targets = 1
    n_features = 3
    dalea = DaleaGibbs(
            n_features=1, n_hidden_nodes=1,
            n_targets=n_targets, target_type='categorical')
    x = np.random.randn(n_observations, n_features)
    y = np.random.choice(n_targets+1, (n_observations, 1))
    dalea.set_data(x=x, y=y)
    assert dalea.data['x'] is not None
    assert dalea.data['y'] is not None

    with pytest.raises(TypeError):
        y_float = y.astype(float)
        dalea.set_data(x=x, y=y_float)

    with pytest.raises(ValueError):
        y_oneindexed = y.copy()
        y_oneindexed[0, 0] = 2
        dalea.set_data(x=x, y=y_oneindexed)

    with pytest.raises(ValueError):
        y_negative = y * (-1)
        dalea.set_data(x=x, y=y_negative)

    with pytest.raises(TypeError):
        y_onehot = np.zeros((n_observations, n_targets+1)).astype(int)
        y_onehot[np.arange(n_observations), y[:, 0]] = 1
        dalea.set_data(x=x, y=y_onehot)


def test_state_not_unchanging():

    n_features = 3
    n_targets = 2
    n_observations = 6

    n_hidden_nodes = 4
    n_states = 5

    x = np.random.randn(n_observations, n_features)
    y = np.random.randn(n_observations, n_targets)

    dalea = DaleaGibbs(n_features, n_targets, n_hidden_nodes)
    dalea.sample_states(x, y, n_states)
    states = dalea.get_states()
    for param in states.values():
        for param_one_layer in param:
            if param_one_layer.dtype.kind not in {'U', 'S'}:
                assert not np.allclose(
                        param_one_layer[0],
                        param_one_layer[-1])


def test_predict_shape():

    n_features = 4
    n_targets = 6
    n_hidden_nodes = 16
    n_hidden_layers = 1
    n_observations_train = 30
    n_observations_test = 40
    chain_shape = (2, 3)
    n_states = 8
    n_realizations = 20

    x_train = np.random.randn(n_observations_train, n_features)
    x_test = np.random.randn(n_observations_test, n_features)
    y_train = np.random.randn(n_observations_train, n_targets)

    dalea = DaleaGibbs(
            n_features, n_targets, n_hidden_nodes,
            n_hidden_layers=n_hidden_layers)
    dalea.sample_states(x_train, y_train, n_states, chain_shape=chain_shape)
    y_pred = dalea.predict(x_test, (n_realizations,))
    assert y_pred.shape == (
            (n_realizations, n_states)
            + chain_shape
            + (n_observations_test, n_targets))


@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
def test_predict_value(target_type):

    n_features = 1
    n_targets = 1
    n_observations = 1
    n_hidden_nodes = 1
    n_states = 1
    n_realizations = 1
    seed = np.random.choice(1000)

    x = np.random.randn(n_observations, n_features)
    if target_type == 'continuous':
        y = np.random.randn(n_observations, n_targets)
    elif target_type == 'categorical':
        y = np.random.choice(n_targets+1, (n_observations, 1))

    dalea = DaleaGibbs(
            n_features, n_targets, n_hidden_nodes,
            target_type=target_type)
    dalea.sample_states(x, y, n_states)
    y_pred = dalea.predict(x, (n_realizations,), seed=seed)
    y_pred_correct = predict_onelayer(
        x=x, states=dalea.get_states(),
        activation_fn=dalea.activation_fn,
        target_type=target_type,
        realization_shape=(n_realizations,), seed=seed)

    assert np.allclose(y_pred, y_pred_correct)


def test_state_shape():

    n_features = 7
    n_targets = 2
    n_observations = 6

    n_hidden_nodes = 4
    n_states = 5

    x = np.random.randn(n_observations, n_features)
    y = np.random.randn(n_observations, n_targets)

    dalea = DaleaGibbs(n_features, n_targets, n_hidden_nodes)
    dalea.sample_states(x, y, n_states)
    states = dalea.get_states()
    assert states['beta'][0].shape == (n_states, n_features, n_hidden_nodes)
    assert states['beta'][1].shape == (n_states, n_hidden_nodes, n_targets)
    assert states['gamma'][0].shape == (n_states, 1, n_hidden_nodes)
    assert states['gamma'][1].shape == (n_states, 1, n_targets)
    assert states['tau'][0].shape == (n_states, 1, n_hidden_nodes)
    assert states['tau'][1].shape == (n_states, 1, n_targets)
    assert states['sigma'][0].shape == (n_states,)
    assert (states['sigma'][0] == 'na').all()
    assert states['sigma'][1].shape == (n_states, 1, n_hidden_nodes)


def test_priors_history():

    n_features = 1
    n_targets = 1
    n_observations = 1

    n_hidden_nodes = 1
    n_states = 5

    x = np.random.randn(n_observations, n_features)
    y = np.random.randn(n_observations, n_targets)

    dalea = DaleaGibbs(
            n_features, n_targets, n_hidden_nodes,
            save_priors_history=True)
    dalea.sample_states(x, y, n_states)
    priors_history = dalea.get_priors_history()
    for prior_name, prior_all_layers in dalea.priors.items():
        for layer, prior in enumerate(prior_all_layers):
            prior_hist = priors_history[prior_name][layer]
            if prior_hist.dtype.kind not in {'S', 'U'}:
                assert prior_hist.shape[-3] == n_states
                assert (
                        np.isclose(prior_hist[-1], prior)
                        or (
                            np.isnan(prior_hist[-1])
                            and np.isnan(prior_hist[-1])))


@pytest.mark.skip(reason='no longer in use')
def test_log_prob_history():

    n_features = 1
    n_targets = 1
    n_observations = 1

    n_hidden_nodes = 1
    n_states = 5

    x = np.random.randn(n_observations, n_features)
    y = np.random.randn(n_observations, n_targets)

    dalea = DaleaGibbs(
            n_features, n_targets, n_hidden_nodes,
            save_log_prob_history=True)
    dalea.sample_states(x, y, n_states)
    log_prob_history = dalea.get_log_prob_history()
    assert log_prob_history.shape == (n_states,)
    assert np.allclose(
            log_prob_history,
            dalea.log_prob(x, y))


class TestResetParams:

    def get_data_model(self, target_type='continuous'):

        n_features = 3
        n_targets = 2
        n_observations = 1000
        n_hidden_nodes = 50

        x = np.random.randn(n_observations, n_features)
        if target_type == 'continuous':
            y = np.random.randn(n_observations, n_targets)
        elif target_type == 'categorical':
            y = np.random.choice(n_targets+1, (n_observations, 1))

        model = DaleaGibbs(
                n_features, n_targets, n_hidden_nodes,
                target_type=target_type)
        model.set_data(x, y)
        data = (x, y)

        return data, model

    @pytest.mark.parametrize('reset_u_v', ['deterministic', 'random'])
    @pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
    def test_reset_params_u_v(self, reset_u_v, target_type):

        (x, y), model = self.get_data_model(target_type)
        n_observations = x.shape[0]
        model.reset_params(
                reset_beta_gamma='random',
                reset_u_v=reset_u_v,
                n_observations=n_observations,
                x=x)
        for layer in range(model.n_hidden_layers+1):
            if layer > 0:
                u_correct = model.activation_fn(
                        model.get_param('v', layer-1))
                u_is_mapped_from_v = np.allclose(
                        model.get_param('u', layer),
                        u_correct)
                if reset_u_v == 'deterministic':
                    assert u_is_mapped_from_v
                else:
                    assert not u_is_mapped_from_v
            if (
                    layer < model.n_hidden_layers
                    or model.target_type == 'categorical'):
                v_correct = (
                    model.get_param('u', layer)
                    @ model.get_param('beta', layer)
                    + model.get_param('gamma', layer))
                v_is_mapped_from_u = np.allclose(
                        model.get_param('v', layer),
                        v_correct)
                if reset_u_v == 'deterministic':
                    assert v_is_mapped_from_u
                elif reset_u_v == 'random':
                    assert not v_is_mapped_from_u

    @pytest.mark.parametrize('reset_u_v', ['deterministic', 'random'])
    @pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
    def test_reset_params_beta_gamma(self, reset_u_v, target_type):

        alpha = 0.01

        (x, y), model = self.get_data_model(target_type)
        n_observations = x.shape[0]
        model.reset_params(
                reset_beta_gamma='random',
                reset_u_v=reset_u_v,
                n_observations=n_observations,
                x=x)

        for layer in range(model.n_hidden_layers):
            v = model.get_param('v', layer)
            u = model.get_param('u', layer)
            u_one = np.concatenate([u, np.ones_like(u[:, -1:])], -1)
            beta_gamma_ols = get_ols(u_one, v)
            beta_gamma_reset = np.concatenate([
                model.get_param('beta', layer),
                model.get_param('gamma', layer)],
                0)
            assert beta_gamma_ols.shape == beta_gamma_reset.shape
            corr_pval = stats.pearsonr(
                    beta_gamma_reset.flatten(),
                    beta_gamma_ols.flatten())[1]
            if reset_u_v == 'deterministic':
                assert corr_pval < alpha
            elif reset_u_v == 'random':
                assert corr_pval > alpha

    @pytest.mark.parametrize('reset_u_v', ['deterministic', 'random'])
    @pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
    def test_reset_params_variances(self, reset_u_v, target_type):

        (x, y), model = self.get_data_model(target_type)
        n_observations = x.shape[0]
        model.reset_params(
                reset_beta_gamma='random',
                reset_u_v=reset_u_v,
                n_observations=n_observations,
                x=x)

        for layer in range(model.n_hidden_layers+1):
            assert (model.get_param('rho', layer) > 0).all()
            assert (model.get_param('xi', layer) > 0).all()
            assert (model.get_param('tau', layer) > 0).all()
            if layer > 0:
                assert (model.get_param('sigma', layer) > 0).all()


@pytest.mark.parametrize('target_type', ['continuous', 'categorical'])
def test_log_prob(target_type):

    n_features = 1
    n_targets = 1
    n_observations = 1
    n_hidden_nodes = 1
    n_hidden_layers = 1

    x = np.zeros((n_observations, n_features))
    y = np.zeros((n_observations, n_targets))
    if target_type == 'categorical':
        y = y.astype(int)
    model = DaleaGibbs(
            n_features, n_targets, n_hidden_nodes,
            n_hidden_layers, target_type=target_type)

    for param in model.scale_param_names:
        for hyperparam in ['a', 'b']:
            for layer in range(n_hidden_layers+1):
                prior = f'{hyperparam}_{param}'
                if layer > 0 or param != 'sigma':
                    value = np.ones_like(model.get_prior(prior, layer))
                    model.set_prior(prior, layer, value)

    model.reset_params(n_observations)
    lp_list = []
    for layer in range(n_hidden_layers+1):
        lp_list.append({})
        for param in model.param_names:
            if model.param_exists(param, layer):
                shape = model.get_param(param, layer).shape
                size = np.prod(shape)
                if param in model.scale_param_names:
                    param_value = np.ones(shape)
                    lp = -1
                elif param in (
                        model.weight_param_names
                        + model.random_effect_param_names):
                    param_value = np.zeros(shape)
                    lp = -0.5 * np.log(2 * np.pi) * size
                model.set_param(param, layer, param_value)
                lp_list[layer][param] = lp
    if target_type == 'continuous':
        y_lp = -0.5 * np.log(2 * np.pi) * y.size
    elif target_type == 'categorical':
        y_lp = np.log(0.5) * n_observations
    lp_list[-1]['y'] = y_lp
    log_prob_correct = np.sum(np.concatenate(
        [list(e.values()) for e in lp_list]))
    log_prob = model.log_prob(x, y, current_state_only=True)
    assert np.allclose(log_prob, log_prob_correct)


def test_softmax():
    n_observations = 100
    n_targets = 9

    assert np.allclose(
            softmax(np.zeros((n_observations, n_targets))),
            1/(n_targets+1))

    x = np.random.randn(n_observations, n_targets)
    assert np.allclose(
            softmax(x, center=False),
            softmax(x, center=True))


def test_activation():
    activation = 'relu'
    n_features = 1
    n_targets = 1
    n_hidden_nodes = 1
    model = DaleaGibbs(
            n_features, n_targets, n_hidden_nodes,
            activation=activation)
    assert np.isclose(model.activation_fn(-1), 0)
    assert np.isclose(model.activation_fn(0), 0)
    assert np.isclose(model.activation_fn(1), 1)


# TODO: test multi-chain
# TODO: test multi-layer
