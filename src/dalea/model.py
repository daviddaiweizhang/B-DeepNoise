import pickle
import sys
from datetime import datetime
from copy import deepcopy

import numpy as np
import tensorflow_probability as tfp

from .samplers_model import (
        sample_features,
        sample_kernel_bias_zero_mean,
        sample_scale_known_mean,
        sample_pre_activation,
        sample_pre_categorization)
from .samplers_basic import sample_invgamma as sample_inverse_gamma
from .numeric import invgamma_log_den as log_prob_inverse_gamma
from .numeric import normal_log_den as log_prob_normal
from utils.evaluate import gaussian_loss

tfd = tfp.distributions


def partition_range(n_total, batch_size):
    n_batches = (n_total + batch_size - 1) // batch_size
    indices = np.random.choice(n_total, n_total, replace=False)
    indices_list = np.array_split(indices, n_batches)
    return indices_list


def has_colinear(x, threshold=0.9999):
    assert x.ndim == 2
    if x.shape[-1] == 1:
        has_colin = False
    else:
        corr = np.corrcoef(x.T)
        idx = np.triu_indices(corr.shape[-1], 1)
        corr = corr[idx]
        has_colin = (np.abs(corr) > threshold).any()
    return has_colin


def get_predictive_distribution(
        features_mean, features_scale, kernel, bias, noise_scale):
    n_features = features_mean.shape[-1]
    assert features_scale.shape[-1] == n_features
    n_targets = kernel.shape[-1]
    assert kernel.shape[-2] == n_features
    assert bias.shape[-2:] == (1, n_targets)
    assert noise_scale.shape[-1] == n_targets

    features_mean = np.expand_dims(features_mean, -2)
    features_scale = np.expand_dims(features_scale, -2)
    bias = np.expand_dims(bias, -2)
    targets_mean = features_mean @ kernel + bias
    targets_cov = (
            kernel.swapaxes(-1, -2) * (features_scale**2) @ kernel)
    idx = list(range(n_targets))
    targets_cov[..., idx, idx] += noise_scale**2
    targets_mean = targets_mean[..., 0, :]
    return targets_mean, targets_cov


def get_piecewise_linear_fn(border, slope, intercept):
    assert np.ndim(border) == 1
    n_components = len(border) + 1
    assert np.shape(slope) == (n_components,)
    assert np.shape(intercept) == (n_components,)

    def piecewise_linear(x):
        i = (np.expand_dims(x, -1) > border).sum(-1)
        y = x * slope[i] + intercept[i]
        return y

    return piecewise_linear


def softmax(x, center=True):
    x_withzero = np.concatenate(
            [x, np.zeros_like(x[..., [-1]])], axis=-1)
    if center:
        x_withzero -= x_withzero.max(-1, keepdims=True)
    prob = (
            np.exp(x_withzero)
            / np.exp(x_withzero).sum(-1, keepdims=True))
    return prob


def hardmax(x):
    with_zeros = np.concatenate([x, np.zeros_like(x[..., -1:])], axis=-1)
    return with_zeros.argmax(-1)[..., np.newaxis]


def are_str_and_eq(x, y):
    return (
            isinstance(x, str)
            and isinstance(y, str)
            and x == y)


def load_from_disk(file):
    model = pickle.load(open(file, 'rb'))
    model.make_activation_fn()
    return model


class DaleaGibbs:

    scale_param_names = ['rho', 'xi', 'sigma', 'tau']
    weight_param_names = ['beta', 'gamma']
    random_effect_param_names = ['u', 'v']
    param_names = (
            scale_param_names
            + weight_param_names
            + random_effect_param_names)

    def __init__(
            self,
            n_features,
            n_targets,
            n_hidden_nodes,
            n_hidden_layers=1,
            target_type='continuous',
            activation='bounded_relu',
            chain_shape=(),
            save_all_params=False,
            save_log_prob_history=False,
            save_priors_history=False,
            extras=None):
        if target_type not in ['continuous', 'categorical']:
            raise ValueError('`target_type` not recognized')
        self.target_type = target_type
        self.n_features = n_features
        self.n_targets = n_targets
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.save_all_params = save_all_params
        self.save_log_prob_history = save_log_prob_history
        if save_log_prob_history:
            self.log_prob_history = []
        else:
            self.log_prob_history = None
        self.save_priors_history = save_priors_history
        if save_priors_history:
            self.init_priors_history()
        else:
            self.priors_history = None

        if activation == 'relu':
            self.border = np.array([0.0])
            self.slope = np.array([0.0, 1.0])
            self.intercept = np.array([0.0, 0.0])
        elif activation == 'bounded_relu':
            self.border = np.array([0.0, 1.0])
            self.slope = np.array([0.0, 1.0, 0.0])
            self.intercept = np.array([0.0, 0.0, 1.0])
        elif activation == 'hard_tanh':
            self.border = np.array([-1.0, 1.0])
            self.slope = np.array([0.0, 1.0, 0.0])
            self.intercept = np.array([-1.0, 0.0, 1.0])
        elif activation == 'leaky_relu':
            self.border = np.array([0.0])
            self.slope = np.array([0.1, 1.0])
            self.intercept = np.array([0.0, 0.0])
        else:
            raise ValueError('Activation not recognized')

        self.extras = extras

        self.x_mean = np.zeros(n_features)
        self.x_std = np.ones(n_features)
        self.y_mean = np.zeros(n_targets)
        self.y_std = np.ones(n_targets)

        self.make_activation_fn()

        self.init_priors(chain_shape)
        self.init_params()
        self.init_data()
        self.init_states(self.save_all_params)

    def make_activation_fn(self):
        self.activation_fn = get_piecewise_linear_fn(
                self.border, self.slope, self.intercept)

    def param_exists(self, name, layer):
        if name in self.scale_param_names:
            if name == 'sigma':
                layer_start = 1
            else:
                layer_start = 0
            layer_stop = self.n_hidden_layers + 1
        elif name in self.weight_param_names:
            layer_start = 0
            layer_stop = self.n_hidden_layers + 1
        elif name in self.random_effect_param_names:
            if name == 'u':
                layer_start = 1
                layer_stop = self.n_hidden_layers + 1
            elif name == 'v':
                layer_start = 0
                if self.target_type == 'continuous':
                    layer_stop = self.n_hidden_layers
                elif self.target_type == 'categorical':
                    layer_stop = self.n_hidden_layers + 1
        else:
            layer_start = 0
            layer_stop = 0
        return layer in range(layer_start, layer_stop)

    def init_priors(self, chain_shape):
        dims = [self.n_hidden_nodes] * (self.n_hidden_layers+2)
        dims[0] = self.n_features
        dims[-1] = self.n_targets
        a = 1e-3
        a_rho = [
                np.full(chain_shape + (din, dout), a)
                for din, dout in zip(dims[:-1], dims[1:])]
        a_xi = [np.full(chain_shape + (1, d), a) for d in dims[1:]]
        a_tau = [np.full(chain_shape + (1, d), a) for d in dims[1:]]
        a_sigma = [np.full(chain_shape + (1, d), a) for d in dims[:-1]]
        a_sigma[0] = 'na'
        self.priors = dict(
                a_rho=a_rho, b_rho=deepcopy(a_rho),
                a_xi=a_xi, b_xi=deepcopy(a_xi),
                a_tau=a_tau, b_tau=deepcopy(a_tau),
                a_sigma=a_sigma, b_sigma=deepcopy(a_sigma))

    def set_prior(self, name, layer, value):
        prior = self.priors[name][layer]
        if name in ['a_sigma', 'b_sigma'] and layer == 0:
            raise ValueError('Prior does not exist')
        if np.shape(value) != np.shape(prior):
            raise ValueError('New shape differs from old shape.')
        self.priors[name][layer] = value

    def get_prior(self, name, layer=None):
        if layer is None:
            prior = self.priors[name]
        else:
            prior = self.priors[name][layer]
            if name in ['a_sigma', 'b_sigma'] and layer == 0:
                raise ValueError('Prior does not exist')
        return prior

    def init_params(self):
        v = [None]*self.n_hidden_layers
        if self.target_type == 'continuous':
            v.append('y')
        elif self.target_type == 'categorical':
            v.append(None)
        self.params = {
                'beta': [None] * (self.n_hidden_layers+1),
                'gamma': [None] * (self.n_hidden_layers+1),
                'rho': [None] * (self.n_hidden_layers+1),
                'xi': [None] * (self.n_hidden_layers+1),
                'tau': [None] * (self.n_hidden_layers+1),
                'v': v,
                'sigma': ['na'] + [None]*self.n_hidden_layers,
                'u': ['x'] + [None]*self.n_hidden_layers}

    def init_data(self):
        self.data = {
                'x': None,
                'y': None,
                'n_observations': None}

    def init_states(self, save_all_params):
        self.states = {
                'beta': [[] for __ in range(self.n_hidden_layers+1)],
                'gamma': [[] for __ in range(self.n_hidden_layers+1)],
                'tau': [[] for __ in range(self.n_hidden_layers+1)],
                'sigma': [[] for __ in range(self.n_hidden_layers+1)]}
        if save_all_params:
            self.states['rho'] = [[] for __ in range(self.n_hidden_layers+1)]
            self.states['xi'] = [[] for __ in range(self.n_hidden_layers+1)]
            self.states['v'] = [[] for __ in range(self.n_hidden_layers+1)]
            self.states['u'] = [[] for __ in range(self.n_hidden_layers+1)]

    def init_priors_history(self):
        self.priors_history = {
                'a_rho': [[] for __ in range(self.n_hidden_layers+1)],
                'b_rho': [[] for __ in range(self.n_hidden_layers+1)],
                'a_xi': [[] for __ in range(self.n_hidden_layers+1)],
                'b_xi': [[] for __ in range(self.n_hidden_layers+1)],
                'a_tau': [[] for __ in range(self.n_hidden_layers+1)],
                'b_tau': [[] for __ in range(self.n_hidden_layers+1)],
                'a_sigma': [[] for __ in range(self.n_hidden_layers+1)],
                'b_sigma': [[] for __ in range(self.n_hidden_layers+1)]}

    def reset_params(
            self, n_observations,
            reset_sigma_tau='random',
            reset_rho_xi='random',
            reset_beta_gamma='random',
            reset_u_v='random',
            x=None):

        for layer in range(self.n_hidden_layers+1):

            # determine which variance params to reset
            scale_name_list = []

            if reset_sigma_tau == 'random':
                if layer > 0:
                    scale_name_list.append('sigma')
                scale_name_list.append('tau')
            elif reset_sigma_tau == 'skip':
                pass
            else:
                raise ValueError('reset method not recognized')

            if reset_rho_xi == 'random':
                scale_name_list += ['rho', 'xi']
            elif reset_rho_xi == 'skip':
                pass
            else:
                raise ValueError('reset method not recognized')

            # sample scale parameters
            for scale_name in scale_name_list:
                # variance = sample_inverse_gamma(
                #         self.get_prior(
                #             'a_{}'.format(scale_name), layer),
                #         self.get_prior(
                #             'b_{}'.format(scale_name), layer),
                #         chain_shape)
                # variance = np.clip(variance, *variance_range)
                ig_concentration = np.ones_like(
                        self.get_prior(f'a_{scale_name}', layer))
                ig_scale = np.ones_like(
                        self.get_prior(f'b_{scale_name}', layer))
                variance = sample_inverse_gamma(
                        concentration=ig_concentration,
                        scale=ig_scale, sample_shape=())
                scale = np.sqrt(variance)
                self.set_param(scale_name, layer, scale)

            if reset_beta_gamma == 'random':
                # set beta to random values
                a_rho = self.get_prior('a_rho', layer)
                b_rho = self.get_prior('b_rho', layer)
                mu_rho = b_rho / a_rho
                beta = np.random.randn(*mu_rho.shape) * np.sqrt(mu_rho)
                self.set_param('beta', layer, beta)

                # set gamma to random values
                a_xi = self.get_prior('a_xi', layer)
                b_xi = self.get_prior('b_xi', layer)
                mu_xi = b_xi / a_xi
                gamma = np.random.randn(*mu_xi.shape) * np.sqrt(mu_xi)
                self.set_param('gamma', layer, gamma)
            elif reset_beta_gamma == 'skip':
                pass
            else:
                raise ValueError('reset method not recognized')

            if (
                    layer < self.n_hidden_layers
                    or self.target_type == 'categorical'):

                if layer == 0:
                    if x is None:
                        u = self.data['x']
                    else:
                        u = x
                else:
                    u = self.get_param('u', layer)
                beta = self.get_param('beta', layer)
                gamma = self.get_param('gamma', layer)
                v = u @ beta + gamma

                if reset_u_v == 'deterministic':
                    pass
                elif reset_u_v in 'semideterministic':
                    tau = self.get_param('tau', layer)
                    noise = np.random.randn(*v.shape) * tau
                    v = v + noise
                elif reset_u_v == 'random':
                    v = np.random.randn(*v.shape)
                else:
                    raise ValueError('reset method not recognized')

                self.set_param('v', layer, v)

            if layer < self.n_hidden_layers:

                u_next = self.activation_fn(v)

                if reset_u_v == 'deterministic':
                    pass
                elif reset_u_v == 'semideterministic':
                    sigma_next = self.get_param('sigma', layer+1)
                    noise = np.random.randn(*u_next.shape) * sigma_next
                    u_next = u_next + noise
                elif reset_u_v == 'random':
                    u_next = np.random.randn(*u_next.shape)
                else:
                    raise ValueError('reset method not recognized')

                self.set_param('u', layer+1, u_next)

    def get_param(self, name, layer=None):
        if layer is None:
            param = self.params[name]
        else:
            param = self.params[name][layer]
            if name == 'u' and are_str_and_eq(param, 'x'):
                param = self.data['x']
            elif name == 'v' and are_str_and_eq(param, 'y'):
                param = self.data['y']
        return param

    def set_param(self, name, layer, value, batch_idx=None):

        # sanity checks
        if not self.param_exists(name, layer):
            raise ValueError('Parameter does not exist')
        param = self.params[name][layer]
        if batch_idx is not None:
            if name not in ['u', 'v']:
                raise ValueError('batch_idx only allowed for `u` and `v`')
            if param is None:
                raise ValueError('param is not initialized')

        # check new value and old value have the same shape
        if batch_idx is not None:
            param_shape = np.shape(
                    param[..., batch_idx, :])
        else:
            param_shape = np.shape(param)
        if param is not None and np.shape(value) != param_shape:
            raise ValueError('New value has incorrect shape')

        # check new value is all finite
        if not np.isfinite(value).all():
            raise ValueError('New value is not all finite.')

        # update parameter value
        if batch_idx is not None:
            self.params[name][layer][..., batch_idx, :] = value
        else:
            self.params[name][layer] = value

    def log_prob(self, x, y, current_state_only=False):
        if current_state_only:
            retrieve_fn = self.get_param
        else:
            retrieve_fn = self.get_states_single
        state_shape = retrieve_fn('beta', 0).shape[:-2]
        state_ndim = len(state_shape)
        lp_list = []
        for layer in range(self.n_hidden_layers+1):
            lp = {}

            rho = retrieve_fn('rho', layer)
            xi = retrieve_fn('xi', layer)
            tau = retrieve_fn('tau', layer)
            lp['rho'] = log_prob_inverse_gamma(
                    rho**2,
                    self.get_prior('a_rho', layer),
                    self.get_prior('b_rho', layer))
            lp['xi'] = log_prob_inverse_gamma(
                    xi**2,
                    self.get_prior('a_xi', layer),
                    self.get_prior('b_xi', layer))
            lp['tau'] = log_prob_inverse_gamma(
                    tau**2,
                    self.get_prior('a_tau', layer),
                    self.get_prior('b_tau', layer))

            beta = retrieve_fn('beta', layer)
            gamma = retrieve_fn('gamma', layer)
            lp['beta'] = log_prob_normal(beta, 0, rho)
            lp['gamma'] = log_prob_normal(gamma, 0, xi)

            if layer == 0:
                u = np.tile(x, state_shape + (1,)*x.ndim)
            else:
                sigma = retrieve_fn('sigma', layer)
                lp['sigma'] = log_prob_inverse_gamma(
                        sigma**2,
                        self.get_prior('a_sigma', layer),
                        self.get_prior('b_sigma', layer))
                u = retrieve_fn('u', layer)
                v_prev = retrieve_fn('v', layer-1)
                u_mean = self.activation_fn(v_prev)
                lp['u'] = log_prob_normal(u, u_mean, sigma)

            v_mean = u @ beta + gamma
            if layer < self.n_hidden_layers:
                v = retrieve_fn('v', layer)
                lp['v'] = log_prob_normal(v, v_mean, tau)
            else:
                y_expanded = np.tile(
                        y, state_shape + (1,)*y.ndim)
                if self.target_type == 'continuous':
                    lp['y'] = log_prob_normal(
                            y_expanded, v_mean, tau)
                elif self.target_type == 'categorical':
                    v = retrieve_fn('v', layer)
                    lp['v'] = log_prob_normal(v, v_mean, tau)
                    v_withzero = np.concatenate(
                            [v, np.zeros_like(v[..., [-1]])],
                            axis=-1)
                    v_centered = v_withzero - v_withzero.max(-1, keepdims=True)
                    logit_hot = np.take_along_axis(
                            v_centered, y_expanded, axis=-1)
                    logit_all = np.log(
                            np.exp(v_centered)
                            .sum(-1, keepdims=True))
                    lp['y'] = logit_hot - logit_all
            lp_list.append(lp)
        log_prob = np.sum([
                va.sum(tuple(range(state_ndim, va.ndim)))
                for e in lp_list
                for va in e.values()],
                axis=0)
        return log_prob

    def set_norm_info(self, x_mean, x_std, y_mean, y_std):
        self.x_mean[:] = x_mean
        self.x_std[:] = x_std
        self.y_mean[:] = y_mean
        self.y_std[:] = y_std

    def nll(self, x, y, n_samples=100):
        __, mean, std = self.predict(
                x=x, realization_shape=(n_samples,),
                return_distribution=True,
                current_state_only=True)
        nll = gaussian_loss(mean, std, y)
        return nll

    def predict(
            self, x, realization_shape=(), seed=None,
            return_logits=False, add_random_effects=True,
            start=None, stop=None, return_distribution=False,
            current_state_only=False):

        if current_state_only:
            assert (start is None) and (stop is None)
            retrieve_fn = self.get_param
        else:
            def retrieve_fn(na, la):
                return self.get_states_single(
                        name=na, layer=la, start=start, stop=stop)

        if seed is not None:
            np.random.seed(seed)

        state_shape = retrieve_fn('beta', 0).shape[:-2]
        x = (x - self.x_mean) / self.x_std
        u = np.tile(x, realization_shape + state_shape + (1,)*x.ndim)
        for layer in range(self.n_hidden_layers+1):
            beta = retrieve_fn('beta', layer)
            gamma = retrieve_fn('gamma', layer)
            tau = retrieve_fn('tau', layer)
            if layer < self.n_hidden_layers:
                sigma_next = retrieve_fn('sigma', layer+1)
            v_mean = u @ beta + gamma
            u = None
            if add_random_effects:
                epsilon = np.random.randn(*v_mean.shape) * tau
                v = v_mean + epsilon
                del v_mean, epsilon
            else:
                v = v_mean

            if layer < self.n_hidden_layers:
                u_next_mean = self.activation_fn(v)
                del v
                if add_random_effects:
                    delta_next = np.random.randn(
                            *u_next_mean.shape) * sigma_next
                    u_next = u_next_mean + delta_next
                    del delta_next
                    if layer < self.n_hidden_layers-1:
                        del u_next_mean
                else:
                    u_next = u_next_mean
                u = u_next

        if return_distribution:
            mean, cov = get_predictive_distribution(
                    features_mean=u_next_mean,
                    features_scale=sigma_next,
                    kernel=np.expand_dims(beta, -3),
                    bias=np.expand_dims(gamma[..., 0, :], -2),
                    noise_scale=tau)
            assert self.n_targets == 1  # for single-target only
            idx = list(range(self.n_targets))
            var = cov[..., idx, idx]
            std = np.sqrt(var)

            n_samples = mean.shape[0]
            std = np.tile(std, (n_samples,) + (1,) * std.ndim)
        del u_next_mean

        if self.target_type == 'continuous':
            y = v * self.y_std + self.y_mean
            if return_distribution:
                mean = mean * self.y_std + self.y_mean
                std = std * self.y_std
                out = y, mean, std
            else:
                out = y
        elif self.target_type == 'categorical':
            if return_logits:
                y = v
            else:
                y = softmax(v)
            out = y
        return out

    def update_v(self, layer, batch_idx=None):

        if are_str_and_eq(self.get_param('v', layer), 'y'):
            raise ValueError('cannot update data y.')

        u = self.get_param('u', layer)
        v = self.get_param('v', layer)
        y = self.data['y']
        if batch_idx is not None:
            u = u[..., batch_idx, :]
            v = v[..., batch_idx, :]
            y = y[..., batch_idx, :]

        pre_nonlinear_mean = (
                u @ self.get_param('beta', layer)
                + self.get_param('gamma', layer))

        if layer < self.n_hidden_layers:
            u_next = self.get_param('u', layer+1)
            if batch_idx is not None:
                u_next = u_next[..., batch_idx, :]
            v_sample = sample_pre_activation(
                    pre_act_mean=pre_nonlinear_mean,
                    pre_act_scale=self.get_param('tau', layer),
                    post_act_observation=u_next,
                    post_act_scale=self.get_param('sigma', layer+1),
                    border=self.border,
                    slope=self.slope,
                    intercept=self.intercept)
        else:
            if self.target_type == 'continuous':
                raise ValueError(
                        'Last `v` for continuous targets is `y`'
                        'and therefore not updatable')
            elif self.target_type == 'categorical':
                v_sample = sample_pre_categorization(
                        pre_cat_mean=pre_nonlinear_mean,
                        pre_cat_scale=self.get_param('tau', layer),
                        pre_cat_observations_old=v,
                        post_cat_observations=y[..., 0]
                    )
        self.set_param(
                'v', layer, v_sample,
                batch_idx=batch_idx)

    def update_u(self, layer, batch_idx=None):

        if are_str_and_eq(self.get_param('u', layer), 'x'):
            raise ValueError('cannot update data x.')

        v = self.get_param('v', layer)
        v_prev = self.get_param('v', layer-1)
        if batch_idx is not None:
            v = v[..., batch_idx, :]
            v_prev = v_prev[..., batch_idx, :]

        features = sample_features(
            targets=v,
            kernel=self.get_param('beta', layer),
            bias=self.get_param('gamma', layer),
            noise_scale=self.get_param('tau', layer),
            features_mean=self.activation_fn(v_prev),
            features_scale=self.get_param('sigma', layer))
        self.set_param('u', layer, features, batch_idx=batch_idx)

    def log_prob_beta_u_v(self, layer):

        beta = self.get_param('beta', layer)
        gamma = self.get_param('gamma', layer)
        u = self.get_param('u', layer)
        v = self.get_param('v', layer)
        w = self.activation_fn(
                    self.get_param('v', layer-1))
        sigma = self.get_param('sigma', layer)
        tau = self.get_param('tau', layer)
        rho = self.get_param('rho', layer)
        xi = self.get_param('xi', layer)

        lp_beta = tfd.Normal(0.0, rho).log_prob(beta)
        lp_gamma = tfd.Normal(0.0, xi).log_prob(gamma)
        lp_u = tfd.Normal(w, sigma).log_prob(u)
        lp_v = tfd.Normal(u @ beta + gamma, tau).log_prob(v)
        lp = np.sum([e.numpy().sum((-1, -2)) for e in [
            lp_beta, lp_gamma, lp_u, lp_v]], 0)
        return lp

    def check_u_colinearity(self, layer):
        u = self.get_param('u', layer)
        u = u.reshape(-1, *u.shape[-2:])
        has_colin = [has_colinear(ui) for ui in u]
        if any(has_colin):
            print(
                    'Warning: colinearity exists among '
                    'post-activation random values of hidden nodes.')

    def update_beta_gamma(self, layer, batch_idx=None):
        self.check_u_colinearity(layer)
        u = self.get_param('u', layer)
        v = self.get_param('v', layer)
        if batch_idx is not None:
            u = u[..., batch_idx, :]
            v = v[..., batch_idx, :]

        kernel, bias = sample_kernel_bias_zero_mean(
                features=u,
                targets=v,
                noise_scale=self.get_param('tau', layer),
                kernel_scale=self.get_param('rho', layer),
                bias_scale=self.get_param('xi', layer))
        self.set_param('beta', layer, kernel)
        self.set_param('gamma', layer, bias)

    def update_rho(self, layer):
        observations = self.get_param('beta', layer)
        scale = sample_scale_known_mean(
                observations=observations,
                mean=np.zeros_like(observations),
                axis=(),
                prior_concentration=self.priors['a_rho'][layer],
                prior_scale=self.priors['b_rho'][layer])
        self.set_param('rho', layer, scale)

    def update_xi(self, layer):
        observations = self.get_param('gamma', layer)
        scale = sample_scale_known_mean(
                observations=observations,
                mean=np.zeros_like(observations),
                axis=(),
                prior_concentration=self.priors['a_xi'][layer],
                prior_scale=self.priors['b_xi'][layer])
        self.set_param('xi', layer, scale)

    def update_tau(self, layer, batch_idx=None):
        u = self.get_param('u', layer)
        v = self.get_param('v', layer)
        if batch_idx is not None:
            u = u[..., batch_idx, :]
            v = v[..., batch_idx, :]

        mean = (
                u @ self.get_param('beta', layer)
                + self.get_param('gamma', layer))
        scale = sample_scale_known_mean(
                observations=v,
                mean=mean,
                axis=(-2,),
                prior_concentration=self.priors['a_tau'][layer],
                prior_scale=self.priors['b_tau'][layer])
        self.set_param('tau', layer, scale)

    def update_sigma(self, layer, batch_idx=None):
        u = self.get_param('u', layer)
        v_prev = self.get_param('v', layer-1)
        if batch_idx is not None:
            u = u[..., batch_idx, :]
            v_prev = v_prev[..., batch_idx, :]

        scale = sample_scale_known_mean(
                observations=u,
                mean=self.activation_fn(v_prev),
                axis=(-2,),
                prior_concentration=self.priors['a_sigma'][layer],
                prior_scale=self.priors['b_sigma'][layer])
        self.set_param('sigma', layer, scale)

    def update_params(self, batch_idx, split_uv_update=True):
        layer_list = list(range(self.n_hidden_layers+1))
        layer_list = layer_list[::-1] + layer_list
        if split_uv_update:
            update_u = batch_idx is None or batch_idx[0] % 2 == 0
            update_v = batch_idx is None or batch_idx[0] % 2 == 1
        for layer in layer_list:
            if (
                    layer < self.n_hidden_layers
                    or self.target_type == 'categorical'):
                if update_v:
                    self.update_v(layer, batch_idx=batch_idx)
            if layer > 0:
                if update_u:
                    self.update_u(layer, batch_idx=batch_idx)
            self.update_beta_gamma(layer)
            self.update_rho(layer)
            self.update_xi(layer)
            if layer > 0:
                self.update_sigma(layer)
            self.update_tau(layer)

    def save_params(self):
        for param_name, state_all_layers in self.states.items():
            param_all_layers = self.get_param(param_name)
            for state_chain, param in zip(
                    state_all_layers, param_all_layers):
                state_chain.append(deepcopy(param))

    def save_log_prob(self):
        self.log_prob_history.append(self.log_prob(
            self.data['x'], self.data['y'],
            current_state_only=True))

    def save_priors(self):
        for prior_name, prior_hist_all_layers in self.priors_history.items():
            prior_all_layers = self.get_prior(prior_name)
            for prior, prior_hist in zip(
                    prior_all_layers, prior_hist_all_layers):
                prior_hist.append(prior)

    def sample_states(
            self, x, y, n_states,
            n_burnin=0, n_thinning=1, reset='random',
            batch_size=None, verbose=False):

        assert n_states >= 1
        assert n_burnin >= 0
        assert n_thinning >= 1

        self.set_data(x, y)
        n_observations = self.data['n_observations']

        if reset == 'random':
            self.reset_params(
                    n_observations,
                    reset_beta_gamma='random',
                    reset_u_v='random')
        elif reset == 'guided':
            self.reset_params(
                    n_observations,
                    reset_beta_gamma='random',
                    reset_u_v='deterministic')
        elif reset == 'skip':
            pass
        else:
            raise ValueError('reset mode not recognized')

        batch_idx_list = []
        for i in range(n_burnin + n_states * n_thinning):
            if i >= n_burnin and (i - n_burnin) % n_thinning == 0:

                # print messages for debugging
                verbose_stride = 1
                if (i - n_burnin) % (n_thinning * verbose_stride) == 0:
                    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                    if verbose:
                        print(
                                f'{timestamp}-{i:04d}',
                                file=sys.stderr, flush=True)

                # save current values of parameters
                self.save_params()

                # save current value of joint probability desnity
                if self.save_log_prob_history:
                    self.save_log_prob()

                # save current values of priors
                if self.save_priors_history:
                    self.save_priors()

            # get a new partition of observations
            if len(batch_idx_list) == 0:
                if batch_size is None:
                    batch_idx_list = [None]
                else:
                    # shuffle partition indices
                    batch_idx_list = partition_range(
                            n_observations, batch_size)

            # get indices of observations
            batch_idx = batch_idx_list.pop(0)
            # update model parameters
            self.update_params(batch_idx=batch_idx)

        self.init_data()

    def set_data(self, x, y):
        n_observations, n_features = x.shape[-2:]
        if y.shape[-2] != n_observations:
            raise ValueError(
                    '`x` and `y` have unequal numbers of observations')
        if self.target_type == 'categorical':
            if not np.issubdtype(y.dtype, np.integer):
                raise TypeError('`y` must be of integer type')
            if not y.shape[-1] == 1:
                raise TypeError('last axis of `y` must have length one')
            if not 0 <= y.min() <= y.max() < self.n_targets + 1:
                raise ValueError('class index must be in [0, `n_targets` + 1)')
        self.data['n_observations'] = n_observations
        self.data['x'] = x
        self.data['y'] = y

    def get_log_prob_history(self):
        if self.log_prob_history is None:
            raise ValueError('log_prob_history does not exist')
        return np.stack(self.log_prob_history)

    def get_priors_history(self):
        if self.priors_history is None:
            raise ValueError('priors_history does not exist')
        priors_history = {}
        for prior_name, prior_hist_all_layers in self.priors_history.items():
            n_layers = len(prior_hist_all_layers)
            priors_history[prior_name] = [
                    self.get_priors_history_single(
                        prior_name, layer)
                    for layer in range(n_layers)]
        return priors_history

    def get_priors_history_single(self, name, layer, start=None, stop=None):
        return np.stack(self.priors_history[name][layer][start:stop])

    def get_states_single(self, name, layer, start=None, stop=None):
        return np.stack(self.states[name][layer][start:stop])

    def set_states_single(self, name, layer, states):
        self.states[name][layer] = [s for s in states]

    def get_states(self):
        states = {}
        for param_name, states_all_layers in self.states.items():
            n_layers = len(states_all_layers)
            states[param_name] = [
                    self.get_states_single(param_name, layer)
                    for layer in range(n_layers)]
        return states

    def set_states(self, states):
        for param_name, states_all_layers in self.states.items():
            n_layers = len(states_all_layers)
            for layer in range(n_layers):
                self.set_states_single(
                        param_name, layer,
                        states[param_name][layer])
        return states

    def save_to_disk(self, outfile):
        activation_fn = self.activation_fn
        self.activation_fn = None
        pickle.dump(self, open(outfile, 'wb'))
        self.activation_fn = activation_fn
