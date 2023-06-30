import dalea.utils as utils
import numpy as np


def get_step_is_saved_list(n_states, n_steps_between_states, n_burnin):
    step_is_saved_list = [False] * n_burnin
    step_is_saved_list += (
            [[True] + [False] * n_steps_between_states] * (n_states - 1))
    step_is_saved_list += [True]
    return step_is_saved_list


def conditional_sample_betagammatau(u, v, prior):
    # Bayesian linear regression with unknown noise variance
    # predictors: u (known)
    # targets: v (known)
    # effects: betagamma (unknown)
    # noise variance: tau^2 (unknown)
    pass


def conditional_sample_u(betagamma, v_prev, sigma_prev, tau, prior):
    # Bayesian linear regression with known noise variance
    # predictors: betagamma.T (known)
    # targets: v.T (known)
    # effects: u.T (unknown)
    # noise variance: tau^2 (known)
    pass


def conditional_sample_v(betagamma, u, u_next, sigma, tau, prior):
    # heterogeneous normal distribution
    # relu (leaky or not): mixture of 2 truncated normal distributions
    # bounded relu (leaky or not): mixture of 3 truncated normal distributions
    pass


def conditional_sample_sigma(u_next, v, prior):
    # InvGamma-Normal conjugate
    # mean: v (known)
    # targets: u_next (known)
    # noise variance: sigma^2 (unknown)
    pass


def get_empty_model(
        n_features, n_targets, hidden_layer_sizes,
        n_observations, chain_shape):

    n_hidden_layers = len(hidden_layer_sizes)
    padded_layer_sizes = (
            [n_features]
            + hidden_layer_sizes
            + [n_targets])

    betagamma = [
            np.empty(chain_shape + [n_nodes_out, n_nodes_in+1])
            for n_nodes_in, n_nodes_out in
            zip(
                padded_layer_sizes[:-1],
                padded_layer_sizes[1:])]
    u = [
            np.empty(
                chain_shape + [n_nodes_in, n_observations])
            for n_nodes_in in padded_layer_sizes[:-1]]
    v = [
            np.empty(
                chain_shape + [n_nodes_out, n_observations])
            for n_nodes_out in padded_layer_sizes[1:]]
    sigma = [
            np.empty(chain_shape)
            for _ in range(n_hidden_layers)]
    tau = [
            np.empty(chain_shape)
            for _ in range(n_hidden_layers + 1)]
    model_dict = {
            'betagamma': betagamma, 'u': u, 'v': v,
            'sigma': sigma, 'tau': tau}
    return utils.dict_to_namedtuple('Model', model_dict)


class DeepAleatoricMLP:

    def __init__(
            self, hidden_layer_sizes, activation='bounded_relu'):
        # n_features = layer_sizes[0]
        # n_targets = layer_sizes[-1]
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._prior = {}

    def set_prior(self, prior):
        # store the model prior parameters
        pass

    def fit(self, x, y, n_states=500, n_steps_between_states=0,
            n_burnin=0, n_chains=1, posterior_sampling_method='standard',
            random_state=None):
        # draw posterior samples of the model parameters

        n_observations, n_features = x.shape[-2:]
        n_targets = y.shape[-1]
        chain_shape = [n_states, n_chains]
        self._model = get_empty_model(
                n_features, n_targets,
                self._hidden_layer_sizes,
                n_observations, chain_shape)
        step_is_saved_list = get_step_is_saved_list(
                n_states, n_steps_between_states, n_burnin)
        self._initialize()
        for step_is_saved in step_is_saved_list:
            self._draw(posterior_sampling_method)
            if step_is_saved:
                self._save()

    def predict(self, x, n_noise_points=100):
        # draw posterior prediction samples of the targets
        # to improve speed, get a finite pool of standard normal samples first,
        # and then keep reusing this pool to generate normal later on
        pass

    def set_model(self, **kwargs):
        # set model parameters
        pass

    def _initialize(self, **kwargs):
        # set the model parameters to random values
        pass

    def _draw(self, method):
        # run one round of Gibbs sampling
        if method == 'standard':
            self._draw_standard()
        elif method == 'backprop':
            self._draw_backprop()

    def _draw_standard(self):
        # draw a posterior sample by standard Gibbs sampling
        self._draw_betagammatau()
        self._draw_v()
        self._draw_u()
        self._draw_sigma()

    def _draw_backprop(self):
        # draw a posterior sample by back propogation Gibbs sampling
        pass

    def _draw_betagammatau(self):
        for i, (u, v) in enumerate(zip(
                self._model.u, self._model.v)):
            self._model.betagamma[i], self._model.tau[i] = (
                    conditional_sample_betagammatau(u, v, self._prior))

    def _draw_u(self):
        for i, (betagamma, v, v_prev, sigma_prev, tau) in enumerate(zip(
                self._model.betagamma[1:],
                self._model.v[1:],
                self._model.v[:-1],
                self._model.sigma[:-1],
                self._model.tau[1:])):
            self._model.u[1+i] = conditional_sample_u(
                    betagamma, v_prev, sigma_prev, tau, self._prior)

    def _draw_v(self):
        for i, (betagamma, u, u_next, sigma, tau) in enumerate(zip(
                self._model.betagamma[:-1],
                self._model.u[:-1],
                self._model.u[1:],
                self._model.sigma[:-1],
                self._model.tau[:-1])):
            self._model.v[i] = conditional_sample_v(
                    betagamma, u, u_next, sigma, tau, **self._prior)

    def _draw_sigma(self):
        for i, (u_next, v) in enumerate(zip(
                self._model.u[1:], self._model.v[:-1])):
            self._model.sigma[i] = conditional_sample_sigma(
                    u_next, v, self._prior)

    def _save(self):
        # save the current state to the state saving list
        self._states.betagamma.append(self._model.betagamma)
        self._states.tau.append(self._model.tau)
        self._states.u.append(self._model.u)
        self._states.v.append(self._model.v)
        self._states.sigma.append(self._model.sigma)
