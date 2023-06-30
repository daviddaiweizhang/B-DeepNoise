from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import losses
import numpy as np

# Custom activation function
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects

from .model import softmax


def truncated_relu(x):
    threshold = 1.0
    return -K.relu(-K.relu(x) + threshold) + threshold


get_custom_objects().update({'truncated_relu': Activation(truncated_relu)})


def get_nn_model(
        depth, width, n_targets,
        activation_middle, activation_last, **kwargs):
    model = Sequential()
    for _ in range(depth):
        model.add(Dense(width, activation=activation_middle, **kwargs))
    model.add(Dense(n_targets, activation=activation_last, **kwargs))
    return model


def train_nn_model(
        model, x, y, target_type,
        learning_rate, batch_size, epochs, sample_weight=None,
        earlystop_patience=None,
        earlystop_tol=0.01,
        reducelr_patience=None,
        reducelr_tol=0.01,
        from_logits=None,
        verbose=0,
        x_test=None, y_test=None):

    n_observations = y.shape[0]
    test = (x_test is not None) and (y_test is not None)

    if target_type == 'continuous':
        loss = 'mse'
        metric_name = 'mse'
        metric_null = y.var()
        if test:
            metric_null_test = y_test.var()
    elif target_type == 'categorical_onehot':
        assert y.ndim == 2
        loss = losses.CategoricalCrossentropy(
                from_logits=from_logits)
        metric_name = 'accuracy'
        metric_null = y.sum(0).max() / n_observations
        if test:
            metric_null_test = y_test.sum(0).max() / n_observations
    elif target_type == 'categorical_sparse':
        assert y.ndim == 2
        loss = losses.SparseCategoricalCrossentropy(
                from_logits=from_logits)

        metric_name = 'accuracy'
        metric_null = (
                np.bincount(y.flatten()).max()
                / n_observations)
        if test:
            metric_null_test = (
                    np.bincount(y_test.flatten()).max()
                    / n_observations)
    else:
        raise ValueError('`target_type` not recognized')

    callbacks = []
    if earlystop_patience is not None:
        callbacks.append(EarlyStopping(
                monitor='loss',
                patience=earlystop_patience,
                min_delta=earlystop_tol,
                verbose=1))
    if reducelr_patience is not None:
        callbacks.append(ReduceLROnPlateau(
                monitor='loss',
                patience=reducelr_patience,
                factor=0.5,
                min_lr=0.001,
                min_delta=reducelr_tol,
                verbose=1))

    model.compile(
            optimizer=Adam(learning_rate),
            loss=loss, metrics=[metric_name])
    model.fit(
            x, y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose)
    metric_fit = model.evaluate(
            x, y, verbose=0, return_dict=True)[metric_name]
    if test:
        metric_fit_test = model.evaluate(
                x_test, y_test,
                verbose=0, return_dict=True)[metric_name]
    if metric_name == 'accuracy':
        metric_fit = 1 - metric_fit
        metric_null = 1 - metric_null
        if test:
            metric_fit_test = 1 - metric_fit_test
            metric_null_test = 1 - metric_null_test
    metric_scaled = metric_fit / metric_null
    print(
            'nn model training loss: '
            f'{metric_scaled:.3f} '
            f'({metric_fit:.3f} / {metric_null:.3f})')
    if test:
        metric_scaled_test = metric_fit_test / metric_null_test
        print(
                'nn model testing loss: '
                f'{metric_scaled_test:.3f} '
                f'({metric_fit_test:.3f} / '
                f'{metric_null_test:.3f})')
    return metric_scaled


def get_extractor(model):
    return Model(
            inputs=model.inputs,
            outputs=[layer.output for layer in model.layers])


def nn_to_dalea_param_dict(
        model, x,
        random_effect_variance_proportion,
        target_type='continuous'):
    weights = model.get_weights()
    assert len(weights) % 2 == 0
    beta = weights[0::2]
    gamma = [w.reshape(1, -1) for w in weights[1::2]]
    n_targets = beta[-1].shape[-1]
    if target_type == 'categorical':
        A = np.eye(n_targets)[:, :-1]
        A[-1] = -1
        beta[-1] = beta[-1] @ A
        gamma[-1] = gamma[-1] @ A
    rho = [np.sqrt(np.mean(e**2)) for e in beta]
    xi = [np.sqrt(np.mean(e**2)) for e in gamma]
    extractor_model = get_extractor(model)
    intermediate_outputs = extractor_model(x)
    random_effect_variances = [
            e.numpy().var() * random_effect_variance_proportion / 2
            for e in intermediate_outputs]
    tau = [np.sqrt(e) for e in random_effect_variances]
    sigma = [np.sqrt(e) for e in random_effect_variances]
    sigma[0] = None
    return dict(
            beta=beta, gamma=gamma,
            rho=rho, xi=xi,
            tau=tau, sigma=sigma)


def initialize_dalea_with_sgd(
        model, x, y, chain_shape,
        max_loss=0.95,
        max_training_attempts=10,
        perturbation=None,
        x_test=None,
        y_test=None,
        **kwargs):

    if np.allclose(model.border, [0, 1]):
        activation_middle = 'truncated_relu'
    if np.allclose(model.border, [0, np.inf]):
        activation_middle = 'relu'
    print(f'activation: {activation_middle}')

    n_observations = x.shape[-2]
    chain_size = np.prod(chain_shape)
    nn_params_list = []
    nn_pred_list = []
    if model.target_type == 'continuous':
        nn_target_type = 'continuous'
        activation_last = None
        n_targets = model.n_targets
    elif model.target_type == 'categorical':
        nn_target_type = 'categorical_sparse'
        activation_last = 'softmax'
        n_targets = model.n_targets+1

    nn_mean_list = []
    for chain in range(chain_size):
        print(f'Initializing mean nn chain {chain}...')
        for i in range(max_training_attempts):
            nn = get_nn_model(
                    depth=model.n_hidden_layers,
                    width=model.n_hidden_nodes,
                    n_targets=n_targets,
                    activation_middle=activation_middle,
                    activation_last=activation_last,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='glorot_uniform')
            loss = train_nn_model(
                    nn, x, y,
                    target_type=nn_target_type,
                    from_logits=False,
                    x_test=x_test,
                    y_test=y_test,
                    **kwargs)
            if loss < max_loss:
                break
            else:
                if i < max_training_attempts - 1:
                    del nn
                else:
                    print('Warning: Training has not converged')
        nn_pred_list.append(nn.predict(x))
        param_dict = nn_to_dalea_param_dict(
                nn, x,
                random_effect_variance_proportion=0.2,
                target_type=model.target_type)
        nn_params_list.append(param_dict)
        nn_mean_list.append(nn)

        if nn_target_type == 'continuous':
            nn_logvar_list = []
            print(f'Initializing logvar nn chain {chain}...')
            y_pred = nn.predict(x)
            log_resid_sq = np.log((y - y_pred)**2)
            y_pred_test = nn.predict(x_test)
            log_resid_sq_test = np.log((y_test - y_pred_test)**2)
            for i in range(max_training_attempts):
                nn_logvar = get_nn_model(
                        depth=model.n_hidden_layers,
                        width=model.n_hidden_nodes,
                        n_targets=n_targets,
                        activation_middle=activation_middle,
                        activation_last=activation_last,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='glorot_uniform')
                loss_logvar = train_nn_model(
                        nn_logvar, x, log_resid_sq,
                        target_type='continuous',
                        from_logits=False,
                        x_test=x_test,
                        y_test=log_resid_sq_test,
                        **kwargs)
                if loss_logvar < max_loss:
                    break
                else:
                    if i < max_training_attempts - 1:
                        del nn_logvar
                    else:
                        print(
                                'Warning: Training of logvar '
                                'nn has not converged')
            nn_logvar_list.append(nn_logvar)

    nn_params = {}
    for param_name in nn_params_list[0].keys():
        nn_params[param_name] = []
        for i in range(len(nn_params_list[0]['beta'])):
            param_flat = np.stack([e[param_name][i] for e in nn_params_list])
            param = param_flat.reshape(
                    *chain_shape, *param_flat.shape[1:])
            nn_params[param_name].append(param)
    for param_name in nn_params.keys():
        for layer in range(model.n_hidden_layers+1):
            param_value = nn_params[param_name][layer]
            if param_value is not None:
                # TODO: take care of non-negative parameters
                if perturbation is not None:
                    noise_std = np.sqrt(
                            param_value.var() * perturbation)
                    param_value += (
                            noise_std * np.random.randn(*param_value.shape))
                if layer > 0 or param_name != 'sigma':
                    model.set_param(
                            param_name, layer, param_value)

    model.reset_params(
            n_observations, chain_shape=chain_shape,
            reset_sigma_tau='skip',
            reset_rho_xi='skip',
            reset_beta_gamma='skip',
            reset_u_v='deterministic',
            x=x)

    # TODO: move these assertions to tests/
    # check output values by using predict
    model.save_params()
    assert np.allclose(
            model.predict(x, add_random_effects=False)[-1],
            np.stack(nn_pred_list),
            1e-4, 1e-4)
    model.init_states(model.save_all_params)
    # check output values by using random effects of final layer
    if model.target_type == 'continuous':
        pass  # TODO: implement
    if model.target_type == 'categorical':
        assert np.allclose(
            softmax(model.get_param('v', model.n_hidden_layers)),
            np.stack(nn_pred_list))
    # layer_outputs = extract_layer_outputs(nn, x)
    # # assert layer_outputs consistent with v and y in model

    to_return = nn_mean_list
    if nn_target_type == 'continuous':
        to_return = (to_return, nn_logvar_list)
    return to_return


def extract_layer_outputs(model, x):
    extractor = Model(
            inputs=model.inputs,
            outputs=[layer.output for layer in model.layers])
    return extractor(x)
