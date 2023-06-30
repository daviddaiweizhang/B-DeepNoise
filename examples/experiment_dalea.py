from time import time
# import pickle

from dalea.model import DaleaGibbs
from dalea.cooling import sample_with_cooling


def run_dalea(data, saveid):

    n_hidden_layers = 1
    n_hidden_nodes = 32
    n_chains = 5
    n_thinning = 1
    n_states = 10
    n_attempts = 1
    batch_size = 50
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    data_is_real = data['is_real']
    target_type = data['target_type']
    if 'y_mean_train' in data.keys():
        y_mean_train = data['y_mean_train']
    else:
        y_mean_train = None
    n_features = x_train.shape[-1]
    n_targets = y_train.shape[-1]

    if data_is_real:
        n_adjustments = 1
        activation = 'relu'
    else:
        n_adjustments = 4
        activation = 'bounded_relu'
    variance_adjustment_rate = 1.0
    variance_prior_mean_init = None
    n_realizations = 20
    ci_falserate = 0.10
    rhat_threshold = 1.1
    init_method = 'random'

    model = DaleaGibbs(
            n_features,
            n_targets,
            n_hidden_nodes=n_hidden_nodes,
            n_hidden_layers=n_hidden_layers,
            target_type=target_type,
            activation=activation,
            save_all_params=False,
            save_priors_history=True)

    print('Running MCMC (Gibbs sampler)...')
    t0 = time()
    sample_with_cooling(
            model=model, x=x_train, y=y_train, init=init_method,
            n_states=n_states, n_thinning=n_thinning,
            n_chains=n_chains, n_cooling=n_adjustments,
            n_attempts=n_attempts,
            n_realizations=n_realizations,
            variance_decay_rate=variance_adjustment_rate,
            variance_prior_mean_init=variance_prior_mean_init,
            rhat_threshold=rhat_threshold,
            ci_falserate=ci_falserate,
            y_mean_train=y_mean_train,
            saveid=saveid,
            batch_size=batch_size,
            x_test=x_test, y_test=y_test)
    elapse = int(time() - t0)
    print(f'Done ({elapse} sec).')
    # model.save_to_disk(f'pickles/{saveid}_model.pickle')
    # pickle.dump(adjustment_idx, open(f'pickles/{saveid}_model.pickle', 'wb'))
