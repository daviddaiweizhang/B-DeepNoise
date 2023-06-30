#! /usr/bin/env python

from time import time
from datetime import datetime
import pickle
import sys

import numpy as np
import tensorflow as tf

from dalea.onelayer import DALEAOneHiddenLayer
from dalea.cooling import sample_with_cooling
from dalea.make_data import simulate_data

saveid = None if len(sys.argv) <= 1 else sys.argv[1]
if saveid is None:
    load_precomputed = False
    saveid = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
else:
    load_precomputed = True
print(f'saveid: {saveid}')

seed = None
if seed is None:
    seed = int(saveid.split('_')[-1])
print(f'seed: {seed}')
np.random.seed(seed)
tf.random.set_seed(seed)

target_type = 'continuous'
n_features = 1
n_targets = 1
n_observations_train = 200
n_observations_test = 2000
x_train, x_test, y_train, y_test, y_mean_train, y_mean_test = simulate_data(
        n_observations_train, n_observations_test, 'continuous')

if load_precomputed:
    data = pickle.load(open(f'pickles/data_{saveid}.pickle', 'rb'))
    x_train, x_test, y_train, y_test = [
            data[e] for e in
            ['x_train', 'x_test', 'y_train', 'y_test']]
else:
    pickle.dump(
            dict(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test),
            open(f'pickles/data_{saveid}.pickle', 'wb'))

n_hidden_layers = 1
n_hidden_nodes = 32
n_chains = 5
n_states = 100
n_thinning = 10
n_attempts = 5
n_adjustments = 4
variance_adjustment_rate = 1.0
variance_prior_mean_init = None
n_realizations = 30
ci_falserate = 0.10
rhat_threshold = 1.1

model = DALEAOneHiddenLayer(
        n_features,
        n_targets,
        n_hidden_nodes=n_hidden_nodes,
        n_hidden_layers=n_hidden_layers,
        target_type=target_type,
        save_all_params=False)

if load_precomputed:
    print('Loading MCMC samples...')
    states = pickle.load(open(f'pickles/states_{saveid}.pickle', 'rb'))
    model.set_states(states)
    print('Done.')
else:
    print('Running MCMC (Gibbs sampler)...')
    t0 = time()
    sample_with_cooling(
            model=model, x=x_train, y=y_train, init='random',
            n_states=n_states, n_thinning=n_thinning,
            n_chains=n_chains, n_cooling=n_adjustments,
            n_attempts=n_attempts,
            n_realizations=n_realizations,
            variance_decay_rate=variance_adjustment_rate,
            variance_prior_mean_init=variance_prior_mean_init,
            rhat_threshold=rhat_threshold,
            ci_falserate=ci_falserate,
            y_mean_train=y_mean_train,
            saveid=saveid)
    elapse = int(time() - t0)
    print(f'Done ({elapse} sec).')
