#!/usr/bin/env python

import pickle
from time import time
import sys
import os

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def get_sampfun(samples):
    # does this improve speed?
    samples = {
            ke: np.stack(va)
            if isinstance(va, list) else va
            for ke, va in samples.items()}
    als = samples['alpha']
    bes = samples['beta']
    nus = samples['nu']
    xis = samples['xi']
    sigsqs = samples['sigsq']

    def phs(x=None, zetas=None):
        assert (x is not None) or (zetas is not None)
        if zetas is None:
            return activate(x @ als + nus) @ bes + xis
        else:
            if isinstance(zetas, list):
                zetas = np.stack(zetas, axis=0)
            return activate(zetas) @ bes + xis

    return phs, sigsqs


def fit_ls(Y, X, alpha=None):
    n, q = X.shape[-2:]
    assert Y.shape[-2] == n
    df = n - q
    A = np.swapaxes(X, -1, -2) @ X
    if n < q:
        print('Warning: n < q. Adding a diagnal matrix')
        evals = np.linalg.eigh(A)[0]
        emin = evals[..., ~np.isclose(evals, 0)].min(-1)
        A += np.eye(q) * emin
        df = n
    A_inv = np.linalg.inv(A)
    assert (A_inv.diagonal(0, -2, -1) > 0).all()
    beta = A_inv @ np.swapaxes(X, -1, -2) @ Y
    sigmasq = ((Y - X @ beta)**2).sum(-2, keepdims=True) / df
    se = np.sqrt(sigmasq * A_inv.diagonal(0, -2, -1)[..., np.newaxis])
    pval = (1 - stats.t.cdf(np.abs(beta / se), df=df))*2
    if alpha is None:
        return beta, se, pval
    else:
        cutoff = se * stats.t.ppf(1 - alpha / 2, df=df)
        return beta, se, pval, cutoff


def logprob(pars, data):
    # TODO: softwire prior
    al, be, nu, xi, sigsq = pars
    x, y = data
    sigsq_rv = tfd.InverseGamma(1.0, 1.0)
    kernel_rv = tfd.Normal(loc=0.0, scale=10.0)
    # TODO: use the same function for numpy and tensorflow
    y_mean = tf.clip_by_value(x @ al + nu, 0, 1) @ be + xi
    y_rv = tfd.Normal(loc=y_mean, scale=tf.sqrt(tf.expand_dims(sigsq, -2)))
    # TODO: make sure sum does not eliminate chain dim
    kernel_logprob = tf.reduce_sum([
        tf.reduce_sum(kernel_rv.log_prob(al), (-1, -2)),
        tf.reduce_sum(kernel_rv.log_prob(be), (-1, -2)),
        tf.reduce_sum(kernel_rv.log_prob(nu), (-1, -2)),
        tf.reduce_sum(kernel_rv.log_prob(xi), (-1, -2))])
    y_logprob = tf.reduce_sum(y_rv.log_prob(y), (-1, -2))
    sigsq_logprob = tf.reduce_sum(sigsq_rv.log_prob(sigsq), -1)
    return kernel_logprob + y_logprob + sigsq_logprob


def make_target(data):
    def target(*pars):
        return logprob(pars, data)
    return target


def addbia_x(x, axis=-1):
    s = list(x.shape)
    s[axis] = 1
    return np.concatenate([x, np.ones(s)], axis)


def activate(x):
    return np.clip(x, 0, 1)


def init_pars(x, y, R, nchain=1, dtype=np.float32):
    N = x.shape[-2]
    assert y.shape[-2:] == (N, 1)
    ze = np.random.randn(nchain, N, R)
    alnu = fit_ls(ze, addbia_x(x))[0]
    al, nu = alnu[..., :-1, :], alnu[..., -1:, :]
    zefit = x @ al + nu
    tausq = ((ze - zefit)**2).mean((-1, -2))[..., np.newaxis]
    # tausq = ze.var() * np.ones((nchain, 1))
    bexi = fit_ls(y, addbia_x(activate(ze)))[0]
    be, xi = bexi[..., :-1, :], bexi[..., -1:, :]
    yfit = activate(x @ al + nu) @ be + xi
    sigsq = ((y - yfit)**2).mean(-2)
    pars = [al, be, nu, xi, sigsq, tausq, ze]
    pars = [dtype(e) for e in pars]
    return pars


def sample_hmc(x, y, R, train):
    step_size = train['step_size']
    num_results = train['num_samples']
    num_steps_between_results = train['num_steps_between_results']
    num_leapfrog_steps = train['num_leapfrog_steps']
    target_accept_prob = train['target_accept_prob']
    num_chains = train['num_chains']
    assert isinstance(num_chains, int)

    num_burnin_steps = 0
    num_adaptation_steps = num_results
    trace_fn = None

    target = make_target((x, y))
    if 'init_state' in train.keys():
        print('Using provided initial state.')
        init_state = train['init_state']
    else:
        print('Using optimized initial state')
        init_state = init_pars(x, y, R, nchain=num_chains)[:5]
    step_size = [e.std() * step_size for e in init_state]

    states = tfp.mcmc.sample_chain(
      num_results=num_results,
      current_state=init_state,
      kernel=tfp.mcmc.SimpleStepSizeAdaptation(
          inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
              target_log_prob_fn=target,
              step_size=step_size,
              num_leapfrog_steps=num_leapfrog_steps),
          target_accept_prob=target_accept_prob,
          num_adaptation_steps=num_adaptation_steps),
      num_steps_between_results=num_steps_between_results,
      num_burnin_steps=num_burnin_steps,
      trace_fn=trace_fn,
      parallel_iterations=10)

    als, bes, nus, xis, sigsqs = [e.numpy() for e in states]
    # sigsqs = np.tile(sigsq, (num_results, num_chains, y.shape[-1]))
    states = {'alpha': als, 'beta': bes, 'nu': nus, 'xi': xis, 'sigsq': sigsqs}
    return states


def get_init_state_dalea(infile, x, y, dtype=np.float32):
    dtype = np.float32
    dalea_model = pickle.load(open(infile, 'rb'))
    dalea_model.make_activation_fn()
    dalea_states = dalea_model.get_states()
    al = dalea_states['beta'][0][0]
    be = dalea_states['beta'][1][0]
    nu = dalea_states['gamma'][0][0]
    xi = dalea_states['gamma'][1][0]
    yfit = dalea_model.activation_fn(x @ al + nu) @ be + xi
    sigsq = ((y - yfit)**2).mean(-2)
    init_state = [al, be, nu, xi, sigsq]
    init_state = [dtype(e) for e in init_state]
    return init_state


def main():

    seed = '0001' if len(sys.argv) <= 1 else sys.argv[1]
    studyname = 'a' if len(sys.argv) <= 2 else sys.argv[2]
    # normal: 23544744
    # abcd: 23557582
    # toy: 23534666

    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))

    saveid = f'{studyname}_{seed}'
    print('saveid:', saveid)
    outdir = f'results/{studyname}/{seed}/hmc'
    os.makedirs(outdir, exist_ok=True)
    print('outdir:', outdir)

    infile_data = f'pickles/{saveid}_data.pickle'
    print(f'Using data from {infile_data}')
    data = pickle.load(open(infile_data, 'rb'))
    x_train = data['x_train']
    y_train = data['y_train']

    n_hidden_nodes = 32
    n_states = 2400
    n_thinning = 10
    n_chains = 5

    infile_dalea = f'pickles/{saveid}_model.pickle'
    print(f'Using initial state from {infile_dalea}')
    init_state = get_init_state_dalea(
            infile=infile_dalea,
            x=x_train, y=y_train)
    training_settings = {
            'num_samples': n_states,
            'num_steps_between_results': n_thinning,
            'step_size': 1e-3,
            'num_leapfrog_steps': 10,
            'target_accept_prob': 0.6,
            'num_chains': n_chains,
            'init_state': init_state}
    tstart = time()
    states = sample_hmc(
            x_train, y_train, n_hidden_nodes,
            training_settings)
    print('elapse:', time() - tstart)
    outfile_states = f'pickles/{saveid}_hmc.pickle'
    pickle.dump(states, open(outfile_states, 'wb'))
    print('HMC model states saved to', outfile_states)


if __name__ == '__main__':
    main()
