#!/usr/bin/env python

import pickle
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from evaluate import evaluate_samples, gof, rhat
import baseline_predictors
from dalea.plot import plot_curve
from hmc import get_sampfun


def predict_blockwise(
        model, x, n_realizations, start=None, stop=None):
    n_states_all = model.get_states_single('beta', 0).shape[0]
    if start is None:
        start = 0
    if stop is None:
        stop = n_states_all
    if start < 0:
        start = n_states_all + start
    if stop < 0:
        stop = n_states_all + stop
    y_dist_list = []
    step = 100
    for i in range(start, stop, step):
        y_dist_single = model.predict(
                x, (n_realizations,),
                start=i,
                stop=min(i+step, stop))
        y_dist_list.append(y_dist_single)
    y_dist = np.concatenate(y_dist_list, -4)
    return y_dist


def load_dalea_model(file):
    model = pickle.load(open(file, 'rb'))
    model.make_activation_fn()
    return model


def adjustment_stage_idx(model, stage=-1, n_states_max=1000):
    assert (
            len(model.priors_history['a_tau'][0])
            == len(model.states['beta'][0]))
    prior_hist = model.get_priors_history()['a_tau'][0]
    prior_uniq = np.unique(prior_hist)
    is_used = np.isclose(prior_hist, prior_uniq[stage])
    start = np.where(is_used)[0][0]
    stop = np.where(is_used)[0][-1] + 1
    assert not (is_used[:start]).any()
    assert not (is_used[stop:]).any()
    assert (is_used[start:stop]).all()
    start = start + (stop - start) // 10
    start = stop - min(stop - start, n_states_max)
    return start, stop


print(sys.argv)
seed = '0001' if len(sys.argv) <= 1 else sys.argv[1]
studyname = 'a' if len(sys.argv) <= 2 else sys.argv[2]
method = 'dalea' if len(sys.argv) <= 3 else sys.argv[3]
# normal: 23544744
# abcd: 23702955
# toy: 23534666

if method not in ['dalea', 'hmc', 'vi', 'sgd']:
    raise ValueError('Method not recognized')
saveid = f'{studyname}_{seed}'
print('saveid:', saveid)
outdir = f'results/{studyname}/{seed}/{method}'
os.makedirs(outdir, exist_ok=True)

n_strata = 5
n_realizations = 20
falserate = 0.10  # for CI
run_baselines = False
plot_results = True

print('Loading data...')
data = pickle.load(open(f'pickles/{saveid}_data.pickle', 'rb'))
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']
# y_mean_truth_train = data['y_mean_train']
y_mean_truth_test = data['y_mean_test']
# y_std_truth_train = data['y_std_train']
# y_std_truth_test = data['y_std_test']

if method == 'dalea':

    dist_file = f'pickles/{saveid}_dist.pickle'
    if os.path.isfile(dist_file):
        dist_dict = pickle.load(open(dist_file, 'rb'))
        y_dist_train = dist_dict['y_dist_train']
        y_dist_test = dist_dict['y_dist_test']
    else:
        print('Loading model...')
        model_file = f'pickles/{saveid}_model.pickle'
        model = load_dalea_model(model_file)
        print('Getting y_dist...')
        adjustment_stage = -1
        n_states_all = model.get_states_single('beta', 0).shape[0]
        print('n_states_all:', n_states_all)
        idx_start, idx_stop = adjustment_stage_idx(model)
        print('adjustment_stage_idx:', idx_start, idx_stop)
        y_dist_train, y_dist_test = [
                predict_blockwise(
                    model, x, n_realizations,
                    start=idx_start, stop=idx_stop)
                for x in [x_train, x_test]]
        dist_dict = {
                'y_dist_train': y_dist_train,
                'y_dist_test': y_dist_test}
        pickle.dump(dist_dict, open(dist_file, 'wb'))
elif method == 'hmc':
    infile_states = f'pickles/{saveid}_hmc.pickle'
    states = pickle.load(open(infile_states, 'rb'))
    n_states = states['beta'].shape[0]
    n_burnin = int(n_states * 0.75)
    phs, eps = get_sampfun(states)
    y_dist_mean_train = phs(x_train)[n_burnin:]
    y_dist_mean_test = phs(x_test)[n_burnin:]
    y_dist_train = y_dist_mean_train + np.random.randn(
            n_realizations, *y_dist_mean_train.shape)
    y_dist_test = y_dist_mean_test + np.random.randn(
            n_realizations, *y_dist_mean_test.shape)

outfile_evaluation = f'{outdir}/evaluation.txt'

if method == 'sgd':

    infile_dalea = f'pickles/{saveid}_model.pickle'
    if os.path.isfile(infile_dalea):
        # initial state of dalea has been optimized by sgd
        model = load_dalea_model(infile_dalea)
        y_pred_train = model.predict(
                x_train, add_random_effects=False,
                start=0, stop=1)[0, 0]
        y_pred_test = model.predict(
                x_test, add_random_effects=False,
                start=0, stop=1)[0, 0]
    else:
        y_pred_train, y_pred_test = baseline_predictors.run_mlp(
                x_train, y_train, x_test, y_test,
                activation='relu',
                outpref=None, seed=data['seed'])
    mse_train = np.square(y_pred_train - y_train).mean()
    mse_test = np.square(y_pred_test - y_test).mean()
    corr_train, pval_train = stats.kendalltau(
            y_train.flatten(), y_pred_train.flatten())
    corr_test, pval_test = stats.kendalltau(
            y_test.flatten(), y_pred_test.flatten())

    with open(outfile_evaluation, 'w') as file:
        print('mse_train:', mse_train, file=file)
        print('mse:', mse_test, file=file)
        print('yfit_corr_train:', corr_train, file=file)
        print('yfit_corr:', corr_test, file=file)
        print('yfit_pval_train:', pval_train, file=file)
        print('yfit_pval:', pval_test, file=file)
    print('Evaluation saved to', outfile_evaluation)

    y_dist_test = y_pred_test

else:
    evaluate_samples(
            y_dist_train=y_dist_train, y_dist_test=y_dist_test,
            y_train=y_train, y_test=y_test,
            y_mean_truth_test=y_mean_truth_test,
            n_strata=n_strata, falserate=falserate,
            outfile=outfile_evaluation)

    if plot_results:
        # plot trace of weighted mse
        trace_gof = gof(y_dist_train, y_train)
        rhat_gof = rhat(trace_gof)
        plt.plot(trace_gof)
        plt.title(f'trace of weighted mse (rhat: {rhat_gof:.3f})')
        plt.savefig(f'{outdir}/rhat.png')
        plt.close()

# plot y vs x curve
n_targets = y_train.shape[-1]
if plot_results and n_targets == 1:
    plt.rcParams.update({'font.size': 10})
    plt.figure(figsize=(8, 3))
    plot_curve(
            y_dist_test, x_test, y_test,
            y_mean_truth_test, falserate)
    method_name = method.upper()
    # plt.title(f'Posterior estimation by {method_name}')
    plt.tick_params(axis='x', direction='in', pad=-15)
    plt.text(
            0.95, 0.2, method_name, fontsize=20,
            horizontalalignment='right',
            verticalalignment='center',
            transform=plt.gca().transAxes)
    plt.savefig(
        f'{outdir}/curve.png',
        dpi=200, bbox_inches='tight')
    plt.close()
    print(f'{outdir}/curve.png')
