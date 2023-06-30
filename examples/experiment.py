#! /usr/bin/env python

from datetime import datetime
import sys
import os
import pickle

import numpy as np
import tensorflow as tf

from get_data import get_data
from experiment_dalea import run_dalea


def run_method(method, data, saveid):
    if method == 'dalea':
        run_dalea(data, saveid)


def evaluate(outcomes):
    pass


dataname = 'kin8nm' if len(sys.argv) <= 1 else sys.argv[1]
method = 'dalea' if len(sys.argv) <= 2 else sys.argv[2]
seed = None if len(sys.argv) <= 3 else sys.argv[3]
studyname = None if len(sys.argv) <= 4 else sys.argv[4]

# set seed
if seed is None:
    seed = np.random.choice(int(1e4))
else:
    seed = int(seed)
print(f'seed: {seed}')
np.random.seed(seed)
tf.random.set_seed(seed)

# get experiment id and create output dirs
if studyname is None:
    studyname = datetime.now().strftime('%Y%m%d_%H%M%S')
print(f'studyname: {studyname}')
saveid = f'{studyname}_{seed:04d}'
print(f'saveid: {saveid}')
os.makedirs(f'results/{saveid}', exist_ok=True)

# get data
print(f'dataname: {dataname}')
data = get_data(dataname, seed)
data_outfile = f'pickles/{saveid}_data.pickle'
# pickle.dump(data, open(data_outfile, 'wb'))
# print(f'data saved to {data_outfile}')

outcomes = run_method(method, data, saveid)
evaluations = evaluate(outcomes)
