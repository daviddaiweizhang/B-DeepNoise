#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
from dalea.make_data import simulate_data

noisetype = 'normal' if len(sys.argv) <= 1 else sys.argv[1]
seed = 1

n_observations_train = 200
n_observations_test = 2000
data = simulate_data(
        n_observations_train, n_observations_test,
        target_type='continuous', noise_type=noisetype,
        seed=seed)
x_train = data[0][:, 0]
x_test = data[1][:, 0]
y_train = data[2][:, 0]
y_test = data[3][:, 0]
y_mean_train = data[4][:, 0]
y_mean_test = data[5][:, 0]

title_dict = {
        'normal': 'Data with Gaussian noise',
        'chisq': 'Data with chi-squared noise',
        'multimodal': 'Data with Gaussian mixture noise'}

plt.rcParams.update({'font.size': 10})
markersize = 3
linewidth = markersize * 0.5
plt.figure(figsize=(8, 3))
plt.plot(
        x_test, y_test, 'o', alpha=0.2, color='tab:blue',
        label='testing samples', markersize=markersize)
plt.plot(
        x_train, y_train, 's', alpha=0.7, color='tab:olive',
        label='training samples', markersize=markersize)
plt.plot(
        x_test, y_mean_test, color='tab:red', linewidth=linewidth,
        label='true mean function')
# plt.title(title_dict[noisetype])
# plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left', markerscale=markersize)
plt.tick_params(axis='x', direction='in', pad=-15)

outfile = f'tablesfigures/{noisetype}/design.png'
plt.savefig(outfile, dpi=200, bbox_inches='tight')
print(outfile)
