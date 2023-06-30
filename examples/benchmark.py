from time import time
import pickle
import os

import numpy as np
from scipy import stats
import tensorflow_probability as tfp
import tensorflow as tf

from dalea.numeric import (
        # normal_log_prob_safe,
        normal_log_den,
        normal_cdf_approx,
        normal_inv_cdf_approx,
        )
from dalea.dist_grid import (
        normal_log_prob_grid,
        normal_inv_cdf_grid,
        make_normal_log_den_grid,
        grid_value,
        )

tfd = tfp.distributions
norm_rv = tfd.Normal(
        loc=tf.Variable(0.0, dtype=tf.float64),
        scale=1.0)
norm_cdf_bij = tfp.bijectors.NormalCDF()


def benchmark_log_prob(n, m, gridfile):

    u_lower = np.random.rand(n)
    u_upper = np.random.rand(n) * (1 - u_lower) + u_lower
    z_lower = stats.norm.ppf(u_lower)
    z_upper = stats.norm.ppf(u_upper)

    # grid = pickle.load(open(gridfile, 'rb'))

    print('===log prob=====')

    # # Too slow
    # t_start = time()
    # for __ in range(m):
    #     log_prob = normal_log_prob_safe(z_lower, z_upper)[0]
    # runtime = (time() - t_start) / m
    # infrate = 1-np.isfinite(log_prob).mean()
    # accuracy = np.quantile(np.abs(log_prob / log_prob_truth - 1), 0.99)
    # print(f'log1mexp: {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    t_start = time()
    for __ in range(m):
        log_prob = np.log(
                stats.norm.cdf(z_upper)
                - stats.norm.cdf(z_lower))
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(log_prob).mean()
    log_prob_truth = log_prob
    accuracy = np.quantile(np.abs(log_prob / log_prob_truth - 1), 0.99)
    print(f'scipy   : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    t_start = time()
    for __ in range(m):
        log_prob = tf.math.log(
                norm_rv.cdf(z_upper)
                - norm_rv.cdf(z_lower)).numpy()
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(log_prob).mean()
    accuracy = np.quantile(np.abs(log_prob / log_prob_truth - 1), 0.99)
    print(f'tfprv   : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    t_start = time()
    for __ in range(m):
        log_prob = np.log(
                normal_cdf_approx(z_upper)
                - normal_cdf_approx(z_lower))
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(log_prob).mean()
    accuracy = np.quantile(np.abs(log_prob / log_prob_truth - 1), 0.99)
    print(f'approx  : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    t_start = time()
    for __ in range(m):
        # log_prob = grid_value_2d(z_lower, z_upper, grid)
        log_prob = normal_log_prob_grid(z_lower, z_upper)
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(log_prob).mean()
    accuracy = np.quantile(np.abs(log_prob / log_prob_truth - 1), 0.99)
    print(f'grid    : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')


def benchmark_log_den(n, m, gridfile):

    z = np.random.randn(n)

    grid = pickle.load(open(gridfile, 'rb'))

    print('===log den======')

    t_start = time()
    for __ in range(m):
        log_den = stats.norm.logpdf(z)
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(log_den).mean()
    log_den_truth = log_den
    accuracy = np.quantile(np.abs(log_den / log_den_truth - 1), 0.99)
    print(f'scipy   : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    t_start = time()
    for __ in range(m):
        log_den = norm_rv.log_prob(z).numpy()
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(log_den).mean()
    accuracy = np.quantile(np.abs(log_den / log_den_truth - 1), 0.99)
    print(f'tfprv   : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    t_start = time()
    for __ in range(m):
        log_den = normal_log_den(z)
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(log_den).mean()
    accuracy = np.quantile(np.abs(log_den / log_den_truth - 1), 0.99)
    print(f'numpy   : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    t_start = time()
    for __ in range(m):
        log_den = grid_value(z, grid, clip=False)
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(log_den).mean()
    accuracy = np.quantile(np.abs(log_den / log_den_truth - 1), 0.99)
    print(f'grid    : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')


def benchmark_inv_cdf(n, m, gridfile):

    u = np.random.rand(n)

    # grid = pickle.load(open(gridfile, 'rb'))

    print('===inv cdf======')

    t_start = time()
    for __ in range(m):
        inv_cdf = stats.norm.ppf(u)
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(inv_cdf).mean()
    inv_cdf_truth = inv_cdf
    accuracy = np.quantile(np.abs(inv_cdf / inv_cdf_truth - 1), 0.99)
    print(f'scipyqt : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    t_start = time()
    for __ in range(m):
        inv_cdf = norm_rv.quantile(u).numpy()
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(inv_cdf).mean()
    accuracy = np.quantile(np.abs(inv_cdf / inv_cdf_truth - 1), 0.99)
    print(f'tfpqt   : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    t_start = time()
    for __ in range(m):
        inv_cdf = norm_cdf_bij.inverse(u).numpy()
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(inv_cdf).mean()
    accuracy = np.quantile(np.abs(inv_cdf / inv_cdf_truth - 1), 0.99)
    print(f'tfpbij  : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    t_start = time()
    for __ in range(m):
        inv_cdf = normal_inv_cdf_approx(u)
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(inv_cdf).mean()
    accuracy = np.quantile(np.abs(inv_cdf / inv_cdf_truth - 1), 0.99)
    print(f'approx  : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    t_start = time()
    for __ in range(m):
        # inv_cdf = grid_value(u, grid)
        inv_cdf = normal_inv_cdf_grid(u)
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(inv_cdf).mean()
    accuracy = np.quantile(np.abs(inv_cdf / inv_cdf_truth - 1), 0.99)
    print(f'grid    : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')


def benchmark_sample_trunc_norm(n, m, gridfile):

    u_lo = np.random.rand(n)
    u_up = np.random.rand(n) * (1 - u_lo) + u_lo
    x_lower = stats.norm.ppf(u_lo)
    x_upper = stats.norm.ppf(u_up)
    del u_lo, u_up

    print('=sample tn======')

    # # Too slow
    # t_start = time()
    # for __ in range(m):
    #     x = stats.truncnorm.rvs(a=x_lower, b=x_upper, size=n)
    # runtime = (time() - t_start) / m
    # infrate = 1-np.isfinite(x).mean()
    # accuracy = stats.ks_2samp(x, x_truth)[0]
    # print(f'scipyrv : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    tn_rv = tfd.TruncatedNormal(
            tf.Variable(0.0, dtype=tf.float64),
            1.0, x_lower, x_upper)
    t_start = time()
    for __ in range(m):
        x = tn_rv.sample().numpy()
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(x).mean()
    x_truth = x
    accuracy = stats.ks_2samp(x, x_truth)[0]
    print(f'tfprv   : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    u_lower = stats.norm.cdf(x_lower)
    u_upper = stats.norm.cdf(x_upper)
    t_start = time()
    for __ in range(m):
        u = np.random.rand(n) * (u_upper - u_lower) + u_lower
        x = stats.norm.ppf(u)
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(x).mean()
    accuracy = stats.ks_2samp(x, x_truth)[0]
    print(f'scipyqt : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    u_lower = stats.norm.cdf(x_lower)
    u_upper = stats.norm.cdf(x_upper)
    t_start = time()
    for __ in range(m):
        u = np.random.rand(n) * (u_upper - u_lower) + u_lower
        x = norm_rv.quantile(u).numpy()
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(x).mean()
    accuracy = stats.ks_2samp(x, x_truth)[0]
    print(f'tfpqt   : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    u_lower = stats.norm.cdf(x_lower)
    u_upper = stats.norm.cdf(x_upper)
    t_start = time()
    for __ in range(m):
        u = np.random.rand(n) * (u_upper - u_lower) + u_lower
        x = norm_cdf_bij.inverse(u).numpy()
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(x).mean()
    accuracy = stats.ks_2samp(x, x_truth)[0]
    print(f'tfpbij  : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    u_lower = stats.norm.cdf(x_lower)
    u_upper = stats.norm.cdf(x_upper)
    t_start = time()
    for __ in range(m):
        u = np.random.rand(n) * (u_upper - u_lower) + u_lower
        x = normal_inv_cdf_approx(u)
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(x).mean()
    accuracy = stats.ks_2samp(x, x_truth)[0]
    print(f'approx  : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')

    # grid = pickle.load(open(gridfile, 'rb'))
    u_lower = stats.norm.cdf(x_lower)
    u_upper = stats.norm.cdf(x_upper)
    t_start = time()
    for __ in range(m):
        u = np.random.rand(n) * (u_upper - u_lower) + u_lower
        # x = grid_value(u, grid)
        x = normal_inv_cdf_grid(u)
    runtime = (time() - t_start) / m
    infrate = 1-np.isfinite(x).mean()
    accuracy = stats.ks_2samp(x, x_truth)[0]
    print(f'grid    : {runtime:.6f} {accuracy:.6f} {infrate:.6f}')


def run_benchmarks():

    # grids for log_prob and inv_cdf have already been computed
    gridfile_log_den = 'data/dist_grids/normal_grid_log_den.pickle'
    if not os.path.exists(gridfile_log_den):
        make_normal_log_den_grid(
                z_lower=-50,
                z_upper=50,
                z_step=1e-2,
                outfile=gridfile_log_den)
    for p in range(3, 8):
        n = int(10**p)
        print(f'\n%%%%%%%%%%%%%%%%n = 10^{p}%%%%%%%%%%%%%%%%%%')
        benchmark_log_prob(n=n, m=100, gridfile=None)
        benchmark_inv_cdf(n=n, m=100, gridfile=None)
        benchmark_sample_trunc_norm(n=n, m=100, gridfile=None)
        benchmark_log_den(n=n, m=100, gridfile=gridfile_log_den)


def main():
    run_benchmarks()


if __name__ == '__main__':
    main()
