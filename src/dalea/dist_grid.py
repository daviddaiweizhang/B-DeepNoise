from collections import namedtuple
import os

import numpy as np
from scipy import stats
import pickle

from dalea.numeric import normal_log_prob_safe, normal_log_den


gridfile_log_prob = 'data/dist_grids/normal_grid_log_prob.pickle'
gridfile_inv_cdf = 'data/dist_grids/normal_grid_inv_cdf.pickle'
# gridfile_log_den = 'data/dist_grids/normal_grid_log_den.pickle'

Grid = namedtuple('Grid', ['value', 'lower', 'upper', 'step', 'info'])


def xtoi(x, lower, upper, step, clip):
    if clip:
        x = np.clip(x, lower, upper)
    else:
        if np.min(x) < lower:
            raise ValueError('`x` must be no less than `lower`')
        if np.max(x) > upper:
            raise ValueError('`x` must be no greater than `upper`')
    return np.round((x - lower) / step).astype(int)


def grid_arr(lower, upper, step):
    x = np.arange(lower, upper+step, step)
    # assert (x == x[xtoi(x=x, step=step, lower=lower)]).all()
    return x


def make_normal_log_prob_matrix(z):
    lower = np.tile(z, [len(z), 1]).T
    upper = np.tile(z, [len(z), 1])
    lower[np.tril_indices(len(z), -1)] = np.nan
    upper[np.tril_indices(len(z), -1)] = np.nan
    log_prob = normal_log_prob_safe(lower, upper)[0]
    assert np.isfinite(log_prob[np.triu_indices(len(z), 1)]).all()
    assert (np.diag(log_prob) == -np.inf).all()
    # np.fill_diagonal(log_prob, -np.inf)
    return log_prob


def make_normal_log_prob_grid(z_lower, z_upper, z_step, outfile=None):
    log_prob = make_normal_log_prob_matrix(grid_arr(
        lower=z_lower,
        upper=z_upper,
        step=z_step))
    grid = Grid(
        value=log_prob,
        lower=z_lower,
        upper=z_upper,
        step=z_step,
        info=dict(ndim=2))
    if outfile is not None:
        pickle.dump(grid, open(outfile, 'wb'))
        print(outfile)
    return grid


def make_normal_log_den_grid(z_lower, z_upper, z_step, outfile=None):
    log_den = normal_log_den(grid_arr(
        lower=z_lower,
        upper=z_upper,
        step=z_step))
    grid = Grid(
        value=log_den,
        lower=z_lower,
        upper=z_upper,
        step=z_step,
        info=dict(ndim=1))
    if outfile is not None:
        pickle.dump(grid, open(outfile, 'wb'))
        print(outfile)
    return grid


def make_normal_inv_cdf_grid(u_lower, u_upper, u_step, outfile=None):
    inv_cdf = stats.norm.ppf(grid_arr(
        lower=u_lower,
        upper=u_upper,
        step=u_step))
    finite_min = inv_cdf[np.isfinite(inv_cdf)].min()
    finite_max = inv_cdf[np.isfinite(inv_cdf)].max()
    assert finite_min < 0
    assert finite_max > 0
    grid = Grid(
        value=inv_cdf,
        lower=u_lower,
        upper=u_upper,
        step=u_step,
        info=dict(
            ndim=1,
            finite_min=finite_min,
            finite_max=finite_max
            ))
    if outfile is not None:
        pickle.dump(grid, open(outfile, 'wb'))
        print(outfile)
    return grid


def grid_value_2d(z0, z1, grid, clip):
    i_lower = xtoi(z0, grid.lower, grid.upper, grid.step, clip)
    i_upper = xtoi(z1, grid.lower, grid.upper, grid.step, clip)
    return grid.value[i_lower, i_upper]


def grid_value(z, grid, clip):
    i = xtoi(z, grid.lower, grid.upper, grid.step, clip)
    return grid.value[i]


def make_grids():
    if os.path.exists(gridfile_log_prob):
        grid_log_prob = pickle.load(open(gridfile_log_prob, 'rb'))
    else:
        grid_log_prob = make_normal_log_prob_grid(
                z_lower=-50,
                z_upper=50,
                z_step=1e-2,
                outfile=gridfile_log_prob)

    if os.path.exists(gridfile_inv_cdf):
        grid_inv_cdf = pickle.load(open(gridfile_inv_cdf, 'rb'))
    else:
        grid_inv_cdf = make_normal_inv_cdf_grid(
                u_lower=0,
                u_upper=1,
                u_step=1e-4,
                outfile=gridfile_inv_cdf)

    # # less efficient than direct computation
    # # since log_den of normal is (almost) only a quadratic func
    # make_normal_log_den_grid(
    #         z_lower=-50,
    #         z_upper=50,
    #         z_step=1e-2,
    #         outfile=gridfile_log_den)

    return grid_log_prob, grid_inv_cdf


grid_log_prob, grid_inv_cdf = make_grids()


# TODO: add linear interpolation
def normal_log_prob_grid_standard(z_lower, z_upper, grid=grid_log_prob):
    return grid_value_2d(z_lower, z_upper, grid, clip=True)


# TODO: add linear interpolation
def normal_inv_cdf_grid_standard(u, grid=grid_inv_cdf, return_info=False):
    z = grid_value(u, grid, clip=False)
    if return_info:
        out = (z, grid.info)
    else:
        out = z
    return out


def normal_log_prob_grid(x_lower, x_upper, mean=0.0, std=1.0):
    z_lower = (x_lower - mean) / std
    z_upper = (x_upper - mean) / std
    log_prob = normal_log_prob_grid_standard(z_lower, z_upper)
    return log_prob


def normal_inv_cdf_grid(u, mean=0.0, std=1.0, return_info=False):
    z, info = normal_inv_cdf_grid_standard(u, return_info=True)
    x = z * std + mean
    if return_info:
        out = (x, info)
    else:
        out = x
    return out
