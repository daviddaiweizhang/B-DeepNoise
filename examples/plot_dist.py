import sys
from synthesize import get_data as get_data_synthetic
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


def get_data():
    dataset = 'features1-uniform-mixed-ntrain4000/'
    data = get_data_synthetic(dataset=dataset, split=0)
    x_quants, y_quants = data[0]
    x_test, y_test = data[2]
    x_quants, y_quants = x_quants[..., 0], y_quants[..., 0]
    x_test, y_test = x_test[..., 0], y_test[..., 0]
    return dict(
            x_quants=x_quants, y_quants=y_quants,
            x_test=x_test, y_test=y_test)


def get_transformed_single(x, y_dist):
    r = 0.2
    s = 1.8

    # generate prior samples
    y_median = np.median(y_dist, 0)
    y_min, y_max = y_dist.min(0), y_dist.max(0)
    y_min = y_median + (y_min - y_median) * s
    y_max = y_median + (y_max - y_median) * s
    y_prior = np.random.rand(*y_dist.shape) * (y_max - y_min) + y_min

    # perturb prior samples
    t = 0.3
    a = np.random.rand() * t - 0.5*t + 1
    b = np.random.rand() * t - 0.5*t
    y_prior = y_prior * a + b

    # mix prior and observed samples
    ny, nx = y_dist.shape
    ny_prior = int(r * ny)
    ny_origin = ny - ny_prior
    iy_origin = np.random.choice(ny, (ny_origin, nx))
    iy_prior = np.random.choice(ny, (ny_prior, nx))
    y_new_origin = y_dist[iy_origin, np.arange(nx)]
    y_new_prior = y_prior[iy_prior, np.arange(nx)]
    y_new = np.concatenate([y_new_origin, y_new_prior], axis=0)
    y_new.sort(0)
    return y_new


def transform_train(data):
    for i in range(len(data['y_train'])):
        data['y_train'][i] = get_transformed_single(
                data['x_train'][i], data['y_train'][i])


def get_train_single(x_quants, y_quants, nx, ny):
    nx_quants = x_quants.shape[0]
    ny_quants = y_quants.shape[0]
    ix = np.random.choice(nx_quants, nx)
    ix.sort()
    ix[0] = 0
    ix[-1] = -1
    x = x_quants[ix]
    iy = np.random.choice(ny_quants, ny*nx).reshape(ny, nx)
    y = y_quants[iy, np.tile(ix, [iy.shape[0], 1])]
    return x, y


def add_train(data, nx, ny, n_splits):
    data['x_train'] = []
    data['y_train'] = []
    for __ in range(n_splits):
        x, y = get_train_single(
            data['x_quants'], data['y_quants'], nx=nx, ny=ny)
        data['x_train'].append(x)
        data['y_train'].append(y)


def get_dist_gaussian(y, heteroscedastic, qs):
    y_mean = y.mean(0)
    axis = 0 if heteroscedastic else None
    y_std = np.square(y - y_mean).mean(axis)**0.5
    zs = norm.ppf(qs)[..., np.newaxis]
    y_dist = zs * y_std[np.newaxis] + y_mean[np.newaxis]
    return y_dist


def get_dist(y_list, mode, n_realizations):
    qs = (np.arange(n_realizations) + 0.5) / n_realizations
    y_dist_list = []
    for y in y_list:
        if mode == 'homoscedastic':
            y_dist = get_dist_gaussian(y, heteroscedastic=False, qs=qs)
        elif mode == 'heteroscedastic':
            y_dist = get_dist_gaussian(y, heteroscedastic=True, qs=qs)
        elif mode == 'nonparametric':
            y_dist = np.quantile(y, qs, axis=0)
        else:
            raise ValueError('Mode not recognized')
        y_dist_list.append(y_dist)
    return y_dist_list


def plot(data):
    plt.figure(figsize=(16, 8))
    plt.plot(
            data['x_test'], data['y_test'], 'o',
            alpha=0.5, color='tab:grey', label='observed data')
    x_train_list, y_train_list = data['x_train'], data['y_dist']
    cmap = plt.get_cmap('tab10')
    for i, (x_train, y_dist) in enumerate(zip(x_train_list, y_train_list)):
        lab = f' {i+1}' if len(x_train_list) > 1 else ''
        for j, y in enumerate(y_dist):
            label = f'Model{lab} quantiles' if j == 0 else None
            plt.plot(x_train, y, color=cmap(i), alpha=0.8, label=label)
    plt.legend(loc='upper left')
    outfile = 'a.png'
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(outfile)


def smoothen(y_dist, x):
    model = KNeighborsRegressor(n_neighbors=10)
    model.fit(x.reshape(-1, 1), y_dist.T)
    y_dist_smooth = model.predict(x.reshape(-1, 1)).T
    return y_dist_smooth


def main():
    nx_train = 500
    ny_train = 100
    n_realizations = 20

    mode = sys.argv[1]
    n_transforms = int(sys.argv[2])
    outfile = sys.argv[3]
    np.random.seed(0)
    data = get_data()
    add_train(data, nx=nx_train, ny=ny_train, n_splits=n_transforms)
    transform_train(data)
    data['y_dist'] = get_dist(
            data['y_train'], mode=mode,
            n_realizations=n_realizations)
    data['y_dist'] = [
            smoothen(y_dist, x_train)
            for y_dist, x_train in zip(data['y_dist'], data['x_train'])]
    plot(data, outfile)


if __name__ == '__main__':
    main()
