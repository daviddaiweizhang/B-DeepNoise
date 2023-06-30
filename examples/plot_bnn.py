import pickle
import numpy as np
from dnne_experiment import get_data
from utils.visual import plot_density


def main():
    dataset = 'features1-uniform-heteroscedastic-ntrain4000'
    prefix = (
            'results/bnn/bnn-simulations/'
            f'simulate_{dataset}_1000burnin_0001/')

    data_train = get_data(dataset=dataset[:-10]+'ntrain2000', split=0)
    x_train = data_train['x_train']
    y_train = data_train['y_train']

    data_test_file = f'{prefix}data.pickle'
    data_test = pickle.load(open(data_test_file, 'rb'))
    x_test = data_test['x_train']
    y_test = data_test['y_train']

    pred_test_file = f'{prefix}posteriors_train.pickle'
    pred_test = pickle.load(open(pred_test_file, 'rb'))
    y_mean_test = pred_test['mean']
    y_std_test = pred_test['std']

    idx_uniq = np.unique(x_test, return_index=True)[1]
    x_test = x_test[idx_uniq]
    y_test = y_test[idx_uniq]
    y_mean_test = y_mean_test[:, idx_uniq]
    y_std_test = y_std_test[:, idx_uniq]

    plot_density(
            y_mean_test, y_std_test,
            x_test, x_train, y_train,
            n_samples=50, prefix=prefix)


if __name__ == '__main__':
    main()
