import numpy as np
import matplotlib.pyplot as plt

from dalea.plot import show_performance


def run_mlp(
        x_train, y_train, x_test, y_test,
        outpref=None, seed=None, **kwargs):
    from sklearn.neural_network import MLPRegressor
    if 'hidden_layer_sizes' in kwargs.keys():
        hidden_layer_sizes = kwargs['hidden_layer_sizes']
    else:
        hidden_layer_sizes = (256, 256, 256, 256)
    print('mlp hidden_layer_sizes:', hidden_layer_sizes)
    print(kwargs)
    model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=seed, max_iter=500,
            **kwargs)
    model.fit(x_train, y_train[:, 0])
    y_pred_train = model.predict(x_train)[:, np.newaxis]
    y_pred_test = model.predict(x_test)[:, np.newaxis]
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    show_performance(y_pred_train, y_train, name='mlp training')
    plt.subplot(1, 2, 2)
    show_performance(y_pred_test, y_test, name='mlp testing')
    if outpref is not None:
        plt.savefig(outpref + '_mlp.png', dpi=300)
        plt.close()
    return y_pred_train, y_pred_test


def run_forest(x_train, y_train, x_test, y_test, outpref=None, seed=None):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=seed)
    model.fit(x_train, y_train[:, 0])
    y_pred_train = model.predict(x_train)[:, np.newaxis]
    y_pred_test = model.predict(x_test)[:, np.newaxis]
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    show_performance(y_pred_train, y_train, name='forest training')
    plt.subplot(1, 2, 2)
    show_performance(y_pred_test, y_test, name='forest testing')
    if outpref is None:
        plt.show()
    else:
        plt.savefig(outpref + '_forest.png', dpi=300)
        plt.close()


def run_lasso(x_train, y_train, x_test, y_test, outpref=None, seed=None):
    from sklearn.linear_model import LassoCV
    model = LassoCV(max_iter=int(1e4))
    model.fit(x_train, y_train[:, 0])
    y_pred_train = model.predict(x_train)[:, np.newaxis]
    y_pred_test = model.predict(x_test)[:, np.newaxis]
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    show_performance(y_pred_train, y_train, name='lasso training')
    plt.subplot(1, 2, 2)
    show_performance(y_pred_test, y_test, name='lasso testing')
    if outpref is None:
        plt.show()
    else:
        plt.savefig(outpref + '_lasso.png', dpi=300)
        plt.close()


def run_svm(x_train, y_train, x_test, y_test, outpref=None, seed=None):
    from sklearn import svm
    model = svm.SVR()
    model.fit(x_train, y_train[:, 0])
    y_pred_train = model.predict(x_train)[:, np.newaxis]
    y_pred_test = model.predict(x_test)[:, np.newaxis]
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    show_performance(y_pred_train, y_train, name='svm training')
    plt.subplot(1, 2, 2)
    show_performance(y_pred_test, y_test, name='svm testing')
    if outpref is None:
        plt.show()
    else:
        plt.savefig(outpref + '_svm.png', dpi=300)
        plt.close()


def run_neigh(x_train, y_train, x_test, y_test, outpref=None, seed=None):
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor()
    model.fit(x_train, y_train[:, 0])
    y_pred_train = model.predict(x_train)[:, np.newaxis]
    y_pred_test = model.predict(x_test)[:, np.newaxis]
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    show_performance(y_pred_train, y_train, name='neigh training')
    plt.subplot(1, 2, 2)
    show_performance(y_pred_test, y_test, name='neigh testing')
    if outpref is None:
        plt.show()
    else:
        plt.savefig(outpref + '_neigh.png', dpi=300)
        plt.close()


def run_all_methods(
        x_train, y_train, x_test, y_test,
        outpref=None, seed=None, **kwargs):
    run_lasso(x_train, y_train, x_test, y_test, outpref=outpref, seed=seed)
    run_svm(x_train, y_train, x_test, y_test, outpref=outpref, seed=seed)
    run_neigh(x_train, y_train, x_test, y_test, outpref=outpref, seed=seed)
    run_forest(x_train, y_train, x_test, y_test, outpref=outpref, seed=seed)
    run_mlp(
            x_train, y_train, x_test, y_test,
            outpref=outpref, seed=seed, **kwargs)
