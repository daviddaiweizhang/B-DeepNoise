from sklearn.model_selection import train_test_split

from dalea.make_data import simulate_data
import fmri
import uci


def get_data(dataname, seed):

    """ Retrieve or generate data for experiments

    Args:
        dataname: A string to specify the name of the dataset.
            It must be one of the following.
            For UCI data: `bostonHousing`, `concrete`, `energy`,
            `kin8nm`, `naval-propulsion-plant`, `power-plant`,
            `protein-tertiary-structure`, `wine-quality-red`,
            `yacht`.
            For details, see `Probabilistic Backpropagation for
            Scalable Learning of Bayesian Neural Networks`
            by Hernandez-Lobato and Adams (2015).
            For neuroimaging data: `abide`, `abcd`.
            For synthesized data: `normal`, `chisq`, `multimodal`, `mixture`.
        seed: An integer to set the random seed of Numpy.
    Returns:
        A dictionary of data, including
            `x_train`: training features
            `x_test`: testing features
            `y_train`: training targets
            `y_test`: testing targets
            `y_mean_train`: true mean function of training data
                (for synthesized datasets only)
            `y_mean_test`: true mean function of testing data
                (for synthesized datasets only)
            `y_std_train`: true standard deviation function
                of training data (for synthesized datasets only)
            `y_std_test`: true standard deviation function
                of testing data (for synthesized datasets only)
            `seed`: random seed used for generating the data
            `is_real`: boolean value indicating whether the data
                is real (or synthesized)
            `target_type`: A string of `continuous` or `categorical`
    """

    uci_dataset_list = [
            'bostonHousing',
            'concrete',
            'energy',
            'kin8nm',
            'naval-propulsion-plant',
            'power-plant',
            'protein-tertiary-structure',
            'wine-quality-red',
            'yacht']
    if dataname in uci_dataset_list:
        datagroup = 'uci'
    elif dataname in ['abcd', 'abide']:
        datagroup = 'fmri'
    elif dataname in ['normal', 'chisq', 'multimodal', 'mixture']:
        datagroup = 'simulated'
    else:
        raise ValueError('`dataname` not recognized')
    data_is_real = datagroup in ['uci', 'fmri']

    target_type = 'continuous'
    if data_is_real:
        if datagroup == 'fmri':
            x_all, y_all = fmri.get_data(dataname)
        elif datagroup == 'uci':
            x_all, y_all = uci.get_data(dataname)
        train_size = 0.75
        x_train, x_test, y_train, y_test = train_test_split(
                x_all, y_all,
                train_size=train_size, random_state=seed)
        n_observations_train = x_train.shape[0]
        n_observations_test = x_test.shape[0]
        y_mean_train = y_mean_test = None
        y_std_train = y_std_test = None
    else:
        n_observations_train = 200
        n_observations_test = 2000
        (
                x_train, x_test,
                y_train, y_test,
                y_mean_train, y_mean_test,
                y_std_train, y_std_test) = simulate_data(
                n_observations_train, n_observations_test,
                target_type, dataname)
    print(f'n_observations_train: {n_observations_train}')
    print(f'n_observations_test: {n_observations_test}')
    data = {}
    data['x_train'] = x_train
    data['x_test'] = x_test
    data['y_train'] = y_train
    data['y_test'] = y_test
    data['y_mean_train'] = y_mean_train
    data['y_mean_test'] = y_mean_test
    data['y_std_train'] = y_std_train
    data['y_std_test'] = y_std_test
    data['seed'] = seed
    data['is_real'] = data_is_real
    data['target_type'] = target_type

    return data
