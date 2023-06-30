import os
import pickle

import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from umap import UMAP
from sklearn.preprocessing import OneHotEncoder


toy_dataset_names = ['moons', 'circles', 'blobs']
gray_dataset_names = ['mnist', 'fashion', 'letters']
rgb_dataset_names = ['cifar10', 'cifar100', 'lsun', 'svhn']
vision_dataset_names = gray_dataset_names + rgb_dataset_names
dataset_names = toy_dataset_names + vision_dataset_names
dataset_names = toy_dataset_names + vision_dataset_names


def is_categorical_dataset(name):
    names = name.split('-')
    return all([na in dataset_names for na in names])


def get_dataset_vision(dataset_name, train, transform=None):

    if transform is None:
        transform = T.Compose([
            T.ToTensor(),
            ])
    elif dataset_name in gray_dataset_names:
        transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.tile(3, 1, 1)),
            transform])

    root = f'data/vision/{dataset_name}/raw/'
    kwargs = dict(
            root=root, train=train, download=True,
            transform=transform)
    if dataset_name not in vision_dataset_names:
        raise ValueError('dataset name not recognized')
    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST(**kwargs)
    elif dataset_name == 'fashion':
        dataset = torchvision.datasets.FashionMNIST(**kwargs)
    elif dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(**kwargs)
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(**kwargs)
    elif dataset_name == 'lsun':
        del kwargs['train'], kwargs['download']
        kwargs['classes'] = 'test'
        dataset = torchvision.datasets.LSUN(**kwargs)
    elif dataset_name == 'svhn':
        del kwargs['train'],
        kwargs['split'] = 'train' if train else 'test'
        dataset = torchvision.datasets.SVHN(**kwargs)
    elif dataset_name == 'letters':
        kwargs['split'] = 'letters'
        dataset = torchvision.datasets.EMNIST(**kwargs)

    return dataset


class PretrainedNet():

    def __init__(self):
        weights = ResNet50_Weights.DEFAULT
        self.preprocess = weights.transforms()
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model.cuda()
        model.eval()
        self.model = model

    def __call__(self, x):
        x = x.cuda()
        y = self.model(x)
        y = y.cpu().detach().numpy()
        return y


def convert_dataset_vision(dataset, n_samples, reducer=None):
    if n_samples is None:
        n_samples = len(dataset)
    assert n_samples <= len(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True, drop_last=True)

    # extract features using pretrained models
    if reducer is not None:
        xy = [(reducer(x), y.detach().numpy()) for x, y in dataloader]
    else:
        xy = [(x.detach().numpy(), y.detach().numpy()) for x, y in dataloader]
    x = np.concatenate([s[0] for s in xy])
    y = np.concatenate([s[1] for s in xy])

    # flatten images
    x = x.reshape(x.shape[0], -1)

    # convert labels to real numbers
    y = y[..., np.newaxis]
    encoder = OneHotEncoder(sparse=False, dtype=np.float32)
    y = encoder.fit_transform(y)
    y = y.astype(np.float32)

    return x, y


def get_embeddings(x_train, x_test):
    reducer = UMAP(random_state=0, verbose=True)
    x_all = np.concatenate([x_train, x_test])
    x_all = reducer.fit_transform(x_all)
    n_train = x_train.shape[0]
    x_train = x_all[:n_train]
    x_test = x_all[n_train:]
    return x_train, x_test


def mix_datasets(name_0, name_1):
    data_0 = get_data(name_0)
    data_1 = get_data(name_1)
    (x_train_0, y_train_0), (x_test_0, y_test_0) = data_0
    __, (x_test_1, y_test_1) = data_1
    x_train, y_train = x_train_0, y_train_0
    x_test = np.concatenate([x_test_0, x_test_1])
    y_test_1 = np.full(y_test_1.shape[:1] + y_test_0.shape[1:], -1.0)
    y_test = np.concatenate([y_test_0, y_test_1])
    return (x_train, y_train), (x_test, y_test)


def get_data(dataset_name, extract_features=True, reduce_dim=False):

    n_train = None
    n_test = None

    torch.manual_seed(0)
    np.random.seed(0)

    if '-' in dataset_name:
        dnames = dataset_name.split('-')
        data = mix_datasets(*dnames)
    elif dataset_name in vision_dataset_names:
        features_file = f'data/vision/{dataset_name}/features.pickle'
        if os.path.exists(features_file):
            data = pickle.load(open(features_file, 'rb'))
            print(f'Features loaded from {features_file}.')
        else:
            if extract_features:
                reducer = PretrainedNet()
                transform = reducer.preprocess
            else:
                transform = None
            dataset_train = get_dataset_vision(
                    dataset_name, train=True, transform=transform)
            dataset_test = get_dataset_vision(
                    dataset_name, train=False, transform=transform)
            x_train, y_train = convert_dataset_vision(
                    dataset_train, n_train, reducer=reducer)
            x_test, y_test = convert_dataset_vision(
                    dataset_test, n_test, reducer=reducer)
            if reduce_dim:
                x_train, x_test = get_embeddings(x_train, x_test)
            data = (x_train, y_train), (x_test, y_test)
            pickle.dump(data, open(features_file, 'wb'))
            print(f'Features saved to {features_file}.')
    else:
        if dataset_name == 'moons':
            n_train, n_test = 500, 500
            x, y = make_moons(
                    n_samples=n_train+n_test, noise=0.2)
        elif dataset_name == 'circles':
            n_train, n_test = 500, 500
            x, y = make_circles(
                    n_samples=n_train+n_test,
                    noise=0.2, factor=0.5)
        elif dataset_name == 'blobs':
            n_train, n_test = 500, 500
            x, y = make_blobs(
                    n_samples=n_train+n_test, centers=7)
        if y.ndim == 1:
            y = y[..., np.newaxis]
        encoder = OneHotEncoder(sparse=False, dtype=np.float32)
        y = encoder.fit_transform(y)
        x, y = x.astype(np.float32), y.astype(np.float32)
        x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=n_test)
        data = (x_train, y_train), (x_test, y_test)
    return data
