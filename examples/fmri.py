import pandas as pd


def get_data(dataname, dxgroup=None):
    infile_imgs = f'data/{dataname}/region_features.tsv'
    infile_phenos = f'data/{dataname}/phenotypes.tsv'
    imgs = pd.read_csv(infile_imgs, sep='\t', index_col=0)
    phenos = pd.read_csv(infile_phenos, sep='\t', index_col=0)
    assert imgs.shape[0] == phenos.shape[0]
    if dataname == 'abide':
        if dxgroup is not None:
            # Select one dxgroup only
            is_selected = phenos['dxgroup'] == dxgroup
            imgs = imgs.loc[is_selected]
            phenos = phenos.loc[is_selected]
        features = pd.concat(
                [phenos[['age', 'sex']], imgs],
                axis=1)
        targets = phenos[['fiq']]
    elif dataname == 'abcd':
        features = pd.concat(
                [phenos[['p', 'Age', 'Female']], imgs],
                axis=1)
        targets = phenos[['g']]
    else:
        raise ValueError('`dataname` not recognized')
    return features.to_numpy(), targets.to_numpy()
