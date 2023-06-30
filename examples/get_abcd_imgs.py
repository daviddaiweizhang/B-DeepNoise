import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.stats import norm as norm_rv
import seaborn as sns


def get_nii(x, affine):
    x_finite = x.copy()
    x_finite[~np.isfinite(x_finite)] = 0
    nii = nib.Nifti1Image(x_finite, affine=affine)
    return nii


def plot_brain(x, affine, mode, outfile, **kwargs):
    nii = get_nii(x, affine)
    if mode == 'roi':
        display = plotting.plot_roi(nii, **kwargs)
    elif mode == 'stat_map':
        display = plotting.plot_stat_map(nii, **kwargs)
    print(outfile)
    display.savefig(outfile, dpi=300)
    display.close()


def get_networks(prefix):
    infile = f'{prefix}AAL_region_functional_networks.csv'
    df = pd.read_csv(infile, header=0)
    df.drop(columns='region_code', inplace=True)
    df.rename(columns={
        'Number': 'region_number',
        'AAL Region Names': 'region_name'},
        inplace=True)
    df.columns = df.columns.str.lower().str.replace('.', ' ').str.strip()
    return df


def get_masks(prefix):
    isdatfile = f'{prefix}mask_isdat.nii.gz'
    regionfile = f'{prefix}mask_region.nii.gz'
    isdat = nib.load(isdatfile).get_fdata().astype(bool)
    region_info = nib.load(regionfile)
    affine = region_info.affine
    region_number = region_info.get_fdata().astype(int)
    network_full = get_networks(prefix)

    region_dict = dict(zip(
        network_full['region_number'],
        network_full['region_name']))
    region_dict[0] = 'NA'
    region_name = np.vectorize(region_dict.get)(region_number)

    network = network_full.loc[
            network_full['region_number'] < region_number.max()]
    network.drop(columns='region_number', inplace=True)
    network.set_index('region_name', inplace=True)
    network.drop(columns=network.columns[network.sum() == 0], inplace=True)
    return isdat, region_name, network, affine


def extract_region_features(images, regions):
    return pd.DataFrame({
        regi: images[:, regions == regi].mean(-1)
        for regi in np.unique(regions)})


def get_data():
    # get covariates and subject ids
    prefix = 'data/neuroimaging/abcd/'
    infile_covariates = f'{prefix}covariates.tsv'
    covariates = pd.read_table(infile_covariates, header=0, index_col=0)

    # load images
    id_list = list(covariates.index)
    infiles_images = [
            f'{prefix}raw/3mm_2bk/3mm_{id}_2bk-baseline_con.nii.gz'
            for id in id_list]
    images, affine = get_imgs(infiles_images)
    # img_shape = images.shape[1:]

    # center images
    images -= images.mean(0)

    # get region info
    regions = get_masks(prefix+'raw/')[1]
    is_dat = regions != 'NA'

    # remove voxels outside regions
    images = images[:, is_dat]
    regions = regions[is_dat]

    # remove baseline fluxtuation
    images = images - images.mean(-1, keepdims=True)

    # get means of regions as features
    features = extract_region_features(images, regions)
    features.index = covariates.index

    # standardize data
    data = covariates.merge(features, left_index=True, right_index=True)
    data = standardize(data)

    # save data
    outfile = f'{prefix}data.tsv'
    data.to_csv(outfile, sep='\t', float_format='%.3f')
    print(outfile)


def get_imgs(infile_list):
    imgs = np.stack([
        nib.load(infile).get_fdata()
        for infile in infile_list])
    affine = nib.load(infile_list[0]).affine
    return imgs, affine


def standardize_qqnorm(x):
    q = (x.rank() - 0.5) / x.shape[0]
    y = q.apply(norm_rv.ppf)
    return y


def standardize_minmax(x, lower=-1, upper=1):
    x = x - x.min()
    x = x / x.max()
    x = x * (upper - lower) + lower
    return x


def standardize(data):
    bounded = [
            'Age', 'Female', 'HighestParentalEducation',
            'HouseholdMaritalStatus', 'HouseholdIncome',
            'RaceEthnicityWhite',
            'RaceEthnicityHispanic',
            'RaceEthnicityBlack',
            'RaceEthnicityAsian',
            ]
    data[bounded] = standardize_minmax(data[bounded])
    to_normalize = data.columns.difference(['sitenum'] + bounded)
    data[to_normalize] = standardize_qqnorm(data[to_normalize])
    data[to_normalize] = standardize_minmax(data[to_normalize])
    return data


def plot_data():
    prefix = 'data/neuroimaging/abcd/'
    infile = f'{prefix}data.tsv'
    x = pd.read_csv(infile, sep='\t', header=0, index_col=0)
    order = x.corr()['g'].to_numpy().argsort()[::-1]
    x = x.iloc[:, order]

    plt.figure(figsize=(16, 14))
    sns.heatmap(
            x.corr(), vmin=-1, vmax=1, cmap='vlag',
            xticklabels=True, yticklabels=True)
    outfile = f'{prefix}pilot/000corrmat.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(outfile)

    for i, col in enumerate(x.columns, 1):
        plt.figure(figsize=(8, 8))
        plt.plot(x[col], x['g'], 'o', alpha=0.2)
        corr, pval = kendalltau(x[col], x['g'])
        pval = -np.log10(pval)
        plt.title(f'corr: {corr:.2f}, -log10 pval: {pval:.1f}')
        plt.xlabel(col)
        plt.ylabel('g')
        outfile = f'{prefix}pilot/{i:03d}{col}.png'
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
        print(outfile)


def main():
    get_data()
    plot_data()


if __name__ == '__main__':
    main()
