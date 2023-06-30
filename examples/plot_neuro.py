import numpy as np
import nibabel as nib
from nilearn import plotting
import pandas as pd
from get_abcd_imgs import get_masks
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_nii(x, affine):
    x_finite = x.copy()
    x_finite[~np.isfinite(x_finite)] = 0
    nii = nib.Nifti1Image(x_finite, affine=affine)
    return nii


def plot_stat_map(x, affine, prefix, **kwargs):
    nii = get_nii(standardize(x), affine)
    display = plotting.plot_stat_map(nii, annotate=False, **kwargs)
    display.annotate(size=15)
    outfile = prefix+'.png'
    display.savefig(outfile, dpi=300)
    plt.close()
    print(outfile)

    fig, ax = plt.subplots(figsize=(3, 10))
    norm = mpl.colors.Normalize(vmin=np.nanmin(x), vmax=np.nanmax(x))
    cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=kwargs['cmap']),
            cax=ax)
    cbar.ax.tick_params(labelsize=50)
    plt.subplots_adjust(left=0.03, right=0.3, bottom=0.03, top=0.97)
    outfile = prefix+'_colorbar.png'
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(outfile)


def standardize(x):
    x = x - np.nanmin(x)
    x = x / np.nanmax(x)
    x = x * 2 - 1
    return x


mask_prefix = 'data/neuroimaging/abcd/raw/'
influence_file = 'results/abcd/influence.txt'
__, regions, __, affine = get_masks(mask_prefix)
influence_df = pd.read_csv(influence_file, sep='\t', header=0, index_col=0)
influence_df = influence_df.iloc[11:]

n_regions = influence_df.shape[0]
n_top = 20

for col in influence_df.columns:
    influ = influence_df[col]
    # cutoff = influ.sort_values(ascending=False).iloc[n_top]
    # influ[influ < cutoff] = np.nan
    influ_dict = influ.to_dict()
    influ_dict['NA'] = np.nan
    influ_arr = np.vectorize(influ_dict.get)(regions)
    plot_stat_map(
            influ_arr, affine,
            display_mode='z', cut_coords=list(range(-5, 16, 5)),
            cmap='viridis', symmetric_cbar=False, colorbar=False,
            prefix=f'results/abcd/{col}')
