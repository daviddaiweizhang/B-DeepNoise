import pandas as pd
from make_splits import save_data, save_split_index

prefix = 'data/neuroimaging/abcd/'
data = pd.read_csv(
        f'{prefix}/data.tsv', sep='\t', header=0, index_col=0)
targets = data[['g']]
features = data.drop(columns=['sitenum', 'g'])
save_data(x=features.to_numpy(), y=targets.to_numpy(), prefix=prefix)
outfile = prefix+'label_features.txt'
features.columns.to_series().to_csv(outfile, header=False, index=False)
outfile = prefix+'label_target.txt'
targets.columns.to_series().to_csv(outfile, header=False, index=False)
save_split_index(
        n_observations=features.shape[0],
        prop_test=0.1, n_splits=20,
        prefix=prefix)
