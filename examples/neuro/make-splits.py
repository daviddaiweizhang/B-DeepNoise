import sys
import pandas as pd
import numpy as np

prefix = sys.argv[1]

prop_train = 0.8
n_splits = 20
seed = 0
np.random.seed(seed)
infile = f'{prefix}features.tsv'
outfile = f'{prefix}splits.tsv'
data = pd.read_csv(infile, sep='\t', header=0, index_col=0)
n_observations = data.shape[0]
n_train = int(n_observations * prop_train)
splits = {}
for i in range(n_splits):
    idx = np.random.choice(n_observations, n_train, replace=False)
    splits[f'split_{i}'] = idx
splits = pd.DataFrame(splits)
splits.to_csv(outfile, sep='\t', index=False)
print(outfile)
