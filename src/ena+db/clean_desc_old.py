"""
This code takes the original descriptors from ena+db.desc, cleans data, and saves.
The original file is not overwritten.

Note! Some commands in this code are very specific to ena+db.desc
and required to clean the specific issues in this dataset.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from glob import glob

filepath = Path(__file__).resolve().parent

# Utils
sys.path.append(str(filepath/'../'))
from classlogger import Logger
from impute import impute_values

datapath = Path('/vol/ml/2019-nCov/drug-screening/original')
outpath  = Path('/vol/ml/2019-nCov/drug-screening')


def dropna(df, axis=0, th=0.4):
    """ Drop rows or cols based on the ratio of NA values along the axis.
    Args:
        th (float) : if the ratio of NA values along the axis is larger that th, then drop the item
        axis (int) : 0 to drop rows; 1 to drop cols
    """
    assert (axis in [0,1]), 'Invalid value for axis'
    if axis==0:
        idx = df.isna().sum(axis=axis)/df.shape[axis] <= th
        df = df.iloc[:, idx.values]
    elif axis==1:
        idx = df.isna().sum(axis=axis)/df.shape[axis] <= th
        df = df.iloc[idx.values, :].reset_index(drop=True)
    return df


t0 = time()
lg = Logger(filepath/'clean.desc.rick.log')
print_fn = lg.logger.info

print_fn('\nPython filepath    {}'.format( filepath ))
print_fn('Original data path {}'.format( datapath ))
print_fn('Output data path   {}'.format( outpath ))

# File names
in_fname = 'ena+db.desc'
in_fpath  = datapath/in_fname
out_fpath = outpath/in_fname

# Load Modred descriptors (this descriptors are the output
# from mordred run on ena+db.can from Rick)
print_fn('\nLoad {}'.format( in_fpath ))
dsc = pd.read_csv(in_fpath)
print_fn(dsc.shape)


# Drop duplicates
print_fn('\nDrop duplicates ...')
cnt0 = dsc.shape[0]; print_fn('Count: {}'.format( cnt0 ))
dsc = dsc.drop_duplicates()
cnt1 = dsc.shape[0]; print_fn('Count: {}'.format( cnt1 ));
print_fn('Dropped duplicates: {}'.format( cnt0-cnt1 ))

# Print
print_fn(dsc.iloc[:5, :4])


# Filter rows/cols based on NaN values - step 1
idx = (dsc.isna().sum(axis=1)==dsc.shape[1]).values
dsc = dsc.iloc[~idx, :]
idx = (dsc.isna().sum(axis=0)==dsc.shape[0]).values
dsc = dsc.iloc[:, ~idx]


# Filter rows/cols based on NaN values - step 2
# print(dsc.isna().sum(axis=1).sort_values().sort_values(ascending=False))
# p=dsc.isna().sum(axis=1).sort_values(ascending=False).hist(bins=100);
th = 0.2
print_fn('Drop rows (drugs) with at least {} NaN values (at least {} out of {}).'.format(
    th, int(th * dsc.shape[1]), dsc.shape[1]))
cnt0 = dsc.shape[0]; print_fn('Count: {}'.format( cnt0 ))
dsc = dropna(dsc, axis=1, th=th).reset_index(drop=True)
cnt1 = dsc.shape[0]; print_fn('Count: {}'.format( cnt1 ));
print_fn('Dropped duplicates: {}'.format( cnt0-cnt1 ))

## print(dsc.isna().sum(axis=0).sort_values(ascending=False))
## p=dsc.isna().sum(axis=0).sort_values(ascending=False).hist(bins=100);
# print_fn('Drop cols (drugs) with at least {} NaN values (at least {} out of {}).'.format(
#     th, int(th * dsc.shape[0]), dsc.shape[0]))
# th = 0.1
# cnt0 = dsc.shape[1]; print_fn('Count: {}'.format( cnt0 ))
# dsc = dropna(dsc, axis=0, th=th)
# cnt1 = dsc.shape[1]; print_fn('Count: {}'.format( cnt1 ));
# print_fn('Dropped duplicates: {}'.format( cnt0-cnt1 ))


# # Drop bad rows by names --> this required before I removed NaNs
# def rmv_bad_rows(data, bad_row_substr):
#     bad_rows = np.array([True if bad_row_substr in ii else False for ii in data['name']])
#     print_fn('Drops {} bad rows.'.format(sum(bad_rows)))
#     data = data[~bad_rows].reset_index(drop=True)
#     return data

# print_fn('\nDrop bad rows ...')
# dsc = rmv_bad_rows(dsc, bad_row_substr='/Users/');
# dsc = rmv_bad_rows(dsc, bad_row_substr='return');
# print_fn(dsc.iloc[:5, :4])


# Cast columns (descriptors)
print_fn('\nCast descriptors to float ...')
dsc = dsc.astype({c: np.float32 for c in dsc.columns[1:]})

# # Some descriptors ended up as strings (need to cast)
# print_fn('value counts:\n{}'.format( dsc.dtypes.value_counts() ))
# print_fn(dsc.dtypes[:7])

# # Cast (no problem)
# dsc['nAcid'] = dsc['nAcid'].astype(np.float32)
# dsc['nBase'] = dsc['nBase'].astype(np.float32)

# # Cast (problem!)
# # dsc['ABC'].astype(np.float32)
# # dsc['ABCGG'].astype(np.float32)

# # Cast values and identify bad rows
# def cast_col(data, col_name, bad_col_ids=[]):
#     for i, v in enumerate(data[col_name].values):
#         if type(v)==str:
#             try:
#                 data.loc[i, col_name] = float(v)
#             except:
#                 print_fn("Could not cast the value to numeric: {}".format(v))
#                 bad_col_ids.append(i)
#     return data, bad_col_ids

# bad_col_ids = []
# col_name = 'ABC'
# dsc, bad_col_ids = cast_col(dsc, col_name, bad_col_ids)
# print_fn('\nPrint rows that would not cast ...')
# print_fn( dsc.iloc[bad_col_ids, :5] ) # rows that couldn't cast

# # Filter out bad rows
# good_col_bool = [True if ii not in bad_col_ids else False for ii in dsc.index]
# dsc = dsc[good_col_bool].reset_index(drop=True)

# # Cast cols
# dsc['ABC'] = dsc['ABC'].astype(np.float32)
# dsc['ABCGG'] = dsc['ABCGG'].astype(np.float32)


# Impute missing values
print_fn('\nImpute NaNs ...')
dsc = dsc.reset_index(drop=True)
dsc = impute_values(dsc, print_fn=print_fn)


# Save descriptors
print_fn('\nSave ...')

# Prefix desc names
# dsc.columns = ['name'] + ['.mod'+c for c in dsc.columns[1:]]
dsc = dsc.reset_index(drop=True);
print_fn(dsc.shape)

# Save ...
dsc.to_parquet(str(out_fpath)+'.parquet')
dsc.to_csv(str(out_fpath)+'.csv', index=False)

print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
print_fn('Done.')


