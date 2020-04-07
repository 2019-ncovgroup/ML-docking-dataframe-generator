"""
This code takes the original descriptors from ena+db.desc,
cleans data, and saves. The original file is not overwritten.

Note that for the big runs we don't remove descriptor cols even
if all values are NaNs (we simply impute with 0).
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

filepath = Path(__file__).resolve().parent

# Utils
sys.path.append(str(filepath/'../'))
from classlogger import Logger
from impute import impute_values

datapath = Path('/vol/ml/apartin/projects/covid-19/data/')
outpath  = Path('/vol/ml/apartin/projects/covid-19/data/')


def drop_dup_rows(data, print_fn=print):
    print_fn('\nDrop duplicates ...')
    cnt0 = data.shape[0]; print_fn('Samples: {}'.format( cnt0 ))
    data = data.drop_duplicates().reset_index(drop=True)
    cnt1 = data.shape[0]; print_fn('Samples: {}'.format( cnt1 ));
    print_fn('Dropped duplicates: {}'.format( cnt0-cnt1 ))
    return data


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
    df = df.reset_index(drop=True)
    return df


t0 = time()
# lg = Logger(filepath/'clean.desc.rick.large.log')
# print_fn = lg.logger.info
print_fn = print

print_fn('\nPython filepath    {}'.format( filepath ))
print_fn('Original data path {}'.format( datapath ))
print_fn('Output data path   {}'.format( outpath ))

# File names
in_fname = 'ADRP-P1.desc'
in_fpath  = datapath/in_fname
out_fpath = outpath/in_fname

# Load Modred descriptors (this descriptors are the output
# from mordred run on ena+db.can from Rick)
print_fn('\nLoad desc ... {}'.format( in_fpath ))
t_start = time()
dsc = pd.read_csv(in_fpath)
t_end = time() - t_start
dsc = dsc.rename(columns={'name': 'reg'})
print_fn(dsc.shape)

# Filter NaN values - step 1
# Drop rows where all values are NaNs
# idx = (dsc.isna().sum(axis=1)==dsc.shape[1]).values
# dsc = dsc.iloc[~idx, :]
# print_fn(dsc.shape)
# Drop cols where all values are NaNs
# idx = (dsc.isna().sum(axis=0)==dsc.shape[0]).values
# dsc = dsc.iloc[:, ~idx]

def cast_to_float(x, float_format=np.float64):
    try:
        x = np.float64(x)
    except:
        print("Could not cast the value to numeric: {}".format(x))
        x = np.nan
    return x

# idx = dsc['reg'].map(lambda x: False if type(x)==str else True)
dsc_org = dsc.copy()
reg_col = dsc['reg'].map(lambda x: cast_to_float(x) )
dsc = dsc[~reg_col.isna()].reset_index(drop=True)
dsc = dsc.reset_index(drop=True)
dsc['reg'] = dsc['reg'].astype(np.float64)
dsc_reg = dsc['reg'].copy()

# Cast values and identify bad rows
# def cast_col(data, col_name, type_format=np.float32, bad_col_ids=[]):
#     for i, v in enumerate(data[col_name].values):
#         if type(v)==str:
#             try:
#                 data.loc[i, col_name] = float(v)
#             except:
#                 print_fn("Could not cast the value to numeric: {}".format(v))
#                 bad_col_ids.append(i)
#     return data, bad_col_ids

# bad_col_ids = []
# col_name = 'reg'
# dsc, bad_col_ids = cast_col(dsc, col_name, bad_col_ids)
# print_fn('\nPrint rows that would not cast ...')
# print_fn( dsc.iloc[bad_col_ids, :5] ) # rows that couldn't cast

# Filter out bad rows
# good_col_bool = [True if ii not in bad_col_ids else False for ii in dsc.index]
# dsc = dsc[good_col_bool].reset_index(drop=True)


print_fn('\nLoad rsp ... {}'.format( in_fpath ))
rsp = pd.read_csv('/vol/ml/apartin/projects/covid-19/data/raw_data/docking_data_march_22/out_v3.csv')
rsp_org = rsp.copy()
rsp = rsp[['smiles','ADRP_pocket1_dock']]
rsp = rsp.rename(columns={'ADRP_pocket1_dock': 'reg'})
rsp_reg = rsp['reg'].copy()
rsp_reg = np.clip(rsp_reg, None, 1) * (-1) + 1
print_fn(rsp.shape)

idx = set(dsc['reg']).intersection(set(rsp['reg']))

print_fn('\nLoad rsp ... {}'.format( in_fpath ))
dd = pd.merge(rsp, dsc, how='inner', on='reg')


# Cast float64 to float32
print_fn('\nCast float64 to float32 ...')
print_fn('Memory {:.3f} GB'.format( dsc.memory_usage().sum()/1e9 ))

print_fn( dsc.dtypes.value_counts() )
dtypes_dict = {c: np.float32 if dsc[c].dtype=='float64' else dsc[c].dtype for c in dsc.columns.tolist()}
dsc = dsc.astype(dtypes_dict)

print_fn( dsc.dtypes.value_counts() )
print_fn('Memory {:.3f} GB'.format( dsc.memory_usage().sum()/1e9 ))


# Filter rows/cols based on NaN values - step 2
# Remove rows and cols based on a thershold of NaN values
# print(dsc.isna().sum(axis=1).sort_values(ascending=False))
# p=dsc.isna().sum(axis=1).sort_values(ascending=False).hist(bins=100);
th = 0.2
print_fn('\nDrop rows (drugs) with at least {} NaNs (at least {} out of {}).'.format(
    th, int(th * dsc.shape[1]), dsc.shape[1]))
cnt0 = dsc.shape[0]; print_fn('Samples: {}'.format( cnt0 ))
dsc = dropna(dsc, axis=1, th=th)
cnt1 = dsc.shape[0]; print_fn('Samples: {}'.format( cnt1 ));
print_fn('Dropped duplicates: {}'.format( cnt0-cnt1 ))


# Cast manually
dsc['reg']=dsc['reg'].astype(np.float32)
dsc['ABC']=dsc['ABC'].astype(np.float32)
dsc['ABCGG']=dsc['ABCGG'].astype(np.float32)
dsc['nAcid']=dsc['nAcid'].astype(np.int16)
dsc['nBase']=dsc['nBase'].astype(np.int16)


# Drop duplicates
# dsc = drop_dup_rows(dsc, print_fn=print_fn)

# Print
print_fn(dsc.iloc[:5, :7])


# Impute missing values
print_fn('\nImpute NaNs ...')
# dsc = dsc.reset_index(drop=True)
# dsc = impute_values(dsc, print_fn=print_fn) # ap's approach
dsc = dsc.fillna(0.0)
print_fn('Total NaNs: {}.'.format( dsc.isna().values.flatten().sum() ))


# Save descriptors
print_fn('\nSave ...')

# Prefix desc names
dsc.columns = ['name'] + ['mod.'+c for c in dsc.columns[1:]]
dsc = dsc.reset_index(drop=True)
print_fn(dsc.shape)
dsc.to_parquet(str(out_fpath)+'.parquet')
# dsc.to_csv(str(out_fpath)+'.csv', index=False)

print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
print_fn('Done.')
lg.kill_logger()


