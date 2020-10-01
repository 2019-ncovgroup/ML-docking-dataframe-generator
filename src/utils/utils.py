from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
# from time import time
# import argparse
from pprint import pprint, pformat
# from glob import glob
import pickle

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns

# import sklearn
import numpy as np
import pandas as pd

# from scipy import stats
# from pandas.api.types import is_string_dtype
# from sklearn.preprocessing import LabelEncoder


def verify_path(path):
    """ Verify that the path exists. """
    if path is None:
        sys.exit('Program terminated. You must specify a correct path.')
    path = Path(path)
    assert path.exists(), f'The specified path was not found: {path}.'
    return path


def load_data(datapath, file_format=None):
    datapath = verify_path(datapath)
    if file_format is None:
        file_format = str(datapath).split('.')[-1]

    if file_format == 'parquet':
        data = pd.read_parquet(datapath)
    elif file_format == 'hdf5':
        data = pd.read_hdf5(datapath)
    elif file_format == 'csv':
        data = pd.read_csv(datapath)
    elif file_format == 'pkl':
        data = pickle.load(open(datapath, 'rb'))
    else:
        try:
            data = pd.read_csv(datapath)
        except:
            print('Cannot load file', datapath)
    return data


def drop_dup_rows(data, print_fn=print):
    """ Drop duplicate rows. """
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


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open( Path(outpath), 'w' ) as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))


def get_print_func(logger=None):
    """ Returns the python 'print' function if logger is None. Othersiwe, returns logger.info. """
    return print if logger is None else logger.info


def cast_to_float(x, float_format=np.float64):
    """ Cast values to float, and return nan if unsuccessful. """
    try:
        x = np.float64(x)
    except:
        print('Could not cast the value to numeric: {}'.format(x))
        x = np.nan
    return x
