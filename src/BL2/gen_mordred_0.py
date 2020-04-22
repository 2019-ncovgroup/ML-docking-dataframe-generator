"""
This script computes Mordred descriptors and saves the dataframe.
(each feature type is prefixed with an appropriate identifier):
    Mordred descriptors (prefix: .mod)
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time
from datetime import datetime
import argparse
from pprint import pprint, pformat

import numpy as np
import pandas as pd

filepath = Path(__file__).resolve().parent

# Utils
sys.path.append( os.path.abspath(filepath/'../utils') )
# from utils.classlogger import Logger
# from utils.utils import load_data, get_print_func, drop_dup_rows
from classlogger import Logger
from utils import load_data, get_print_func, drop_dup_rows, dropna
from smiles import canon_df, smiles_to_mordred

# DATADIR
datadir = Path('/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/raw/Baseline-Screen-Datasets/BL2 (current)')

# OUTDIR
t = datetime.now()
t = [t.year, '-', t.month, '-', t.day]
date = ''.join( [str(i) for i in t] )
outdir = Path( '/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/processed/features/BL2/', date )

# SMILES
SMILES_PATH = str( datadir/'BL2.smi' )


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate molecular feature dataframe.')
    parser.add_argument('--smiles_path', default=SMILES_PATH, type=str,
                        help=f'Full path to the smiles file (default: {SMILES_PATH}).')
    parser.add_argument('--par_jobs', default=1, type=int, 
                        help=f'Number of joblib parallel jobs (default: 1).')
    args, other_args = parser.parse_known_args(args)
    return args


def run(args):
    t0 = time()
    os.makedirs( outdir, exist_ok=True )

    # Logger
    lg = Logger( outdir/'gen.mord.df.log' )
    print_fn = get_print_func( lg.logger )
    print_fn( f'File path: {filepath}' )
    print_fn( f'\n{pformat(args)}' )

    print_fn('\nPython filepath {}'.format( filepath ))
    print_fn('Input data dir  {}'.format( datadir ))
    print_fn('Output data dir {}'.format( outdir ))

    # Load smiles and descriptors
    print_fn('\nLoad smiles ...')
    smiles_path = Path(args['smiles_path'])
    fname = smiles_path.with_suffix('').name
    # smi = pd.read_csv( smiles_path, sep='\t', names=['smiles', 'name'] )
    smi = pd.read_csv( smiles_path )
    print_fn('smi {}'.format( smi.shape ))

    # Remove duplicates
    print_fn('\nDrop duplicates from smiles ...')
    smi = smi.drop_duplicates().reset_index( drop=True )
    print_fn('smi {}'.format( smi.shape ))
    
    # Duplicates
    # smi = pd.read_csv( 'BL2.smi', sep='\t', names=['smiles', 'name'] )
    # dup = smi[ smi.duplicated(subset=['smiles'], keep=False) ].reset_index(drop=True)
    # print( dup['smiles'].value_counts() )

    print_fn("\nCanonicalize smiles ...")
    # smi = smi[:1000]
    smi = canon_df( smi, par_jobs=args['par_jobs'] )

    # Drop bad SMILES (that were no canonicalized)
    ids = smi['smiles'].isna()
    bad_smi = smi[ ids ].reset_index(drop=True)
    smi = smi[ ~ids ].reset_index(drop=True)

    # Generate descriptors
    dsc = smiles_to_mordred(smi, smi_name='smiles', par_jobs=args['par_jobs'])

    # Filter NaN values - step 1
    # Drop rows where all values are NaNs
    print_fn('\nDrop rows where all values are NaN ...')
    print_fn('Shape: {}'.format( dsc.shape ))
    idx = (dsc.isna().sum(axis=1)==dsc.shape[1]).values
    dsc = dsc.iloc[~idx, :]
    # Drop cols where all values are NaNs
    # idx = (dsc.isna().sum(axis=0)==dsc.shape[0]).values
    # dsc = dsc.iloc[:, ~idx]
    print_fn('Shape: {}'.format( dsc.shape ))

    # Filter rows/cols based on the ratio of NaNs - step 2
    # Remove rows and cols based on a thershold of NaN values
    # print(dsc.isna().sum(axis=1).sort_values(ascending=False))
    # p=dsc.isna().sum(axis=1).sort_values(ascending=False).hist(bins=100);
    th = 0.2
    print_fn('\nDrop rows (drugs) with at least {} NaNs (at least {} out of {}).'.format(
        th, int(th * dsc.shape[1]), dsc.shape[1]))
    print_fn('Shape: {}'.format( dsc.shape ))
    dsc = dropna(dsc, axis=1, th=th)
    print_fn('Shape: {}'.format( dsc.shape ))

    # Cast features (descriptors)
    print_fn('\nCast descriptors to float ...')
    id0 = 2
    dsc = dsc.astype({c: np.float32 for c in dsc.columns[id0:]})

    # Impute missing values
    print_fn('\nImpute NaNs ...')
    print_fn('Total NaNs: {}'.format( dsc.isna().values.flatten().sum() ))
    # dsc = dsc.reset_index(drop=True)
    # dsc = impute_values(dsc, print_fn=print_fn) # ap's approach
    dsc = dsc.fillna(0.0)
    print_fn('Total NaNs: {}'.format( dsc.isna().values.flatten().sum() ))

    # Prefix features
    def add_fea_prfx(df, prfx:str, id0:int):
        return df.rename(columns={s: prfx+str(s) for s in df.columns[id0:]})

    dsc = add_fea_prfx(dsc, prfx='mod.', id0=id0)

    # Save
    print_fn('\nSave ...')
    dsc = dsc.reset_index(drop=True)
    # dsc.to_parquet( outdir/'BL2.dsc.parquet' )
    dsc.to_parquet( outdir/(fname+'.dsc.parquet') )

    print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
    print_fn('Done.')
    lg.kill_logger()
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])


