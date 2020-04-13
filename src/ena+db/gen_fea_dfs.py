"""
This script generates a dataframe with the following feature types
(each feature type is prefixed with an appropriate identifier):
    1. Mordred descriptors (prefix: .mod)
    2. ECFP2 (prefix: .ecfp2)
    3. ECFP4 (prefix: .ecfp4)
    4. ECFP6 (prefix: .ecfp6)
We first merge smiles with mordred descriptors. Mordred was not 
able to produce descriptors for all smiles, so this step filters
out a small subset of smiles. Then, we compute different types of
fingerprint (FPs).
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time
import argparse
from pprint import pprint, pformat
import pandas as pd

filepath = Path(__file__).resolve().parent

# Utils
sys.path.append( os.path.abspath(filepath/'../utils') )
# from utils.classlogger import Logger
# from utils.utils import load_data, get_print_func, drop_dup_rows
from classlogger import Logger
from utils import load_data, get_print_func, drop_dup_rows
from smiles import smiles_to_fps

# datadir  = Path('/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/processed/descriptors')
datadir  = Path('/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/processed/features/ena+db')
outdir = datadir

SMILES_PATH = str( datadir/'ena+db.smi.can.csv' )
DESC_PATH   = str( datadir/'ena+db.desc.parquet' )


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate molecular feature dataframes.')
    parser.add_argument('--smiles_path', default=SMILES_PATH, type=str,
                        help=f'Full path to the smiles file (default: {SMILES_PATH}).')
    parser.add_argument('--desc_path', default=DESC_PATH, type=str,
                        help=f'Full path to the descriptors file (default: {DESC_PATH}).')
    args, other_args = parser.parse_known_args(args)
    return args


def run(args):
    t0 = time()
    os.makedirs( outdir, exist_ok=True )

    # Logger
    lg = Logger( outdir/'gen.fea.dfs.log' )
    print_fn = get_print_func( lg.logger )
    print_fn( f'File path: {filepath}' )
    print_fn( f'\n{pformat(args)}' )

    print_fn('\nPython filepath  {}'.format( filepath ))
    print_fn('Input data dir  {}'.format( datadir ))
    print_fn('Output data dir {}'.format( outdir ))

    # Load smiles and descriptors
    print_fn('\nLoad smiles and descriptors ...')
    smi = load_data( args['smiles_path'] )
    dsc = load_data( args['desc_path'] )
    print_fn('smi {}'.format( smi.shape ))
    print_fn('dsc {}'.format( dsc.shape ))

    # Remove duplicates
    print_fn('\nDrop duplicates from smiles and descriptors ...')
    smi = smi.drop_duplicates().reset_index( drop=True )
    dsc = dsc.drop_duplicates().reset_index( drop=True )
    print_fn('smi {}'.format( smi.shape ))
    print_fn('dsc {}'.format( dsc.shape ))

    # Merge
    print_fn("\nMerge smiles with descriptors on 'name' ...")
    unq_smiles = set(smi['name']).intersection(set(dsc['name']))
    print_fn( "Unique 'name' in smi: {}".format( smi['name'].nunique() ))
    print_fn( "Unique 'name' in dsc: {}".format( dsc['name'].nunique() ))
    print_fn( "Intersect on 'name':  {}".format( len(unq_smiles) ))
    smi_dsc = pd.merge(smi, dsc, on='name', how='inner')
    del smi, dsc
    smi_dsc = smi_dsc.drop_duplicates().reset_index( drop=True )
    print_fn('Merged smi_dsc {}'.format( smi_dsc.shape ))

    # Remove duplicates
    print_fn("\nKeep unique rows on smiles and descriptors ...")
    cols = smi_dsc.columns.tolist()
    cols.remove('name')
    # aa = smi_dsc.duplicated(subset=cols); print(sum(aa))
    smi_dsc = smi_dsc.drop_duplicates(subset=cols).reset_index( drop=True )
    print_fn('Final smi_dsc {}'.format( smi_dsc.shape ))

    # Now generate fingerprints
    smi_vec = smi_dsc[['smiles']].copy()
    ecfp2_df = smiles_to_fps(smi_vec, radius=2**0.5, smi_name='smiles')
    ecfp4_df = smiles_to_fps(smi_vec, radius=4**0.5, smi_name='smiles')
    ecfp6_df = smiles_to_fps(smi_vec, radius=6**0.5, smi_name='smiles')

    def add_fea_prfx(df, prfx:str):
        df = df.rename(columns={prfx+str(s) for s in df.columns[1:]})

    ecfo2_df = add_fea_prfx(ecfo2_df, prfx='ecfp2.')
    ecfo4_df = add_fea_prfx(ecfo4_df, prfx='ecfp4.')
    ecfo2_df = add_fea_prfx(ecfo6_df, prfx='ecfp6.')
    # ecfo2_df = ecfp2_df.rename(columns={'ecfp2.'+str(s) for s in ecfp2_df.columns[1:]})

    # print_fn('\nSave ...')
    # ecfp2_df.to_parquet( outdir/'ena+db.smi.ecfp2.parquet' )
    # ecfp4_df.to_parquet( outdir/'ena+db.smi.ecfp4.parquet' )
    # ecfp6_df.to_parquet( outdir/'ena+db.smi.ecfp6.parquet' )

    # Save
    print_fn('\nSave ...')
    # smi_dsc.to_parquet( outdir/'ena+db.smi.desc.parquet' )

    print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
    print_fn('Done.')
    lg.kill_logger()
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])


