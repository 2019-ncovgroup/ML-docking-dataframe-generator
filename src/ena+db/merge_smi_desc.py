"""
This script merges smiles and modred descriptors on drug name.
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

# datadir  = Path('/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/processed/descriptors')
datadir  = Path('/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/processed/descriptors/ena+db')
outdir = datadir

SMILES_PATH = str( datadir / 'ena+db.smi.can.csv' )
DESC_PATH   = str( datadir / 'ena+db.desc.parquet' )


def parse_args(args):
    parser = argparse.ArgumentParser(description='Merge smiles and descriptors.')
    parser.add_argument('--smiles_path', default=SMILES_PATH, type=str,
                        help=f'Full path to the smiles file (default: {SMILES_PATH}).')
    parser.add_argument('--desc_path', default=DESC_PATH, type=str,
                        help=f'Full path to the descriptors file (default: {DESC_PATH}).')
    args, other_args = parser.parse_known_args(args)
    return args


def run(args):
    t0 = time()

    # Logger
    lg = Logger( outdir/'merge.smi.desc.log' )
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

    # Save
    print_fn('\nSave ...')
    smi_dsc.to_parquet( outdir/'ena+db.smi.desc.parquet' )
        
    print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
    print_fn('Done.')
    lg.kill_logger()
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])


