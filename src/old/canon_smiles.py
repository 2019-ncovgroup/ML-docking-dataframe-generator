"""
This script is primarily used to canonicalize smiles in the
'smiles' column in the docking results files.

Example:
    python canon_smiles -p ../../data/raw/raw_data/docking_data_march_23/dock_out_ml_v4.csv
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time
import argparse

from utils.smiles import canon_single_smile, canon_df
from utils.utils import load_data


def parse_args(args):
    parser = argparse.ArgumentParser(description='Canonicalize SMILES.')
    parser.add_argument('-dp', '--datapath', default=None, type=str,
                        help='Path to docking scores resutls file (default: None).')
    parser.add_argument('--outdir', default=None, type=str,
                        help='Folder path to save the processed file (default: None).')
    args, other_args = parser.parse_known_args(args)
    return args


def run(args):
    t0 = time()
    datapath = Path( args['datapath'] ).absolute()
    rsp = load_data(datapath)
    print(rsp.shape)
    
    # Create outdir
    if args['outdir'] is not None:
        outdir = Path( args['outdir'] )
    else:
        split_str_on_sep = str(datapath).split('/data/')
        dir1 = split_str_on_sep[0] + '/data/processed'
        dir2 = str(Path( split_str_on_sep[1] ).parent).split(os.sep)[-1]
        outdir = Path(dir1, dir2)
    os.makedirs(outdir, exist_ok=True)

    # Canonicalize
    smi_vec = rsp[['smiles']].copy()
    smi_vec_ret = canon_df(smi_vec, smi_name='smiles')
    rsp['smiles'] = smi_vec_ret
    
    # Save
    outname = str(datapath.name).split('.csv')[0]
    outpath = outdir / (outname + '.can.csv')
    rsp.to_csv(outpath, index=False)

    outpath = outdir / (outname + '.can.parquet')
    rsp.to_parquet(outpath)

    print('Runtime {:.2f} mins'.format( (time() - t0)/60 ))


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
