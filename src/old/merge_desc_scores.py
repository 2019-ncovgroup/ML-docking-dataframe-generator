"""
This script parses docking score results and merges the
results of each target with modred descriptors.
Refer to this repo for into on docking results.
(github.com/2019-ncovgroup/HTDockingDataInstructions)
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time
import argparse
from pprint import pprint, pformat

# from multiprocessing import Pool
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

filepath = Path(__file__).resolve().parent

# Utils
from utils.classlogger import Logger
from utils.utils import load_data, get_print_func, drop_dup_rows

# ENA+DB 300K
DESC_PATH = filepath/'../data/processed/descriptors/ena+db/ena+db.smi.desc.parquet' # ENA+DB
meta_cols = ['name', 'smiles']  # for ena+db

# DESC_PATH = filepath / '../data/processed/descriptors/UC-molecules/UC.smi.desc.parquet' # UC-molecules
# meta_cols = ['smiles']  # for UC-molecules

SCORES_MAIN_PATH = filepath / '../data/processed'

# 03/30/2020
# SCORES_PATH = SCORES_MAIN_PATH / 'docking_data_march_30/docking_data_out_v2.0.can.parquet'

# 04/09/2020
# SCORES_PATH = SCORES_MAIN_PATH / 'V3_docking_data_april_9/docking_data_out_v3.1.can.parquet'


def parse_args(args):
    parser = argparse.ArgumentParser(description='Merge drug features (modred descriptors) and docking scores.')
    parser.add_argument('-sp', '--scores_path', default=str(SCORES_PATH), type=str,
                        help='Path to the docking scores resutls file (default: {SCORES_PATH}).')
    parser.add_argument('--desc_path', default=str(DESC_PATH), type=str,
                        help='Path to the descriptors file (default: {DESC_PATH}).')
    parser.add_argument('-od', '--outdir', default=None, type=str,
                        help=f'Output dir (default: None).')
    parser.add_argument('--q_bins', default=[ 0.025 ], type=float, nargs='+',
                        help=f'Quantiles to bin the dock score (default: 0.025).')
    parser.add_argument('--par_jobs', default=1, type=int, 
                        help=f'Number of joblib parallel jobs (default: 1).')
    args, other_args = parser.parse_known_args( args )
    return args


def gen_ml_df(dd, trg_name, meta_cols=['name', 'smiles'], score_name='reg',
              q_cls=0.025, bin_th=2.0, print_fn=print, outdir=Path('out'),
              outfigs=Path('outfigs')):
    """ Generate a single ML dataframe for the specified target column trg_name.
    Args:
        dd : dataframe with (drugs x targets) where the first col is smiles.
        trg_name : a column in dd representing the target 
        meta_cols : metadata columns to include in the dataframe
        score_name : rename the trg_name with score_name
        q_cls : quantile value to compute along the docking scores to generate the 'cls' col
        bin_th : threshold value of docking score to generate the 'binner' col
    
    Returns:
        dd_trg : the ML dataframe 
    """
    res = {}
    res['target'] = trg_name

    cols = [trg_name] + meta_cols + [c for c in dd.columns if 'mod.' in c]
    cols_parquet = cols
    dd_trg = dd[ cols ]

    # Drop NaN scores
    dd_trg = dd_trg[~dd_trg[trg_name].isna()].reset_index(drop=True)

    # Rename the scores col
    dd_trg = dd_trg.rename(columns={trg_name: score_name})

    # File name
    prefix = 'ml.'
    fname = prefix + trg_name
    
    # Translate scores to positive
    ## dd_trg[score_name] = np.clip(dd_trg[score_name], a_min=None, a_max=1)
    ## dd_trg[score_name] = dd_trg[score_name] * (-1) + 1
    dd_trg[score_name] = np.clip(dd_trg[score_name], a_min=None, a_max=0)
    dd_trg[score_name] = dd_trg[score_name] * (-1)
    dd_trg[score_name] = np.clip(dd_trg[score_name], a_min=0, a_max=None)
    res['min'] = dd_trg[score_name].min()
    res['max'] = dd_trg[score_name].max()
    bins = 50
    """
    p = dd[score_name].hist(bins=bins);
    p.set_title(f'Scores Clipped to 0: {fname}');
    p.set_ylabel('Count'); p.set_xlabel('Docking Score');
    plt.savefig(outfigs/f'dock_scores_clipped_{fname}.png');
    """
    
    # Add binner
    binner = [1 if x>=bin_th else 0 for x in dd_trg[score_name]]
    dd_trg.insert(loc=1, column='binner', value=binner)

    # -----------------------------------------    
    # Create binner
    # -----------------------------------------      
    # Find quantile value
    # for q in args['q_bins']:
    if dd_trg[score_name].min() >= 0: # if scores were transformed to >0
        q_cls = 1.0 - q_cls
    cls_th = dd_trg[score_name].quantile(q=q_cls)
    res['cls_th'] = cls_th
    print_fn('Quantile score (q_cls={:.3f}): {:.3f}'.format( q_cls, cls_th ))

    # Generate a classification target col
    if dd_trg[score_name].min() >= 0: # if scores were transformed to >0
        value = (dd_trg[score_name] >= cls_th).astype(int)
    else:
        value = (dd_trg[score_name] <= cls_th).astype(int)
    dd_trg.insert(loc=1, column='cls', value=value)
    # dd.insert(loc=1, column=f'dock_bin_{q}', value=value)
    # print_fn('Ratio {:.3f}'.format( dd['dock_bin'].sum() / dd.shape[0] ))

    # Plot
    hist, bin_edges = np.histogram(dd_trg[score_name], bins=bins)
    x = np.ones((10,)) * cls_th
    y = np.linspace(0, hist.max(), len(x))

    fig, ax = plt.subplots()
    plt.hist(dd_trg[score_name], bins=bins, density=False, facecolor='b', alpha=0.5)
    plt.title(f'Scores Clipped to 0: {fname}');
    plt.ylabel('Count'); plt.xlabel('Docking Score');
    plt.plot(x, y, 'r--', alpha=0.7, label=f'{q_cls}-th quantile')
    plt.grid(True)
    plt.savefig(outfigs/f'dock.score.bin.{fname}.png')

    # Save
    dd_trg.to_parquet( outdir / (fname+'.parquet') )
    
    # We decided to remove the 'mod.' prefix from modred descriptors
    dd_trg = dd_trg.rename(columns={c: c.split('mod.')[-1] if 'mod.' in c else c for c in dd_trg.columns})
    dd_trg.to_csv( outdir / (fname+'.csv'), index=False)        
    return res


def run(args):
    scores_path = Path( args['scores_path'] ).resolve()
    desc_path = Path( args['desc_path'] ).resolve()
    par_jobs = int( args['par_jobs'] )
    assert par_jobs > 0, f"The arg 'par_jobs' must be at least 1 (got {par_jobs})"

    if args['outdir'] is not None:
        outdir = Path( args['outdir'] ).resolve()
    else:
        outdir = scores_path.parent
    outfigs = outdir/'figs'
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outfigs, exist_ok=True)
    args['outdir'] = outdir
    
    # Logger
    lg = Logger( outdir/'create.ml.data.log' )
    print_fn = get_print_func( lg.logger )
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(args)}')
    
    print_fn('\nPython filepath  {}'.format( filepath ))
    print_fn('Scores data path {}'.format( scores_path ))
    print_fn('Descriptors path {}'.format( desc_path ))
    print_fn('Outdir data path {}'.format( outdir ))

    # -----------------------------------------
    # Load data (descriptors and dock scores)
    # -----------------------------------------    
    # Descriptors (with smiles)
    print_fn('\nLoad descriptors ...')
    dsc = load_data( DESC_PATH )
    print_fn('dsc {}'.format( dsc.shape ))
    dsc = drop_dup_rows(dsc, print_fn=print_fn)

    # Docking scores
    print_fn('\nLoad docking scores ...')
    rsp = load_data( args['scores_path'] )
    print_fn('rsp {}'.format( rsp.shape ))
    rsp = drop_dup_rows(rsp, print_fn=print_fn)

    print_fn( '\n{}'.format( rsp.columns.tolist() ))
    print_fn( '\n{}\n'.format( rsp.iloc[:3,:4] ))

    # -----------------------------------------    
    # Merge descriptors with dock scores
    # -----------------------------------------    
    unq_smiles = set( rsp['smiles'] ).intersection( set(dsc['smiles']) )
    print_fn( "Unique 'smiles' in rsp: {}".format( rsp['smiles'].nunique() ))
    print_fn( "Unique 'smiles' in dsc: {}".format( dsc['smiles'].nunique() ))
    print_fn( "Intersect on 'smiles':  {}".format( len(unq_smiles) ))

    print_fn("\nMerge descriptors with docking scores on 'smiles' ...")
    dd = pd.merge(rsp, dsc, on='smiles', how='inner')
    print_fn('Merged {}'.format( dd.shape ))
    print_fn('Unique smiles in final df: {}'.format( dd['smiles'].nunique() ))

    score_name = 'reg'
    ## meta_cols = ['name', 'smiles']  # for ena+db
    ## meta_cols = ['smiles']  # for UC-molecules
    bin_th = 2.0
    kwargs = { 'dd': dd, 'meta_cols': meta_cols, 'score_name': score_name,
               'q_cls': args['q_bins'][0], 'bin_th': bin_th, 'print_fn': print_fn,
               'outdir': outdir, 'outfigs': outfigs }

    t0 = time()
    if par_jobs > 1:
        # https://joblib.readthedocs.io/en/latest/parallel.html
        results = Parallel(n_jobs=par_jobs, verbose=1)(
                delayed(gen_ml_df)(trg_name=trg, **kwargs) for trg in rsp.columns[1:].tolist() )
        # with Pool(processes=par_jobs) as p:
        #     results = [p.apply(gen_ml_df, args=(trg_name=trg_name, **kwargs)) for trg_name in rsp.columns[1:].tolist()[:4]]
    else:
        results = []
        runtimes = []
        for trg_name in rsp.columns[1:].tolist():
            t_start = time()
            print_fn('\nProcessing {}'.format( trg_name ))

            res = gen_ml_df(trg_name=trg_name, **kwargs)
            results.append( res )

            print_fn('Runtime {:.2f} mins'.format( (time()-t_start)/60 ))
            runtimes.append( str((time() - t_start)/60) + ' mins' )

    pd.DataFrame(results).to_csv( outdir/'dock_summary.csv', index=False )

    print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
    print_fn('Done.')
    lg.kill_logger()
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])


