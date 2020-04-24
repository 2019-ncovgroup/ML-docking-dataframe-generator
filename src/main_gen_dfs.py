"""
This script parses docking score results and merges the
results of each target with mulitple types of molecular features.
An ML dataframe, containing a single feature type is saved into a file.
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
from pprint import pformat

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
from ml.data import extract_subset_fea, extract_subset_fea_col_names

# ENA+DB 300K
# DESC_PATH = filepath/'../data/processed/descriptors/ena+db/ena+db.smi.desc.parquet' # ENA+DB
FEA_PATH = filepath/'../data/processed/features/ena+db/ena+db.features.parquet' # ENA+DB
meta_cols = ['name', 'smiles']  # for ena+db

# DESC_PATH = filepath / '../data/processed/descriptors/UC-molecules/UC.smi.desc.parquet' # UC-molecules
# meta_cols = ['smiles']  # for UC-molecules

SCORES_MAIN_PATH = filepath/'../data/processed'

# 03/30/2020
# SCORES_PATH = SCORES_MAIN_PATH / 'docking_data_march_30/docking_data_out_v2.0.can.parquet'

# 04/09/2020
# SCORES_PATH = SCORES_MAIN_PATH/'V3_docking_data_april_9/docking_data_out_v3.1.can.parquet'

# 04/09/2020
SCORES_PATH = SCORES_MAIN_PATH/'V3_docking_data_april_16/docking_data_out_v3.2.can.parquet'


def parse_args(args):
    parser = argparse.ArgumentParser(description='Merge drug features and docking scores.')
    parser.add_argument('-sp', '--scores_path', default=str(SCORES_PATH), type=str,
                        help='Path to the docking scores resutls file (default: {SCORES_PATH}).')
    # parser.add_argument('--desc_path', default=str(DESC_PATH), type=str,
    #                     help='Path to the descriptors file (default: {DESC_PATH}).')
    parser.add_argument('--fea_path', default=str(FEA_PATH), type=str,
                        help='Path to the features file (default: {FEA_PATH}).')
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
    print_fn( f'Start processing {trg_name} ...' )
    res = {}
    res['target'] = trg_name

    fea_list = ['mod', 'ecfp2', 'ecfp4', 'ecfp6']
    fea_cols = extract_subset_fea_col_names(dd, fea_list=fea_list, fea_sep='.')
    cols = [trg_name] + meta_cols + fea_cols
    dd_trg = dd[ cols ]
    del dd

    # Drop NaN scores
    dd_trg = dd_trg[ ~dd_trg[trg_name].isna() ].reset_index(drop=True)

    # Rename the scores col
    dd_trg = dd_trg.rename( columns={trg_name: score_name} )

    # File name
    prefix = 'ml.'
    fname = prefix + trg_name
    
    # Translate scores to positive
    dd_trg[score_name] = abs( np.clip(dd_trg[score_name], a_min=None, a_max=0) )
    res['min'], res['max'] = dd_trg[score_name].min(), dd_trg[score_name].max()
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

    # Separate the features
    def extract_and_save_fea( df, fea, name, to_csv=False ):
        """ Extract specific feature type (including metadata) and
        save to file. 
        """
        fea_prfx_drop = [i for i in fea_list if i!=fea]
        fea_cols_drop = extract_subset_fea_col_names(df, fea_list=fea_prfx_drop, fea_sep='.')
        data = df.drop( columns=fea_cols_drop )
        outpath_name = outdir/(fname+f'.{name}')
        data.to_parquet( str(outpath_name)+'.parquet' )
        if to_csv:
            data.to_csv( str(outpath_name)+'.csv' , index=False )
        return data

    print_fn( f'Create and save dataframes ...' )
    to_csv = True
    extract_and_save_fea( dd_trg, fea='ecfp2', name='ecfp2', to_csv=to_csv );
    extract_and_save_fea( dd_trg, fea='ecfp4', name='ecfp4', to_csv=to_csv );
    extract_and_save_fea( dd_trg, fea='ecfp6', name='ecfp6', to_csv=to_csv );
    dsc_df = extract_and_save_fea( dd_trg, fea='mod', name='dsc', to_csv=False )

    # Scale desciptors and save scaler (doesn't save the scaled features)
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.externals import joblib
    xdata = extract_subset_fea(dsc_df, fea_list='mod', fea_sep='.')
    cols = xdata.columns
    sc = StandardScaler( with_mean=True, with_std=True )
    sc.fit( xdata )
    sc_outpath = outdir/(fname+f'.dsc.scaler.pkl')
    joblib.dump(sc, sc_outpath)
    # sc_ = joblib.load( sc_outpath ) 

    # We decided to remove the feature-specific prefixes 
    dsc_df = dsc_df.rename(columns={c: c.split('mod.')[-1] if 'mod.' in c else c for c in dsc_df.columns})
    dsc_df.to_csv( outdir/(fname+'.dsc.csv'), index=False)        
    return res


def run(args):
    scores_path = Path( args['scores_path'] ).resolve()
    fea_path = Path( args['fea_path'] ).resolve()
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
    print_fn('Features path    {}'.format( fea_path ))
    print_fn('Outdir data path {}'.format( outdir ))

    # -----------------------------------------
    # Load data (features and dock scores)
    # -----------------------------------------    
    # Features (with smiles)
    print_fn('\nLoad features ...')
    fea = load_data( FEA_PATH )
    print_fn('fea {}'.format( fea.shape ))
    fea = drop_dup_rows(fea, print_fn=print_fn)

    # Docking scores
    print_fn('\nLoad docking scores ...')
    rsp = load_data( args['scores_path'] )
    print_fn('rsp {}'.format( rsp.shape ))
    rsp = drop_dup_rows(rsp, print_fn=print_fn)

    print_fn( '\n{}'.format( rsp.columns.tolist() ))
    print_fn( '\n{}\n'.format( rsp.iloc[:3,:4] ))

    # -----------------------------------------    
    # Merge features with dock scores
    # -----------------------------------------    
    unq_smiles = set( rsp['smiles'] ).intersection( set(fea['smiles']) )
    print_fn( "Unique smiles in rsp: {}".format( rsp['smiles'].nunique() ))
    print_fn( "Unique smiles in fea: {}".format( fea['smiles'].nunique() ))
    print_fn( "Intersect on smiles:  {}".format( len(unq_smiles) ))

    print_fn("\nMerge features with docking scores on 'smiles' ...")
    dd = pd.merge(rsp, fea, on='smiles', how='inner')
    print_fn('Merged {}'.format( dd.shape ))
    print_fn('Unique smiles in final df: {}'.format( dd['smiles'].nunique() ))
    trg_names = rsp.columns[1:].tolist()
    del rsp, fea

    # dd = load_data( '/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/processed/V3_docking_data_april_9/dd.parquet' )
    # trg_names = dd.columns[1:20].tolist()
    score_name = 'reg'
    bin_th = 2.0
    kwargs = { 'dd': dd, 'meta_cols': meta_cols, 'score_name': score_name,
               'q_cls': args['q_bins'][0], 'bin_th': bin_th, 'print_fn': print_fn,
               'outdir': outdir, 'outfigs': outfigs }

    t0 = time()
    if par_jobs > 1:
        # https://joblib.readthedocs.io/en/latest/parallel.html
        results = Parallel(n_jobs=par_jobs, verbose=20)(
                delayed(gen_ml_df)(trg_name=trg, **kwargs) for trg in trg_names )
    else:
        results = []
        runtimes = []
        for trg in trg_names:
            t_start = time()
            res = gen_ml_df(trg_name=trg, **kwargs)
            results.append( res )
            # print_fn('Runtime {:.2f} mins'.format( (time()-t_start)/60 ))
            runtimes.append( str((time()-t_start)/60) + ' mins' )

    pd.DataFrame(results).to_csv( outdir/'dock.summary.csv', index=False )

    print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
    print_fn('Done.')
    lg.kill_logger()
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])


