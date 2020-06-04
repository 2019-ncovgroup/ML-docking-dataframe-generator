"""
This script parses docking score results and merges the
scores of each target with mulitple types of molecular features.
An ML dataframe, containing a single feature type is saved into a file.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time
import argparse
from pprint import pformat
import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

filepath = Path(__file__).resolve().parent

# Utils
from utils.classlogger import Logger
from utils.utils import load_data, get_print_func, drop_dup_rows
from ml.data import extract_subset_fea, extract_subset_fea_col_names
from utils.smiles import canon_smiles

# Features
FEA_PATH = filepath/'../data/raw/features/fea-subsets-hpc/descriptors/dd_fea.parquet'
meta_cols = ['Inchi-key', 'TITLE', 'SMILES']

# Docking
SCORES_MAIN_DIR = filepath/'../data/raw/dock-2020-06-01'
SCORES_MAIN_DIR = SCORES_MAIN_DIR/'OZD/'

# Global outdir
GOUT = filepath/'../out'


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate ML datasets from molecular features and docking scores.')
    parser.add_argument('-sd', '--scores_main_dir', default=str(SCORES_MAIN_DIR), type=str,
                        help=f'Path to docking scores file (default: {SCORES_MAIN_DIR}).')
    parser.add_argument('--fea_path', default=str(FEA_PATH), type=str,
                        help=f'Path to molecular features file (default: {FEA_PATH}).')
    parser.add_argument('-od', '--outdir', default=None, type=str,
                        help=f'Output dir (default: {GOUT}/<batch_name>).')
    parser.add_argument('-f', '--fea_list', default=['dd'], nargs='+', type=str,
                        help=f'Prefix of feature column names (default: dd).')
    parser.add_argument('--fea_sep', default=['_'], type=str,
                        help=f'Prefix of feature column names (default: `_`).')
    parser.add_argument('--q_bins', default=0.025, type=float,
                        help=f'Quantile to bin the docking score (default: 0.025).')
    parser.add_argument('--par_jobs', default=1, type=int, 
                        help=f'Number of joblib parallel jobs (default: 1).')
    # args, other_args = parser.parse_known_args( args )
    args= parser.parse_args( args )
    return args


def add_binner(dd_trg, score_name='reg', bin_th=2.0):
    """ Add 'binner' col to train classifier for filtering out non-dockers. """
    binner = [1 if x>=bin_th else 0 for x in dd_trg[score_name]]
    dd_trg.insert(loc=1, column='binner', value=binner)
    return dd_trg


def cast_to_float(x, float_format=np.float64):
    try:
        x = np.float64(x)
    except:
        print("Could not cast the value to numeric: {}".format(x))
        x = np.nan
    return x


def gen_ml_df_new(fpath, fea_df, meta_cols=['TITLE', 'SMILES'], fea_list=['dd'], fea_sep='_',
        score_name='reg', q_cls=0.025, bin_th=2.0, print_fn=print, binner=False, n_sample=3e5,
        baseline=False, outdir=Path('out'), outfigs=Path('outfigs')):
    """ Generate a single ML dataframe for the specified target column trg_name.
    This func was specifically created to process the new LARGE DOE-MD datasets
    with ZINC drugs that contains >6M molecules.
    Args:
        fpath : path to load docking scores file
        fea_df : df with features
        meta_cols : metadata columns to include in the dataframe
        score_name : rename the docking score col with score_name
        q_cls : quantile value to compute along the docking scores to generate the 'cls' col
        bin_th : threshold value of docking score to generate the 'binner' col
        baseline : whther to compute ML baseline scores
    
    Returns:
        ml_df : the ML dataframe 
    """
    print_fn( f'Processing  {fpath.name} ...' )
    res = {}
    trg_name = fpath.name.split('.')[0] # TODO depends on dock file names
    res['target'] = trg_name

    # Outdir
    trg_outdir = outdir/f'DIR.ml.{trg_name}'
    os.makedirs(trg_outdir, exist_ok=True)

    # Load dockings
    dock = load_data(fpath)
    if dock.empty:
        print_fn('Empty file')
        return None

    # Get meta columns
    meta_cols = list( set(meta_cols).intersection(set(dock.columns.tolist())) )

    # Rename the scores col
    dock = dock.rename(columns={'Chemgauss4': score_name}) # TODO Chemgauss4 might be different

    # Cast and drop NaN scores
    dock[score_name] = dock[score_name].map(lambda x: cast_to_float(x) )
    dock = dock[ ~dock[score_name].isna() ].reset_index(drop=True)

    # Transform scores to positive
    dock[score_name] = abs( np.clip(dock[score_name], a_min=None, a_max=0) )
    res['min'], res['max'] = dock[score_name].min(), dock[score_name].max()
    bins = 50
    """
    p = dd[score_name].hist(bins=bins);
    p.set_title(f'Scores Clipped to 0: {trg_name}');
    p.set_ylabel('Count'); p.set_xlabel('Docking Score');
    plt.savefig(outfigs/f'dock_scores_clipped_{trg_name}.png');
    """
    
    # Add binner TODO may not be necessary since now we get good docking scores
    if binner:
        dock = add_binner(dock, score_name=score_name, bin_th=bin_th)

    # -----------------------------------------    
    # Create cls col
    # ---------------
    # Find quantile value
    if dock[score_name].min() >= 0: # if scores were transformed to >=0
        q_cls = 1.0 - q_cls
    cls_th = dock[score_name].quantile(q=q_cls)
    res['cls_th'] = cls_th
    print_fn('Quantile score (q_cls={:.3f}): {:.3f}'.format( q_cls, cls_th ))

    # Generate a classification target col
    if dock[score_name].min() >= 0: # if scores were transformed to >=0
        value = (dock[score_name] >= cls_th).astype(int)
    else:
        value = (dock[score_name] <= cls_th).astype(int)
    dock.insert(loc=1, column='cls', value=value)
    # print_fn('Ratio {:.3f}'.format( dd['dock_bin'].sum() / dd.shape[0] ))

    # Plot
    hist, bin_edges = np.histogram(dock[score_name], bins=bins)
    x = np.ones((10,)) * cls_th
    y = np.linspace(0, hist.max(), len(x))

    fig, ax = plt.subplots()
    plt.hist(dock[score_name], bins=bins, density=False, facecolor='b', alpha=0.5)
    plt.title(f'Scores clipped to 0: {trg_name}');
    plt.ylabel('Count'); plt.xlabel('Docking Score');
    plt.plot(x, y, 'r--', alpha=0.7, label=f'{q_cls}-th quantile')
    plt.grid(True)
    plt.savefig(outfigs/f'dock.score.{trg_name}.png')
    # -----------------------------------------    

    # Merge docks and features
    merger = ['TITLE', 'SMILES']
    ml_df = pd.merge(dock, fea_df, how='inner', on=merger).reset_index(drop=True)

    # Re-org cols
    fea_cols = extract_subset_fea_col_names(ml_df, fea_list=fea_list, fea_sep=fea_sep)
    meta_cols = ['Inchi-key', 'SMILES', 'TITLE', 'CAT', 'reg', 'cls']
    cols = meta_cols + fea_cols
    ml_df = ml_df[ cols ]

    # Separate the features
    def extract_and_save_fea( df, fea, to_csv=False, to_feather=True ):
        """ Extract specific feature type (including metadata) and
        save to file. 
        """
        fea_prfx_drop = [i for i in fea_list if i!=fea]
        fea_cols_drop = extract_subset_fea_col_names(df, fea_list=fea_prfx_drop, fea_sep=fea_sep)
        data = df.drop( columns=fea_cols_drop )

        # Save
        outpath = trg_outdir/f'ml.{trg_name}.{fea}'
        data.to_parquet( str(outpath)+'.parquet' )
        if to_csv:
            data.to_csv( str(outpath)+'.csv', index=False )
        if to_feather:
            data.to_feather( str(outpath)+'.feather' )
        return data

    # import ipdb; ipdb.set_trace()
    print_fn( f'Create and save df ...' )
    for fea in fea_list:
        # to_csv = False if 'dd' in fea else True # don't save dsc to csv yet
        to_csv = True
        ml_df = extract_and_save_fea( ml_df, fea=fea, to_csv=to_csv )

    # Save subset to csv
    # import ipdb; ipdb.set_trace()
    # ml_df = ml_df.sample(n=int(n_sample), random_state=0).reset_index(drop=True)
    # ml_df.to_csv( trg_outdir/f'ml.{trg_name}.dd.csv', index=False)        

    if baseline:
        try:
            # Train LGBM as a baseline model
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            from datasplit.splitter import data_splitter
            from ml.evals import calc_preds, calc_scores, dump_preds
            ml_model_def = lgb.LGBMRegressor
            ml_init_args = {'n_jobs': 8}
            ml_fit_args = {'verbose': False, 'early_stopping_rounds': 10}
            model = ml_model_def( **ml_init_args )
            ydata = ml_df['reg']
            xdata = extract_subset_fea(ml_df, fea_list=fea_list, fea_sep=fea_sep)
            x_, xte, y_, yte = train_test_split(xdata, ydata, test_size=0.2)
            xtr, xvl, ytr, yvl = train_test_split(x_, y_, test_size=0.2)
            ml_fit_args['eval_set'] = (xvl, yvl)
            model.fit(xtr, ytr, **ml_fit_args)
            y_pred, y_true = calc_preds(model, x=xte, y=yte, mltype='reg')
            te_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype='reg', metrics=None)
            res['r2'] = te_scores['r2']
            res['mae'] = te_scores['median_absolute_error']
        except:
            print('Could not import lightgbm.')

    return res


def run(args):
    t0=time()
    scores_main_dir = Path( args['scores_main_dir'] ).resolve()
    fea_path = Path( args['fea_path'] ).resolve()

    par_jobs = int( args['par_jobs'] )
    fea_list = args['fea_list']
    assert par_jobs > 0, f"The arg 'par_jobs' must be int >1 (got {par_jobs})"

    if args['outdir'] is not None:
        outdir = Path( args['outdir'] ).resolve()
    else:
        batch_name = scores_main_dir.parent.name
        outdir = Path( GOUT/batch_name ).resolve()

    outfigs = outdir/'figs'
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outfigs, exist_ok=True)
    args['outdir'] = outdir
    
    # Logger
    lg = Logger( outdir/'gen.ml.data.log' )
    print_fn = get_print_func( lg.logger )
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(args)}')
    
    print_fn('\nDocking files  {}'.format( scores_main_dir ))
    print_fn('Features       {}'.format( fea_path ))
    print_fn('Outdir         {}'.format( outdir ))

    # (ap) new added ------------------------------------
    file_pattern = '*4col.csv'
    files = sorted(scores_main_dir.glob(file_pattern))

    # Load fea
    fea_df = load_data(fea_path)
    fea_df = fea_df.sample(n=int(3e5), random_state=0).reset_index(drop=True)
    # (ap) new added ------------------------------------

    score_name = 'reg' # unified name for docking scores column in all output dfs
    bin_th = 2.0 # threshold value for the binner column (classifier)
    kwargs = { 'fea_df': fea_df, 'meta_cols': meta_cols, 'fea_list': fea_list,
               'score_name': score_name, 'q_cls': args['q_bins'], 'bin_th': bin_th,
               'print_fn': print_fn, 'outdir': outdir, 'outfigs': outfigs,
               'baseline': True }

    if par_jobs > 1:
        results = Parallel(n_jobs=par_jobs, verbose=20)(
                delayed(gen_ml_df_new)(fpath=f, **kwargs) for f in files )
    else:
        results = [] # docking summary including ML baseline scores
        for f in files:
            res = gen_ml_df_new(fpath=f, **kwargs)
            results.append( res )

    # TODO consider to generate baselines using ecfp features as well
    results = [r for r in results if r is not None]
    results = np.round(pd.DataFrame(results), decimals=3)
    results.sort_values('target').reset_index(drop=True)
    results.to_csv( outdir/'dock.ml.dd.baseline.csv', index=False )

    # --------------------------------------------------------
    print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
    # import ipdb; ipdb.set_trace()
    print_fn('Done.')
    lg.kill_logger()
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])


