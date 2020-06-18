"""
This script parses docking score results and merges the scores
of a target with mulitple types of molecular features.
A single docking scores file contains scores for a single target including
some metadata (such as Inchi-key, TITLE, SMILES, Chemgauss4).
An ML dataframe containing a single feature type is saved into a file.
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
from utils.utils import load_data, get_print_func, cast_to_float
from utils.resample import flatten_dist
from ml.data import extract_subset_fea, extract_subset_fea_col_names

# Features
FEA_DIR = filepath/'../data/raw/features/fea-subsets-hpc'
DRG_SET = 'OZD'
FEA_TYPE = 'descriptors'
meta_cols = ['Inchi-key', 'TITLE', 'SMILES']

# Docking
SCORES_DIR = filepath/'../data/raw/docking/V5.1'
SCORES_DIR = SCORES_DIR/'OZD'

# Global outdir
GOUT = filepath/'../out'


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Generate ML datasets from molecular features and docking scores.')

    parser.add_argument('-sd', '--scores_dir',
                        type=str,
                        default=str(SCORES_DIR),
                        help=f'Path to docking scores file (default: {SCORES_DIR}).')
    parser.add_argument('--fea_dir',
                        type=str,
                        default=str(FEA_DIR),
                        help=f'Path to molecular features file (default: {FEA_DIR}).')
    parser.add_argument('--drg_set',
                        type=str,
                        default=DRG_SET, 
                        choices=['OZD'], 
                        help=f'Drug set (default: {DRG_SET}).')
    parser.add_argument('--fea_type',
                        type=str,
                        default=FEA_TYPE, 
                        choices=['descriptors'],
                        help=f'Feature type (default: {FEA_TYPE}).')
    parser.add_argument('-od', '--outdir',
                        type=str,
                        default=None,
                        help=f'Output dir (default: {GOUT}/<batch_name>).')
    parser.add_argument('-f', '--fea_list',
                        type=str,
                        default=['dd'], nargs='+',
                        help=f'Prefix of feature column names (default: dd).')
    parser.add_argument('--fea_sep',
                        type=str,
                        default=['_'],
                        help=f'Prefix of feature column names (default: `_`).')
    parser.add_argument('--q_bins',
                        type=float,
                        default=0.025, 
                        help=f'Quantile to bin the docking score (default: 0.025).')
    parser.add_argument('--baseline',
                        action='store_true',
                        help=f'Number of drugs to get from features dataset (default: None).')
    parser.add_argument('--frm',
                        type=str,
                        nargs='+',
                        default=['parquet'],
                        choices=['parquet', 'feather', 'csv', 'none'],
                        help=f'Output file format for ML dfs (default: parquet).')
    parser.add_argument('--par_jobs',
                        type=int, 
                        default=1,
                        help=f'Number of joblib parallel jobs (default: 1).')
    parser.add_argument('--n_samples',
                        type=int,
                        default=None,
                        help=f'Number of docking scores to get into the ML df (default: None).')
    parser.add_argument('--n_top',
                        type=int,
                        default=None,
                        help=f'Number of top-most docking scores. This is irrelevant if n_samples \
                        was not specified (default: None).')
    parser.add_argument('--flatten',
                        action='store_true',
                        help=f'Flatten the distribution of docking scores (default: False).')

    # args, other_args = parser.parse_known_args( args )
    args= parser.parse_args( args )
    return args


def add_binner(dd_trg, score_name='reg', bin_th=2.0):
    """ Add 'binner' col to train classifier for filtering out non-dockers. """
    binner = [1 if x>=bin_th else 0 for x in dd_trg[score_name]]
    dd_trg.insert(loc=1, column='binner', value=binner)
    return dd_trg


def gen_ml_df_new(fpath, fea_df, meta_cols=['TITLE', 'SMILES'], fea_list=['dd'], fea_sep='_',
                  score_name='reg', q_cls=0.025, bin_th=2.0, print_fn=print, binner=False,
                  n_samples=None, n_top=None, baseline=False, flatten=False, frm=['parquet'],
                  outdir=Path('out'), outfigs=Path('outfigs')):
    """ Generate a single ML df for the loaded target from fpath.
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
    trg_name = fpath.with_suffix('').name # TODO depends on dock file names
    res['target'] = trg_name

    # Load dockings
    dock = load_data(fpath)
    if dock.empty:
        print_fn('Empty file')
        return None

    # Some filtering
    dock = dock.rename(columns={'Chemgauss4': score_name}) # TODO Chemgauss4 might be different
    dock = dock[ dock['TITLE'].notna() ].reset_index(drop=True) # drop TITLE==nan
    dock[score_name] = dock[score_name].map(lambda x: cast_to_float(x) ) # cast scores to float
    dock = dock[ dock[score_name].notna() ].reset_index(drop=True) # drop non-float
    dock[score_name] = abs( np.clip(dock[score_name], a_min=None, a_max=0) ) # conv scores to >=0 
    
    # Start merging docks and features using only the necessary columns
    merger = ['TITLE', 'SMILES']
    aa = pd.merge(dock, fea_df[merger], how='inner', on=merger)
    aa = aa.sort_values('reg', ascending=False).reset_index(drop=True)

    # Extract subset of samples based on docking scores
    if (n_samples is not None) and (n_top is not None):    
        n_bot = n_samples - n_top
        df_top = aa[:n_top].reset_index(drop=True)
        df_rest = aa[n_top:].reset_index(drop=True)
        
        if flatten:
            df_bot = flatten_dist(df=df_rest, n=n_bot, score_name=score_name)
        else:
            df_bot = df_rest.sample(n=n_bot, replace=False)

        assert df_top.shape[1] == df_bot.shape[1], 'Num cols must be the same when concat'
        aa = pd.concat([df_top, df_bot], axis=0).reset_index(drop=True)

        fig, ax = plt.subplots()
        ax.hist(df_bot[score_name], bins=50, facecolor='b', alpha=0.7, label='The rest (balanced)');
        ax.hist(df_top[score_name], bins=50, facecolor='r', alpha=0.7, label='Top dockers');
        plt.grid(True)
        plt.legend(loc='best', framealpha=0.5)
        plt.title(f'Samples {n_samples}; n_top {n_top}')
        plt.savefig(outfigs/f'dock.dist.{trg_name}.png')
        del df_top, df_bot, df_rest

    elif (n_samples is not None):    
        if flatten:
            aa = flatten_dist(df=aa, n=n_samples, score_name=score_name)
        else:
            aa = aa.sample(n=n_samples, replace=False)

        fig, ax = plt.subplots()
        ax.hist(aa[score_name], bins=50, facecolor='b', alpha=0.7);
        plt.grid(True)
        plt.title(f'Samples {n_samples}')
        plt.savefig(outfigs/f'dock.dist.{trg_name}.png')        

    else:
        # Plot
        fig, ax = plt.subplots()
        ax.hist(aa[score_name], bins=50, facecolor='b', alpha=0.7);
        plt.grid(True)
        plt.title(f'Samples {len(aa)}')
        plt.savefig(outfigs/f'dock.dist.{trg_name}.png')        

    bb = fea_df[ fea_df['TITLE'].isin( aa['TITLE'] ) ].reset_index(drop=True)
    ml_df = pd.merge(aa, bb, how='inner', on=merger).reset_index(drop=True)

    if n_samples is not None:
        assert n_samples==ml_df.shape[0], 'Final ml_df size must match n_samples {}'.format(fpath)

    # Add binner TODO may not be necessary since now we get good docking scores
    # if binner:
    #     dock = add_binner(dock, score_name=score_name, bin_th=bin_th)

    # -----------------------------------------    
    # Create cls col
    # ---------------
    # Find quantile value
    if ml_df[score_name].min() >= 0: # if scores were transformed to >=0
        q_cls = 1.0 - q_cls
    cls_th = ml_df[score_name].quantile(q=q_cls)
    res['cls_th'] = cls_th
    print_fn('Quantile score (q_cls={:.3f}): {:.3f}'.format( q_cls, cls_th ))

    # Generate a classification target col
    if ml_df[score_name].min() >= 0: # if scores were transformed to >=0
        value = (ml_df[score_name] >= cls_th).astype(int)
    else:
        value = (ml_df[score_name] <= cls_th).astype(int)
    ml_df.insert(loc=1, column='cls', value=value)
    # print_fn('Ratio {:.3f}'.format( dd['dock_bin'].sum() / dd.shape[0] ))

    # # Plot
    # hist, bin_edges = np.histogram(ml_df[score_name], bins=bins)
    # x = np.ones((10,)) * cls_th
    # y = np.linspace(0, hist.max(), len(x))

    # fig, ax = plt.subplots()
    # plt.hist(ml_df[score_name], bins=bins, density=False, facecolor='b', alpha=0.5)
    # plt.title(f'Scores clipped to 0: {trg_name}');
    # plt.ylabel('Count'); plt.xlabel('Docking Score');
    # plt.plot(x, y, 'r--', alpha=0.7, label=f'{q_cls}-th quantile')
    # plt.grid(True)
    # plt.savefig(outfigs/f'dock.score.{trg_name}.png')
    # -----------------------------------------    

    # Re-org cols
    fea_cols = extract_subset_fea_col_names(ml_df, fea_list=fea_list, fea_sep=fea_sep)
    meta_cols = ['Inchi-key', 'SMILES', 'TITLE', 'CAT', 'reg', 'cls']
    cols = meta_cols + fea_cols
    ml_df = ml_df[ cols ]

    # Extract the features
    def extract_and_save_fea( df, fea, frm=['parquet'] ):
        """ Extract specific feature type (including metadata) and
        save to file. 
        """
        fea_prfx_drop = [i for i in fea_list if i!=fea]
        fea_cols_drop = extract_subset_fea_col_names(df, fea_list=fea_prfx_drop, fea_sep=fea_sep)
        data = df.drop( columns=fea_cols_drop )

        frm = [i.lower() for i in frm]
        if 'none' in frm:
            return data
        
        # Outdir
        trg_outdir = outdir/f'DIR.ml.{trg_name}'
        os.makedirs(trg_outdir, exist_ok=True)
        outpath = trg_outdir/f'ml.{trg_name}.{fea}'

        for f in frm:
            if f == 'parquet':
                data.to_parquet( str(outpath)+'.parquet' )
            elif f == 'feather':
                data.to_feather( str(outpath)+'.feather' )
            elif f == 'csv':
                data.to_csv( str(outpath)+'.csv', index=False )
        return data

    print_fn( f'Create and save df ...' )
    for fea in fea_list:
        ml_df = extract_and_save_fea( ml_df, fea=fea, frm=frm )

    res['min'], res['max'] = ml_df[score_name].min(), ml_df[score_name].max()
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
    import ipdb; ipdb.set_trace()
    t0 = time()
    scores_dir = Path( args['scores_dir'] ).resolve()
    fea_dir = Path( args['fea_dir'] ).resolve()
    drg_set = Path( args['drg_set'] )
    fea_type = Path( args['fea_type'] )

    par_jobs = int( args['par_jobs'] )
    fea_list = args['fea_list']
    assert par_jobs > 0, f"The arg 'par_jobs' must be int >1 (got {par_jobs})"

    if args['outdir'] is not None:
        outdir = Path( args['outdir'] ).resolve()
    else:
        batch_name = scores_dir.parent.name
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
    
    print_fn('\nDocking files  {}'.format( scores_dir ))
    print_fn('Features       {}'.format( fea_dir ))
    print_fn('Outdir         {}'.format( outdir ))

    # Glob docking file names
    file_pattern = '*4col.csv'
    files = sorted(scores_dir.glob(file_pattern))

    # Load fea
    ID = 'TITLE'
    fea_path = Path( fea_dir, drg_set, fea_type, 'dd_fea.parquet' ).resolve()
    fea_df = load_data(fea_path)
    fea_df = fea_df.fillna(0)

    score_name = 'reg' # unified name for docking scores column in all output dfs
    bin_th = 2.0 # threshold value for the binner column (classifier)
    kwargs = {'fea_df': fea_df, 'meta_cols': meta_cols, 'fea_list': fea_list,
              'score_name': score_name, 'q_cls': args['q_bins'], 'bin_th': bin_th,
              'print_fn': print_fn, 'outdir': outdir, 'outfigs': outfigs,
              'baseline': args['baseline'], 'n_samples': args['n_samples'],
              'n_top': args['n_top'], 'frm': args['frm'], 'flatten': args['flatten'],
              }

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
    print_fn('Done.')
    lg.kill_logger()
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])


