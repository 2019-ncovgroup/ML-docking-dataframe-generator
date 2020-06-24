"""
This script parses docking score results and merges the
scores of each target with mulitple types of molecular features.
An ML dataframe, containing a single feature type is saved into a file.
This script as docking scores that are stored in a single file which follows
the older format of 'raw_data'.

For parellel processing we use: joblib.readthedocs.io/en/latest/parallel.html
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
from utils.smiles import canon_smiles

# Features
# FEA_PATH = filepath/'../data/raw/features/BL1/ena+db.smi.desc.parquet' # BL1 (ENA+DB: ~305K)
# FEA_PATH = filepath/'../data/raw/features/BL2/BL2.dsc.parquet' # BL2 (ENA+DB: ~305K)
FEA_PATH = filepath/'../sample_data/sample_features/BL2.dsc.subset.parquet'
meta_cols = ['TITLE', 'SMILES']

# IMG_PATH
# IMG_PATH = filepath/'../data/raw/features/BL2_test_images/images.ids.0-30000.pkl'

# Docking
SCORES_MAIN_PATH = filepath/'../data/raw/raw_data'
# SCORES_PATH = SCORES_MAIN_PATH/'V3_docking_data_april_16/docking_data_out_v3.2.csv'
# SCORES_PATH = SCORES_MAIN_PATH/'V5_docking_data_april_24/pivot_SMILES.csv'
# SCORES_PATH = SCORES_MAIN_PATH/'V5_docking_data_april_24/pivot_TITLE.csv'
SCORES_PATH = filepath/'../sample_data/sample_scores/pivot_TITLE.subset.csv'

# Global outdir
GOUT = filepath/'../out'


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Generate ML datasets from molecular features and docking scores.')

    parser.add_argument('-sp', '--scores_path',
                        type=str,
                        default=str(SCORES_PATH), 
                        help=f'Path to docking scores file (default: {SCORES_PATH}).')
    parser.add_argument('--fea_path',
                        type=str,
                        default=str(FEA_PATH), 
                        help=f'Path to molecular features file (default: {FEA_PATH}).')
    parser.add_argument('--img_path',
                        type=str,
                        default=None, 
                        help='Path to molecule images file (default: None.')
    parser.add_argument('-od', '--outdir',
                        type=str,
                        default=None, 
                        help=f'Output dir (default: {GOUT}/<batch_name>).')
    parser.add_argument('-f', '--fea_list',
                        type=str,
                        default=['dsc'], nargs='+', 
                        help=f'Prefix of feature column names (default: dsc).')
    parser.add_argument('--q_bins',
                        default=0.025,
                        type=float,
                        help=f'Quantile to bin the docking score (default: 0.025).')
    parser.add_argument('--par_jobs',
                        type=int, 
                        default=1,
                        help=f'Number of joblib parallel jobs (default: 1).')
    # args, other_args = parser.parse_known_args( args )
    args= parser.parse_args( args )
    return args


def add_binner(dd_trg, score_name='reg', bin_th=2.0):
    """ Add 'binner' col to train classifier for filtering out non-dockers. """
    binner = [1 if x>=bin_th else 0 for x in dd_trg[score_name]]
    dd_trg.insert(loc=1, column='binner', value=binner)
    return dd_trg


def gen_ml_df(dd, trg_name, meta_cols=['TITLE', 'SMILES'], fea_list=['dsc'],
              score_name='reg', q_cls=0.025, bin_th=2.0, print_fn=print,
              outdir=Path('out'), outfigs=Path('outfigs')):
    """ Generate a single ML dataframe for the specified target column trg_name.
    Args:
        dd : dataframe with (molecules x targets) where the first col is TITLE
        trg_name : a column in dd representing the target 
        meta_cols : metadata columns to include in the dataframe
        score_name : rename the trg_name with score_name
        q_cls : quantile value to compute along the docking scores to generate the 'cls' col
        bin_th : threshold value of docking score to generate the 'binner' col
    
    Returns:
        dd_trg : the ML dataframe 
    """
    print_fn( f'Processing {trg_name} ...' )
    res = {}
    res['target'] = trg_name

    meta_cols = set(meta_cols).intersection(set(dd.columns.tolist()))
    meta_cols = [i for i in meta_cols]

    # fea_list = ['dsc', 'ecfp2', 'ecfp4', 'ecfp6']
    fea_sep = '.'
    fea_cols = extract_subset_fea_col_names(dd, fea_list=fea_list, fea_sep=fea_sep)
    cols = [trg_name] + meta_cols + fea_cols
    dd_trg = dd[ cols ]
    del dd

    # Drop NaN scores
    dd_trg = dd_trg[ ~dd_trg[trg_name].isna() ].reset_index(drop=True)

    # Rename the scores col
    dd_trg = dd_trg.rename( columns={trg_name: score_name} )

    # Transform scores to positive
    dd_trg[score_name] = abs( np.clip(dd_trg[score_name], a_min=None, a_max=0) )
    res['min'], res['max'] = dd_trg[score_name].min(), dd_trg[score_name].max()
    bins = 50
    """
    p = dd[score_name].hist(bins=bins);
    p.set_title(f'Scores Clipped to 0: {trg_name}');
    p.set_ylabel('Count'); p.set_xlabel('Docking Score');
    plt.savefig(outfigs/f'dock_scores_clipped_{trg_name}.png');
    """
    
    # Add binner
    dd_trg = add_binner(dd_trg, score_name=score_name, bin_th=bin_th)

    # -----------------------------------------    
    # Create cls col
    # ---------------
    # Find quantile value
    if dd_trg[score_name].min() >= 0: # if scores were transformed to >=0
        q_cls = 1.0 - q_cls
    cls_th = dd_trg[score_name].quantile(q=q_cls)
    res['cls_th'] = cls_th
    print_fn('Quantile score (q_cls={:.3f}): {:.3f}'.format( q_cls, cls_th ))

    # Generate a classification target col
    if dd_trg[score_name].min() >= 0: # if scores were transformed to >=0
        value = (dd_trg[score_name] >= cls_th).astype(int)
    else:
        value = (dd_trg[score_name] <= cls_th).astype(int)
    dd_trg.insert(loc=1, column='cls', value=value)
    # print_fn('Ratio {:.3f}'.format( dd['dock_bin'].sum() / dd.shape[0] ))

    # Plot
    hist, bin_edges = np.histogram(dd_trg[score_name], bins=bins)
    x = np.ones((10,)) * cls_th
    y = np.linspace(0, hist.max(), len(x))

    fig, ax = plt.subplots()
    plt.hist(dd_trg[score_name], bins=bins, density=False, facecolor='b', alpha=0.5)
    plt.title(f'Scores Clipped to 0: {trg_name}');
    plt.ylabel('Count'); plt.xlabel('Docking Score');
    plt.plot(x, y, 'r--', alpha=0.7, label=f'{q_cls}-th quantile')
    plt.grid(True)
    plt.savefig(outfigs/f'dock.score.bin.{trg_name}.png')

    # Separate the features
    def extract_and_save_fea( df, fea, to_csv=False ):
        """ Extract specific feature type (including metadata) and
        save to file. 
        """
        fea_prfx_drop = [i for i in fea_list if i!=fea]
        fea_cols_drop = extract_subset_fea_col_names(df, fea_list=fea_prfx_drop, fea_sep=fea_sep)
        data = df.drop( columns=fea_cols_drop )
        outpath = outdir/f'DIR.ml.{trg_name}'/f'ml.{trg_name}.{fea}'
        data.to_parquet( str(outpath)+'.parquet' )
        if to_csv:
            data.to_csv( str(outpath)+'.csv', index=False )
        return data

    print_fn( f'Create and save dataframes ...' )
    for fea in fea_list:
        to_csv = False if 'dsc' in fea else True # don't save dsc to csv yet
        dsc_df = extract_and_save_fea( dd_trg, fea=fea, to_csv=to_csv )

    # Scale desciptors and save scaler (save raw features rather the scaled)
    if sum([True for i in fea_list if 'dsc' in i]):
        # from sklearn.preprocessing import StandardScaler
        # import joblib
        # xdata = extract_subset_fea(dsc_df, fea_list='dsc', fea_sep=fea_sep)
        # cols = xdata.columns
        # sc = StandardScaler( with_mean=True, with_std=True )
        # sc.fit( xdata )
        # sc_outpath = outdir/f'ml.{trg_name}.dsc.scaler.pkl'
        # joblib.dump(sc, sc_outpath)
        # # sc_loaded = joblib.load( sc_outpath ) 

        # We decided not to prefix descriptors in csv file
        dsc_prfx = 'dsc'+fea_sep
        dsc_df = dsc_df.rename(columns={c: c.split(dsc_prfx)[-1] if dsc_prfx in c else c for c in dsc_df.columns})
        dsc_df.to_csv( outdir/f'DIR.ml.{trg_name}'/f'ml.{trg_name}.dsc.csv', index=False)        

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
        ydata = dd_trg['reg']
        xdata = extract_subset_fea(dd_trg, fea_list=fea_list, fea_sep=fea_sep)
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


def gen_ml_images(images, rsp, trg_name, score_name='reg', print_fn=print,
                  outdir=Path('out')):
    """ Generate a single ML dataframe for the specified target column trg_name.
    Args:
        images : dataframe with (molecules x targets) where the first col is TITLE
        rsp : 
        trg_name : a column in dd representing the target 
    """
    # Find intersect on TITLE
    img_title_names = [ ii['TITLE'] for ii in images ]
    titles = set(rsp['TITLE'].values).intersection(set(img_title_names))

    # Keep images with specific TITLE
    images = { ii['TITLE']: ii for ii in images }
    images = [ images[ii] for ii in titles ]
    
    # Dump images
    trg_outdir = outdir/f'DIR.ml.{trg_name}'
    outpath = trg_outdir/f'ml.{trg_name}.dct.images.pkl'
    os.makedirs(trg_outdir, exist_ok=True)
    pickle.dump( images, open(outpath, 'wb') )


def dump_single_trg(rsp, trg_name, meta_cols=['TITLE', 'SMILES'],
              score_name='reg', q_cls=0.025, print_fn=print,
              outdir=Path('out')):
    """ Dump docking scores of the specified target. """
    meta_cols = set(meta_cols).intersection(set(rsp.columns.tolist()))
    meta_cols = [i for i in meta_cols]

    cols = [trg_name] + meta_cols
    dd_trg = rsp[ cols ]; del rsp

    # Drop NaN scores
    dd_trg = dd_trg[ ~dd_trg[trg_name].isna() ].reset_index(drop=True)

    # Rename the scores col
    dd_trg = dd_trg.rename( columns={trg_name: 'dock'} )
    dd_trg[score_name] = dd_trg['dock']

    # Re-org cols
    first_cols = ['dock', score_name]
    cols = first_cols + [i for i in dd_trg.columns.tolist() if i not in first_cols]
    dd_trg = dd_trg[cols]

    # Transform scores to positive
    dd_trg[score_name] = abs( np.clip(dd_trg[score_name], a_min=None, a_max=0) )
    
    # Save
    trg_outdir = outdir/f'DIR.ml.{trg_name}'
    outpath = trg_outdir/f'docks.df.{trg_name}.csv'
    os.makedirs(trg_outdir, exist_ok=True)
    dd_trg.to_csv(outpath, index=False)


def run(args):
    # import ipdb; ipdb.set_trace()
    t0 = time()
    scores_path = Path( args['scores_path'] ).resolve()
    fea_path = Path( args['fea_path'] ).resolve()
    img_path = None if args['img_path'] is None else Path( args['img_path'] ).resolve()
    par_jobs = int( args['par_jobs'] )
    fea_list = args['fea_list']
    assert par_jobs > 0, f"The arg 'par_jobs' must be at least 1 (got {par_jobs})"

    if args['outdir'] is not None:
        outdir = Path( args['outdir'] ).resolve()
    else:
        batch_name = scores_path.parent.name
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
    
    print_fn('\nDocking scores {}'.format( scores_path ))
    print_fn('Features       {}'.format( fea_path ))
    print_fn('Images         {}'.format( img_path ))
    print_fn('Outdir         {}'.format( outdir ))

    # -----------------------------------------
    # Load data (features and docking scores)
    # -----------------------------------------    
    # Docking scores
    print_fn('\nLoad docking scores ...')
    rsp = load_data( args['scores_path'] )
    print_fn('Docking {}'.format( rsp.shape ))
    rsp = drop_dup_rows(rsp, print_fn=print_fn)

    # Get target names
    trg_names = rsp.columns[1:].tolist()[:2]


    # -----------------------------------------    
    # Dump docks of each trg to separate file
    # -----------------------------------------    
    score_name = 'reg' # unified name for docking scores column in all output dfs
    bin_th = 2.0 # threshold value for the binner column (classifier)
    kwargs = {'rsp': rsp, 'meta_cols': meta_cols,
              'score_name': score_name, 'q_cls': args['q_bins'],
              'print_fn': print_fn, 'outdir': outdir}

    # import pdb; pdb.set_trace()
    if par_jobs > 1:
        results = Parallel(n_jobs=par_jobs, verbose=20)(
                delayed(dump_single_trg)(
                    trg_name=trg, **kwargs) for trg in trg_names )
    else:
        for trg in trg_names:
            dump_single_trg( trg_name=trg, **kwargs )
    # -----------------------------------------------------
    

    # -----------------------------------------    
    # Process Images
    # -----------------------------------------    
    # Load images
    # import pdb; pdb.set_trace()
    if img_path is not None:
        print_fn('\nLoad images ...')
        images = load_data( img_path )
        print_fn('Images {} {}'.format( type(images), len(images) ))

        # Keep intersect on samples (TITLE)
        kwargs = { 'images': images, 'rsp': rsp,
                   'print_fn': print_fn, 'outdir': outdir }

        if par_jobs > 1:
            Parallel(n_jobs=par_jobs, verbose=20)(
                    delayed(gen_ml_images)(
                        trg_name=trg, **kwargs) for trg in trg_names )
        else:
            for trg in trg_names:
                gen_ml_images(trg_name=trg, **kwargs)
    # -----------------------------------------------------


    # Features (with SMILES)
    print_fn('\nLoad features ...')
    fea = load_data( fea_path )
    print_fn('Features {}'.format( fea.shape ))
    fea = drop_dup_rows(fea, print_fn=print_fn)

    print_fn( '\n{}'.format( rsp.columns.tolist() ))
    print_fn( '\n{}\n'.format( rsp.iloc[:3,:4] ))

    # -----------------------------------------    
    # Merge features with dock scores
    # -----------------------------------------    
    merger = 'TITLE' # we used 'SMILES' before
    assert merger in rsp.columns, f"Column '{merger}' must exist in the docking scores file."
    unq_smiles = set( rsp[merger] ).intersection( set(fea[merger]) )
    print_fn( 'Unique {} in rsp: {}'.format( merger, rsp[merger].nunique() ))
    print_fn( 'Unique {} in fea: {}'.format( merger, fea[merger].nunique() ))
    print_fn( 'Intersect on {}:  {}'.format( merger, len(unq_smiles) ))

    print_fn(f'\nMerge features with docking scores on {merger} ...')
    dd = pd.merge(rsp, fea, on=merger, how='inner')
    print_fn('Merged {}'.format( dd.shape ))
    print_fn('Unique {} in final df: {}'.format( merger, dd[merger].nunique() ))
    del rsp, fea

    score_name = 'reg' # unified name for docking scores column in all output dfs
    bin_th = 2.0 # threshold value for the binner column (classifier)
    kwargs = { 'dd': dd, 'meta_cols': meta_cols, 'fea_list': fea_list,
               'score_name': score_name, 'q_cls': args['q_bins'], 'bin_th': bin_th,
               'print_fn': print_fn, 'outdir': outdir, 'outfigs': outfigs }

    # import pdb; pdb.set_trace()
    if par_jobs > 1:
        results = Parallel(n_jobs=par_jobs, verbose=20)(
                delayed(gen_ml_df)(trg_name=trg, **kwargs) for trg in trg_names )
    else:
        results = [] # docking summary including ML baseline scores
        for trg in trg_names:
            res = gen_ml_df(trg_name=trg, **kwargs)
            results.append( res )

    # TODO consider to generate baselines using ecfp features as well
    results = np.round(pd.DataFrame(results), decimals=3)
    results.to_csv( outdir/'dock.ml.dsc.baseline.csv', index=False )

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


