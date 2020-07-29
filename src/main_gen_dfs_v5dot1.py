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

from utils.classlogger import Logger
from utils.utils import load_data, get_print_func, cast_to_float
from utils.resample import flatten_dist
from ml.data import extract_subset_fea, extract_subset_fea_col_names

filepath = Path(__file__).resolve().parent

# Drug set
DRG_SET = 'OZD'

# Docking
SCR_DIR = filepath/'../data/raw/docking/V5.1'

# Features
FEA_DIR = Path(filepath, '../data/raw/features/fea-subsets-hpc').resolve()

# Global outdir
GOUT = filepath/'../out'


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Generate ML datasets from molecular features and docking scores.')

    parser.add_argument('--drg_set',
                        type=str,
                        default=DRG_SET,
                        choices=['OZD'],
                        help=f'Drug set (default: {DRG_SET}).')
    parser.add_argument('-sd', '--scr_dir',
                        type=str,
                        default=str(SCR_DIR),
                        help=f'Path to docking scores file (default: {SCR_DIR}).')
    # parser.add_argument('--fea_dir',
    #                     type=str,
    #                     default=str(FEA_DIR),
    #                     help=f'Path to molecular features file (default: {FEA_DIR}).')
    parser.add_argument('--fea_type',
                        type=str,
                        default=['descriptors'],
                        nargs='+',
                        choices=['descriptors', 'images', 'fps'],
                        help='Feature type (default: descriptors).')
    parser.add_argument('-od', '--outdir',
                        type=str,
                        default=None,
                        help=f'Output dir (default: {GOUT}/<batch_name>).')

    # parser.add_argument('-f', '--fea_list',
    #                     type=str,
    #                     default=['dd'], nargs='+',
    #                     help='Prefix of feature column names (default: dd).')
    parser.add_argument('--fea_sep',
                        type=str,
                        default=['_'],
                        help='Prefix of feature column names (default: `_`).')

    parser.add_argument('--n_samples',
                        type=int,
                        default=None,
                        help='Number of docking scores to get into the ML df (default: None).')
    parser.add_argument('--n_top',
                        type=int,
                        default=None,
                        help='Number of top-most docking scores. This is irrelevant if n_samples \
                        was not specified (default: None).')
    parser.add_argument('--flatten',
                        action='store_true',
                        help='Flatten the distribution of docking scores (default: False).')

    parser.add_argument('--baseline',
                        action='store_true',
                        help='Number of drugs to get from features dataset (default: None).')
    parser.add_argument('--frm',
                        type=str,
                        nargs='+',
                        default=['parquet'],
                        choices=['parquet', 'feather', 'csv', 'none'],
                        help='Output file format for ML dfs (default: parquet).')
    parser.add_argument('--par_jobs',
                        type=int,
                        default=1,
                        help='Number of joblib parallel jobs (default: 1).')

    args = parser.parse_args(args)
    return args


def add_binner(dd_trg, score_name='reg', bin_th=2.0):
    """ Add 'binner' col to train classifier for filtering out non-dockers. """
    binner = [1 if (x >= bin_th) else 0 for x in dd_trg[score_name]]
    dd_trg.insert(loc=1, column='binner', value=binner)
    return dd_trg


def trn_baseline(ml_df, fea_list=['dd'], fea_sep='_'):
    """ Train baseline model using LGBM. """
    try:
        import lightgbm as lgb
    except ImportError:
        print('Could not import lightgbm.')
        return None

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
    te_scr = calc_scores(y_true=y_true, y_pred=y_pred, mltype='reg', metrics=None)
    return te_scr


def gen_ml_data(fpath,
                common_samples,
                fea_type,
                drg_set,
                meta_cols=['TITLE', 'SMILES'],
                fea_sep='_', score_name='reg',
                n_samples=None, n_top=None, flatten=False,
                q_cls=0.025, bin_th=2.0, print_fn=print, binner=False,
                frm=['parquet'], baseline=False,
                outdir=Path('out'), outfigs=Path('outfigs')):
    """ Generate a single set of ML data for the loaded target from fpath.
    This func was specifically created to process the new LARGE DOE-MD datasets
    with ZINC drugs that contains >6M molecules.
    Args:
        fpath: path to load docking scores file
        fea_df: df with features
        meta_cols: metadata columns to include in the dataframe
        score_name: rename the docking score col with score_name
        q_cls: quantile value to compute along the docking scores to generate the 'cls' col
        bin_th: threshold value of docking score to generate the 'binner' col
        baseline: whether to compute ML baseline scores
        n_samples: total number of samples in the final ml_df
        n_top: keep this number of top-most dockers
        flatten: if True, extract dock scores such that the final histogram of
                 scores is more uniform.
                 TODO: at this point, this is used only if n_samples is not None.

    Returns:
        ml_df: the ML dataframe
    """
    print_fn(f'\nProcess {fpath.name} ...')
    res = {}
    trg_name = fpath.with_suffix('').name  # note! depends on dock file names
    res['target'] = trg_name

    # Load docking
    dock = load_data(fpath)
    if dock.empty:
        print_fn('Empty file')
        return None

    # Pre-proc the dock file
    ID = 'TITLE'
    dock = dock.rename(columns={'Chemgauss4': score_name})  # note! Chemgauss4 might be different
    dock = dock[ dock[ID].notna() ].reset_index(drop=True)  # drop TITLE==nan
    dock[score_name] = dock[score_name].map(lambda x: cast_to_float(x) )  # cast scores to float
    dock = dock[ dock[score_name].notna() ].reset_index(drop=True)  # drop non-float
    dock[score_name] = abs( np.clip(dock[score_name], a_min=None, a_max=0) )  # convert and bound to >=0
    print_fn('dock: {}'.format(dock.shape))

    # Plot histogram of all scores
    outfigs_all = outfigs/'all'
    os.makedirs(outfigs_all, exist_ok=True)
    fig, ax = plt.subplots()
    ax.hist(dock[score_name], bins=50, facecolor='b', alpha=0.7);
    ax.set_xlabel('Dock Score')
    ax.set_ylabel('Count')
    plt.grid(True)
    plt.title(f'Samples {len(dock)}')
    plt.savefig(outfigs_all/f'dock.dist.all.{trg_name}.png')

    # -----------------------------------------
    # Sample a subset of scores
    # -------------------------
    # old!
    # Start merging docks and features using only a subset of cols that are
    # necessary to perform the merge.
    # merger = ['TITLE', 'SMILES']
    # aa = pd.merge(dock, fea_df[merger], how='inner', on=merger)
    # aa = aa.sort_values('reg', ascending=False).reset_index(drop=True)

    # new!
    # Extract samples that are common to all feature types
    aa = dock[ dock[ID].isin(common_samples) ].reset_index(drop=True)

    # Extract subset of samples
    if (n_samples is not None) and (n_top is not None):
        n_bot = n_samples - n_top

        aa = aa.sort_values('reg', ascending=False).reset_index(drop=True)
        df_top = aa[:n_top].reset_index(drop=True)  # e.g. 100K
        df_rest = aa[n_top:].reset_index(drop=True)

        if flatten:
            df_bot = flatten_dist(df=df_rest, n=n_bot, score_name=score_name)
        else:
            df_bot = df_rest.sample(n=n_bot, replace=False)

        assert df_top.shape[1] == df_bot.shape[1], 'Num cols must be the same when concat.'
        aa = pd.concat([df_top, df_bot], axis=0).reset_index(drop=True)

        # Plot histogram of sampled scores
        outfigs_sampled = outfigs/'sampled'
        os.makedirs(outfigs_sampled, exist_ok=True)
        fig, ax = plt.subplots()
        ax.hist(df_bot[score_name], bins=50, facecolor='b', alpha=0.7, label='The rest (balanced)');
        ax.hist(df_top[score_name], bins=50, facecolor='r', alpha=0.7, label='Top dockers');
        ax.set_xlabel('Dock Score')
        ax.set_ylabel('Count')
        plt.grid(True)
        plt.legend(loc='best', framealpha=0.5)
        plt.title(f'Samples {n_samples}; n_top {n_top}')
        plt.savefig(outfigs_sampled/f'dock.dist.sampled.{trg_name}.png')
        del df_top, df_bot, df_rest

    elif (n_samples is not None):
        if flatten:
            aa = flatten_dist(df=aa, n=n_samples, score_name=score_name)
        else:
            aa = aa.sample(n=n_samples, replace=False)

        # Plot histogram of sampled scores
        outfigs_sampled = outfigs/'sampled'
        os.makedirs(outfigs_sampled, exist_ok=True)
        fig, ax = plt.subplots()
        ax.hist(aa[score_name], bins=50, facecolor='b', alpha=0.7);
        ax.set_xlabel('Dock Score')
        ax.set_ylabel('Count')
        plt.grid(True)
        plt.title(f'Samples {n_samples}')
        plt.savefig(outfigs_sampled/f'dock.dist.sampled.{trg_name}.png')

    dock = aa
    del aa

    # -----------------------------------------
    # Create cls col
    # --------------
    # Find quantile value
    if dock[score_name].min() >= 0:  # if scores were transformed to >=0
        q_cls = 1.0 - q_cls
    cls_th = dock[score_name].quantile(q=q_cls)
    res['cls_th'] = cls_th
    print_fn('Quantile score (q_cls={:.3f}): {:.3f}'.format(q_cls, cls_th))

    # Generate a classification target col
    if dock[score_name].min() >= 0:  # if scores were transformed to >=0
        value = (dock[score_name] >= cls_th).astype(int)
    else:
        value = (dock[score_name] <= cls_th).astype(int)
    dock.insert(loc=1, column='cls', value=value)
    # print_fn('Ratio {:.2f}'.format( dd['dock_bin'].sum() / dd.shape[0] ))

    # Plot
    # hist, bin_edges = np.histogram(dock[score_name], bins=bins)
    # x = np.ones((10,)) * cls_th
    # y = np.linspace(0, hist.max(), len(x))

    # fig, ax = plt.subplots()
    # plt.hist(dock[score_name], bins=bins, density=False, facecolor='b', alpha=0.5)
    # plt.title(f'Scores clipped to 0: {trg_name}');
    # plt.ylabel('Count'); plt.xlabel('Docking Score');
    # plt.plot(x, y, 'r--', alpha=0.7, label=f'{q_cls}-th quantile')
    # plt.grid(True)
    # plt.savefig(outfigs/f'dock.score.{trg_name}.png')
    # -----------------------------------------

    # Save dock scores
    cols = ['Inchi-key', 'SMILES', 'TITLE', 'reg', 'cls']
    dock = dock[cols]
    trg_outdir = outdir/f'DIR.ml.{trg_name}'
    outpath = trg_outdir/f'docks.df.{trg_name}.csv'
    os.makedirs(trg_outdir, exist_ok=True)
    dock.to_csv(outpath, index=False)

    # Merge scores with features
    # bb = fea_df[ fea_df[ID].isin( dock[ID] ) ].reset_index(drop=True)
    # ml_df = pd.merge(dock, bb, how='inner', on=merger).reset_index(drop=True)

    # if n_samples is not None:
    #     assert n_samples == ml_df.shape[0], 'Final ml_df size must match n_samples {}'.format(fpath)

    # Add binner (note! may not be necessary since we get good dock scores)
    # if binner:
    #     dock = add_binner(dock, score_name=score_name, bin_th=bin_th)

    """
    At this point, the drug names that we're interested in is in common_samples.
    Also, the dock var contains the filtered set of docking scores.
    Thus, we can further reduce the list of drug names of our intereset by
    taking the intersect of drugs available in dock and the list common_samples.
    Then, we need to extract the appropriate features for each feature set of
    the remaining drugs.
    """
    drug_names = set(common_samples).intersection(set(dock[ID].values))

    def load_and_get_samples(f, cols, col_name):
        """ Load a subset of features and retain samples of interest. """
        df = pd.read_csv(f, names=cols)
        df = df[ df[col_name].isin(drug_names) ]
        return df

    def fps_to_nparr(x):
        """ Convert fps strings (base64) to integers. """
        import base64
        from rdkit.Chem import DataStructs
        x = DataStructs.ExplicitBitVect( base64.b64decode(x) )
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(x, arr)
        return arr

    # Merge only on TITLE (when including also SMILES, there is a mismatch on
    # certain samples; maybe smiles that come with features are canonicalied)
    merger = ['TITLE']

    for fea_name in fea_type:
        if 'descriptors' == fea_name:
            files_path = Path(FEA_DIR, drg_set, fea_name).resolve()
            fea_files = sorted(files_path.glob(f'{drg_set}-*.csv'))

            if len(fea_files) > 0:
                fea_prfx = 'dd'
                fea_names = pd.read_csv(FEA_DIR/'dd_fea_names.csv').columns.tolist()
                fea_names = [c.strip() for c in fea_names]  # clean names
                fea_names = [fea_prfx+fea_sep+str(c) for c in fea_names]  # prefix fea names
                cols = ['CAT', 'TITLE', 'SMILES'] + fea_names
                # cols = ['CAT', 'TITLE'] + fea_names

                dfs = Parallel(n_jobs=32, verbose=10)(
                    delayed(load_and_get_samples)(f, cols, col_name=ID) for f in fea_files
                )
                # dfs = []
                # for f in fea_files:
                #     df = load_and_get_samples(f, cols, col_name=ID)
                #     dfs.append(df)
                fea_df = pd.concat(dfs, axis=0).reset_index(drop=True)
                fea_df.drop(columns='SMILES', inplace=True)
                del dfs

                # Merge scores with features
                ml_df = pd.merge(dock, fea_df, how='inner', on=merger).reset_index(drop=True)
                del fea_df

                # Re-org cols
                fea_cols = extract_subset_fea_col_names(ml_df, fea_list=[fea_prfx], fea_sep=fea_sep)
                meta_cols = ['Inchi-key', 'SMILES', 'TITLE', 'CAT', 'reg', 'cls']
                cols = meta_cols + fea_cols
                ml_df = ml_df[cols]
                print_fn('descriptors: {}'.format(ml_df.shape))

                # Save
                outpath = trg_outdir/f'ml.{trg_name}.{fea_name}'
                ml_df.to_parquet(str(outpath)+'.parquet')

                # Compute baseline
                if baseline:
                    te_scr = trn_baseline(ml_df, fea_list=[fea_prfx], fea_sep=fea_sep)
                    res['dd_r2'] = te_scr['r2']
                    res['dd_mae'] = te_scr['median_absolute_error']
                    del te_scr

                res['min'], res['max'] = ml_df[score_name].min(), ml_df[score_name].max()
                del ml_df

        elif 'fps' == fea_name:
            files_path = Path(FEA_DIR, drg_set, fea_name).resolve()
            fea_files = sorted(files_path.glob(f'{drg_set}-*.csv'))

            if len(fea_files) > 0:
                fea_prfx = 'ecfp2'
                # cols = ['CAT', 'TITLE', 'SMILES', 'fps']
                cols = ['CAT', 'TITLE', 'fps']

                dfs = Parallel(n_jobs=32, verbose=10)(
                    delayed(load_and_get_samples)(f, cols, col_name=ID) for f in fea_files
                )
                fea_df = pd.concat(dfs, axis=0).reset_index(drop=True)
                del dfs

                aa = Parallel(n_jobs=32, verbose=10)(
                    delayed(fps_to_nparr)(x) for x in fea_df['fps'].values
                )
                fea_names = [fea_prfx+fea_sep+str(i+1) for i in range(len(aa[0]))]  # prfx fea names
                cols = ['CAT', 'TITLE', 'SMILES'] + fea_names
                aa = pd.DataFrame(np.vstack(aa), columns=fea_names)
                meta = fea_df.drop(columns='fps')
                fea_df = pd.concat([meta, aa], axis=1)
                del aa, meta

                # Merge scores with features
                ml_df = pd.merge(dock, fea_df, how='inner', on=merger).reset_index(drop=True)
                del fea_df

                # Re-org cols
                fea_cols = extract_subset_fea_col_names(
                    ml_df, fea_list=[fea_prfx], fea_sep=fea_sep)
                meta_cols = ['Inchi-key', 'SMILES', 'TITLE', 'CAT', 'reg', 'cls']
                cols = meta_cols + fea_cols
                ml_df = ml_df[cols]
                print_fn('fps (ecfp2): {}'.format(ml_df.shape))

                # Save
                outpath = trg_outdir/f'ml.{trg_name}.{fea_name}'
                ml_df.to_parquet(str(outpath) + '.parquet')

                # Compute baseline if specified
                if baseline:
                    te_scr = trn_baseline(ml_df, fea_list=[fea_prfx], fea_sep=fea_sep)
                    res['ecfp2_r2'] = te_scr['r2']
                    res['ecfp2_mae'] = te_scr['median_absolute_error']
                    del te_scr

                del ml_df

        elif 'images' == fea_name:
            # TODO
            pass

    # # Re-org cols
    # fea_cols = extract_subset_fea_col_names(ml_df, fea_list=fea_list, fea_sep=fea_sep)
    # meta_cols = ['Inchi-key', 'SMILES', 'TITLE', 'CAT', 'reg', 'cls']
    # cols = meta_cols + fea_cols
    # ml_df = ml_df[ cols ]

    # # New! Save docks only
    # dock_df = ml_df[ meta_cols ]
    # trg_outdir = outdir/f'DIR.ml.{trg_name}'
    # outpath = trg_outdir/f'docks.df.{trg_name}.csv'
    # os.makedirs(trg_outdir, exist_ok=True)
    # dock_df.to_csv(outpath, index=False)

    # # Extract the features
    # def extract_and_save_fea( df, fea, frm=['parquet'] ):
    #     """ Extract specific feature type (including metadata) and
    #     save to file.
    #     """
    #     fea_prfx_drop = [i for i in fea_list if i!=fea]
    #     fea_cols_drop = extract_subset_fea_col_names(df, fea_list=fea_prfx_drop, fea_sep=fea_sep)
    #     data = df.drop( columns=fea_cols_drop )

    #     frm = [i.lower() for i in frm]
    #     if 'none' in frm:
    #         return data

    #     # Outdir
    #     trg_outdir = outdir/f'DIR.ml.{trg_name}'
    #     os.makedirs(trg_outdir, exist_ok=True)
    #     outpath = trg_outdir/f'ml.{trg_name}.{fea}'

    #     for f in frm:
    #         if f == 'parquet':
    #             data.to_parquet( str(outpath)+'.parquet' )
    #         elif f == 'feather':
    #             data.to_feather( str(outpath)+'.feather' )
    #         elif f == 'csv':
    #             data.to_csv( str(outpath)+'.csv', index=False )
    #     return data

    # print_fn( 'Create and save df ...' )
    # for fea in fea_list:
    #     ml_df = extract_and_save_fea( ml_df, fea=fea, frm=frm )

    # res['min'], res['max'] = ml_df[score_name].min(), ml_df[score_name].max()
    # if baseline:
    #     te_scr = trn_baseline(df=ml_df, fea_list=fea_list, fea_sep=fea_sep)
    #     res['r2'] = te_scr['r2']
    #     res['mae'] = te_scr['median_absolute_error']

    return res


def run(args):
    # import pdb; pdb.set_trace()
    t0 = time()

    drg_set = Path(args.drg_set)
    scr_dir = Path(args.scr_dir).resolve()
    fea_type = args.fea_type
    # fea_list = args.fea_list  # TODO: not sure if we need this

    par_jobs = int(args.par_jobs)
    assert par_jobs > 0, f"The arg 'par_jobs' must be int >0 (got {par_jobs})"

    if args.outdir is not None:
        outdir = Path(args.outdir).resolve()
    else:
        batch_name = scr_dir.parent.name
        outdir = Path(GOUT, batch_name).resolve()

    outfigs = outdir/'figs'
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outfigs, exist_ok=True)

    # Logger
    lg = Logger(outdir/'gen.ml.data.log')
    print_fn = get_print_func(lg.logger)
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(vars(args))}')

    print_fn(f'\nDocking files  {scr_dir}')
    print_fn(f'Features dir   {FEA_DIR}')
    print_fn(f'Outdir         {outdir}')

    # ========================================================
    # Get (glob) the docking files
    # ----------------------------
    scr_dir = Path(scr_dir, drg_set).resolve()
    scr_file_pattern = '*4col.csv'
    scr_files = sorted(scr_dir.glob(scr_file_pattern))
    # scr_files = scr_files[:2]

    # ========================================================
    # Get the common samples (by ID)
    # ------------------------------
    """
    For each feature type (descriptors, fps, images), obtain the list
    of drug names for which the features are available. Also, get the
    intersect of drugs names across the feature types. This is required
    for multimodal learning (we want to make sure that we have all the
    feature types for a drug).
    """
    def load_and_get_names(f, cols, col_name):
        """
        Load a file (f) that contains a subset of features and get drug names
        from a col col_name.
        """
        df = pd.read_csv(f, names=cols, usecols=[0, 1, 2])
        return df[col_name].values.tolist()

    # N = 7
    N = None
    ID = 'TITLE'
    for fea_name in fea_type:
        if 'descriptors' == fea_name:
            files_path = Path(FEA_DIR, drg_set, fea_name).resolve()
            fea_files = sorted(files_path.glob(f'{drg_set}-*.csv'))

            if len(fea_files) > 0:
                cols = ['CAT', 'TITLE', 'SMILES']

                # Returns list of list (of drug names)
                id_names = Parallel(n_jobs=32, verbose=10)(
                    delayed(load_and_get_names)(f, cols, col_name=ID) for f in fea_files[:N]
                )

                # Flatten the list of lists
                id_names = [item for sublit in id_names for item in sublit]
                dd_names = set(id_names)
                del id_names

        if 'fps' == fea_name:
            files_path = Path(FEA_DIR, drg_set, fea_name).resolve()
            fea_files = sorted( files_path.glob(f'{drg_set}-*.csv') )

            if len(fea_files) > 0:
                cols = ['CAT', 'TITLE', 'SMILES']

                # Returns list of list (of drug names)
                id_names = Parallel(n_jobs=32, verbose=10)(
                    delayed(load_and_get_names)(f, cols, col_name=ID) for f in fea_files[:N]
                )

                # Flatten the list of lists
                id_names = [item for sublit in id_names for item in sublit]
                fps_names = set(id_names)
                del id_names

        if 'images' == fea_name:
            # TODO: haven't finished testing this!
            files_path = Path(FEA_DIR, drg_set, fea_name).resolve()
            fea_files = sorted(files_path.glob(f'{drg_set}-*.pkl'))

            if len(fea_files) > 0:
                id_names = []

                for i, f in enumerate(fea_files[:N]):
                    if (i+1) % 100 == 0:
                        print(f'Load {i+1} ...')
                    imgs = pickle.load(open(fea_files[i], 'rb'))
                    tmp = [item[1] for item in imgs]
                    if '' in tmp:  # if ID is missing
                        jj = np.where(np.array(tmp) == '')[0][0]
                        print(imgs[jj])
                        continue
                    id_names.extend( tmp )
                    tmp = [item[2] for item in imgs]

                img_names = set(id_names)
                del id_names

    # Union of TITLE names across all features types
    all_names = []
    if 'descriptors' in fea_type:
        all_names.extend(list(dd_names))
    if 'fps' in fea_type:
        all_names.extend(list(fps_names))
    if 'images' in fea_type:
        all_names.extend(list(img_names))
    print_fn(f'Union of titles across all feature types: {len(set(all_names))}')

    # Intersect of TITLE names across all features types
    common_names = dd_names
    if 'fps' in fea_type:
        common_names.intersection(fps_names)
    if 'images' in fea_type:
        common_names.intersection(img_names)
    print_fn(f'Intersect of titles across all feature types: {len(set(common_names))}')

    # Get TITLEs that are not available across all feature types
    bb_names = list(set(all_names).difference(common_names))
    if len(bb_names) > 0:
        # TODO consider to dump these titles!
        print_fn(f'Difference of titles across all feature types: {len(set(bb_names))}')

    # ========================================================
    kwargs = {'common_samples': common_names,
              'fea_type': fea_type,
              'drg_set': drg_set,
              'meta_cols': meta_cols,
              'print_fn': print_fn,
              'outdir': outdir,
              'outfigs': outfigs,
              'baseline': args.baseline,
              'n_samples': args.n_samples,
              'n_top': args.n_top,
              'frm': args.frm,
              'flatten': args.flatten,
              }

    if par_jobs > 1:
        results = Parallel(n_jobs=par_jobs, verbose=20)(
            delayed(gen_ml_data)(fpath=f, **kwargs) for f in scr_files )
    else:
        results = []  # dock summary including ML baseline scores
        for f in scr_files:
            res = gen_ml_data(fpath=f, **kwargs)
            results.append(res)

    results = [r for r in results if r is not None]
    results = np.round(pd.DataFrame(results), decimals=3)
    results.sort_values('target').reset_index(drop=True)
    results.to_csv(outdir/'dock.ml.baseline.csv', index=False)

    # ========================================================
    if (time()-t0)//3600 > 0:
        print_fn('\nRuntime: {:.1f} hrs'.format((time()-t0)/3600))
    else:
        print_fn('\nRuntime: {:.1f} min'.format((time()-t0)/60))
    print_fn('Done.')
    lg.kill_logger()


def main(args):
    args = parse_args(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
