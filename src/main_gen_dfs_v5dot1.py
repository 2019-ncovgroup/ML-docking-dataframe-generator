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
                        choices=['OZD', 'ORD'],
                        help=f'Drug set (default: {DRG_SET}).')
    parser.add_argument('-sd', '--scr_dir',
                        type=str,
                        default=str(SCR_DIR),
                        help=f'Path to docking scores file (default: {SCR_DIR}).')

    # parser.add_argument('--fea_dir',
    #                     type=str,
    #                     default=str(FEA_DIR),
    #                     help=f'Path to molecular features file (default: {FEA_DIR}).')
    # parser.add_argument('--fea_type',
    #                     type=str,
    #                     default=['descriptors'],
    #                     nargs='+',
    #                     choices=['descriptors', 'images', 'fps'],
    #                     help='Feature type (default: descriptors).')

    parser.add_argument('-od', '--outdir',
                        type=str,
                        default=None,
                        help=f'Output dir (default: {GOUT}/<batch_name>).')

    parser.add_argument('--dd_fpath',
                        type=str,
                        default=None,
                        help=f'Path to aggregated file of drug descriptors (default: None).')
    parser.add_argument('--fps_fpath',
                        type=str,
                        default=None,
                        help=f'Path to aggregated file of ecfp2 fingerprints (default: None).')
    parser.add_argument('--img_fpath',
                        type=str,
                        default=None,
                        help=f'Path to img files (default: None).')

    parser.add_argument('--n_samples',
                        type=int,
                        default=None,
                        help='Number of docking scores to get into the ML df (default: None).')
    parser.add_argument('--n_top',
                        type=int,
                        default=None,
                        help='Number of top-most docking scores. This is irrelevant if n_samples \
                        was not specified (default: None).')
    # parser.add_argument('--flatten',
    #                     action='store_true',
    #                     help='Flatten the distribution of docking scores (default: False).')
    parser.add_argument('--sampling',
                        default=None,
                        choices=[None, 'random', 'flatten'],
                        help='Sampling approach of scores (default: None).')

    parser.add_argument('--baseline',
                        action='store_true',
                        help='Number of drugs to get from features dataset (default: None).')
    # parser.add_argument('--frm',
    #                     type=str,
    #                     nargs='+',
    #                     default=['parquet'],
    #                     choices=['parquet', 'feather', 'csv', 'none'],
    #                     help='Output file format for ML dfs (default: parquet).')
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
    model = ml_model_def(**ml_init_args)
    ydata = ml_df['reg']
    xdata = extract_subset_fea(ml_df, fea_list=fea_list, fea_sep=fea_sep)
    x_, xte, y_, yte = train_test_split(xdata, ydata, test_size=0.2)
    xtr, xvl, ytr, yvl = train_test_split(x_, y_, test_size=0.2)
    ml_fit_args['eval_set'] = (xvl, yvl)
    model.fit(xtr, ytr, **ml_fit_args)
    y_pred, y_true = calc_preds(model, x=xte, y=yte, mltype='reg')
    te_scr = calc_scores(y_true=y_true, y_pred=y_pred, mltype='reg', metrics=None)
    return te_scr


def proc_dock_score(dock, ID: str='TITLE', score_name: str='reg',
                    scoring_func: str='Chemgauss4'):
    """ Process dock score column.
    Args:
        dock : df that contains the docking scores column
    """
    dock = dock.rename(columns={scoring_func: score_name})  # note! Chemgauss4 might be different
    dock = dock[dock[ID].notna()].reset_index(drop=True)  # drop TITLE==nan
    dock[score_name] = dock[score_name].map(lambda x: cast_to_float(x))  # cast scores to float
    dock = dock[dock[score_name].notna()].reset_index(drop=True)  # drop non-float
    return dock


def plot_hist_dock_scores(df, outfigs, subdir_name, trg_name,
                          score_name: str='reg',
                          scoring_func: str='Chemgauss4'):
    """ Plot histogram of docking scores and save the plot. """
    outfigs_dir = outfigs/subdir_name
    os.makedirs(outfigs_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.hist(df[score_name], bins=100, facecolor='b', alpha=0.7);
    ax.set_xlabel(f'Docking Score ({scoring_func})')
    ax.set_ylabel('Count')
    plt.title(f'{subdir_name}; Samples {len(df)}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfigs_dir/f'dock.dist.{trg_name}.png')


def load_and_get_samples(f, cols, col_name, drug_names=None):
    """ Load a subset of features and retain samples of interest.
    Args:
        f : file name
        cols : cols to read in
        col_name : ID col name for drug names (e.g. TITLE)
        drug_names : list of drug to extract
    """
    df = pd.read_csv(f, names=cols)
    if drug_names is not None:
        df = df[df[col_name].isin(drug_names)]
    return df


def gen_ml_data(fpath,
                common_samples,
                # fea_type,
                # drg_set,
                dd_fea=None,
                fps_fea=None,
                img_fea=None,
                fea_sep='_',
                score_name='reg',
                n_samples=None, n_top=None, sampling=None,
                q_cls=0.025,
                binner=False, bin_th=2.0,
                baseline=False,
                print_fn=print,
                outdir=Path('out'), outfigs=Path('outfigs')):
    """ Generate a single set of ML data for the loaded target from fpath.
    This func was specifically created to process the new LARGE DOE-MD datasets
    with ZINC drugs that contains >6M molecules.
    Args:
        fpath: path to load docking scores file
        common_samples : list of drug names that are commong to all features
                         types including dd_fea, fps_fea, and img_fea
        dd_fea : df of Mordred descriptors
        fps_fea : df pf ecfp2 fingerprints
        img_fea : image data (TODO: this is not supported yet!)
        fea_sep : separator between feature prefix string and feature name
        score_name : rename the docking score col with score_name
        n_samples : total number of samples in the final ml_df
        n_top : keep this number of top-most dockers
        flatten : if True, extract dock scores such that the final histogram of
                 scores is more uniform.
        q_cls : quantile value to compute along the docking scores to generate the 'cls' col
        bin_th : threshold value of docking score to generate the 'binner' col
        binner : add binner column
        baseline : whether to compute ML baseline scores

    Returns:
        res : results summary
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
    scoring_func = 'Chemgauss4'
    dock = proc_dock_score(dock, ID=ID, score_name=score_name,
                           scoring_func=scoring_func)

    # Plot histogram of all (raw) scores
    plot_hist_dock_scores(dock, outfigs=outfigs, subdir_name='all.raw',
                          trg_name=trg_name, scoring_func=scoring_func)

    # Convert and bound scores to >=0
    dock[score_name] = abs(np.clip(dock[score_name], a_min=None, a_max=0))
    print_fn('dock: {}'.format(dock.shape))

    # Plot histogram of all (transformed) scores
    plot_hist_dock_scores(dock, outfigs=outfigs, subdir_name='all.transformed',
                          trg_name=trg_name, scoring_func=scoring_func)

    # -----------------------------------------
    # Sample a subset of scores
    # -------------------------
    # Extract samples that are common to all feature types
    aa = dock[dock[ID].isin(common_samples)].reset_index(drop=True)

    # Extract subset of samples
    if (n_samples is not None) and (n_top is not None):
        n_bot = n_samples - n_top

        aa = aa.sort_values('reg', ascending=False).reset_index(drop=True)
        df_top = aa[:n_top].reset_index(drop=True)  # e.g. 100K
        df_rest = aa[n_top:].reset_index(drop=True)

        # if flatten:
        #     df_bot = flatten_dist(df=df_rest, n=n_bot, score_name=score_name)
        # else:
        #     df_bot = df_rest.sample(n=n_bot, replace=False)
        if sampling == 'flatten':
            df_bot = flatten_dist(df=df_rest, n=n_bot, score_name=score_name)
        elif sampling == 'random':
            df_bot = df_rest.sample(n=n_bot, replace=False)
        else:
            raise ValueError("'sampling' arg must be specified.")

        assert df_top.shape[1] == df_bot.shape[1], 'Num cols must be the same when concat.'
        aa = pd.concat([df_top, df_bot], axis=0).reset_index(drop=True)

        # Plot histogram of sampled scores
        outfigs_dir = outfigs/'sampled.transformed'
        os.makedirs(outfigs_dir, exist_ok=True)
        fig, ax = plt.subplots()
        ax.hist(df_top[score_name], bins=100, facecolor='r', alpha=0.7, label='Top 10K Docking Ligands');
        ax.hist(df_bot[score_name], bins=100, facecolor='b', alpha=0.7, label='Other Ligands (balanced)');
        ax.set_xlabel(f'Docking Score ({scoring_func})')
        ax.set_ylabel('Count')
        plt.grid(True)
        plt.legend(loc='best', framealpha=0.5)
        plt.title(f'sampled.transformed; Samples {n_samples}; n_top {n_top}')
        plt.savefig(outfigs_dir/f'dock.dist.{trg_name}.png', dpi=150)
        del df_top, df_bot, df_rest

    elif (n_samples is not None):
        # if flatten:
        #     aa = flatten_dist(df=aa, n=n_samples, score_name=score_name)
        # else:
        #     aa = aa.sample(n=n_samples, replace=False)
        if sampling == 'flatten':
            aa = flatten_dist(df=aa, n=n_samples, score_name=score_name)
        elif sampling == 'random':
            aa = aa.sample(n=n_samples, replace=False)
        else:
            raise ValueError("'sampling' arg must be specified.")

        plot_hist_dock_scores(dock, outfigs=outfigs, subdir_name='sampled.transformed',
                              trg_name=trg_name, scoring_func=scoring_func)
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
    hist, bin_edges = np.histogram(dock[score_name], bins=100)
    x = np.ones((10,)) * cls_th
    y = np.linspace(0, hist.max(), len(x))

    fig, ax = plt.subplots()
    plt.hist(dock[score_name], bins=200, density=False, facecolor='b', alpha=0.7)
    plt.title(f'Scores clipped to 0: {trg_name}')
    plt.xlabel(f'Docking Score ({scoring_func})')
    plt.ylabel('Count')
    plt.plot(x, y, 'm--', alpha=0.7, label=f'{q_cls}-th quantile')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfigs/f'dock.dist.cls.{trg_name}.png')
    # -----------------------------------------

    # Save dock scores
    cols = ['Inchi-key', 'SMILES', 'TITLE', 'reg', 'cls']
    dock = dock[[c for c in cols if c in dock.columns]]
    trg_outdir = outdir/f'DIR.ml.{trg_name}'
    outpath = trg_outdir/f'docks.df.{trg_name}.csv'
    os.makedirs(trg_outdir, exist_ok=True)
    dock.to_csv(outpath, index=False)

    # Add binner (note! may not be necessary since we get good dock scores)
    # if binner:
    #     dock = add_binner(dock, score_name=score_name, bin_th=bin_th)

    # Merge only on TITLE (when including also SMILES, there is a mismatch on
    # certain samples; maybe smiles that come with features are canonicalied)
    merger = ID

    def merge_dock_and_fea(dock, fea_df, fea_prfx, fea_sep,
                           merger='TITLE', fea_name=None, baseline=False):
        """ ... """
        # drug_names = set(common_samples).intersection(set(dock[ID].values))

        ml_df = pd.merge(dock, fea_df, how='inner', on=merger).reset_index(drop=True)
        del fea_df

        # bb = fea_df[ fea_df[merger].isin(dock[merger].tolist()) ].reset_index(drop=True)
        # xdata = extract_subset_fea(bb, fea_list=[fea_prfx], fea_sep=fea_sep)
        # bb = pd.concat([bb[merger], xdata], axis=1)  # keep only the merger meta col from fea_df

        # xdata = extract_subset_fea(fea_df, fea_list=[fea_prfx], fea_sep=fea_sep)
        # fea_df = pd.concat([fea_df[merger], xdata], axis=1)  # keep only the merger meta col from fea_df
        # ml_df = pd.merge(dock, fea_df, how='inner', on=merger).reset_index(drop=True)
        # del fea_df, xdata

        # Re-org cols
        fea_cols = extract_subset_fea_col_names(
            ml_df, fea_list=[fea_prfx], fea_sep=fea_sep)
        meta_cols = ['Inchi-key', 'SMILES', 'TITLE', 'CAT', 'reg', 'cls']
        cols = meta_cols + fea_cols
        # ml_df = ml_df[cols]
        ml_df = ml_df[[c for c in cols if c in ml_df.columns]]
        print_fn('{}: {}'.format(fea_name, ml_df.shape))

        # Save
        outpath = trg_outdir/f'ml.{trg_name}.{fea_name}'
        ml_df.to_parquet(str(outpath) + '.parquet')

        # Compute baseline if specified
        if baseline:
            te_scr = trn_baseline(ml_df, fea_list=[fea_prfx], fea_sep=fea_sep)
            res[f'{fea_prfx}_r2'] = te_scr['r2']
            res[f'{fea_prfx}_mae'] = te_scr['median_absolute_error']
            del te_scr

        del ml_df

    if dd_fea is not None:
        merge_dock_and_fea(dock, fea_df=dd_fea, fea_prfx='dd', fea_sep=fea_sep,
                           merger=ID, fea_name='descriptors', baseline=baseline)

    if fps_fea is not None:
        merge_dock_and_fea(dock, fea_df=fps_fea, fea_prfx='ecfp2', fea_sep=fea_sep,
                           merger=ID, fea_name='ecfp2', baseline=baseline)

    if img_fea is not None:
        pass

    # if n_samples is not None:
    #     assert n_samples == ml_df.shape[0], 'Final ml_df size must match n_samples {}'.format(fpath)
    return res


def run(args):
    import pdb; pdb.set_trace()
    t0 = time()

    drg_set = Path(args.drg_set)
    scr_dir = Path(args.scr_dir).resolve()
    # fea_type = args.fea_type

    ID = 'TITLE'

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
    # Glob the docking files
    # ----------------------
    scr_dir = Path(scr_dir, drg_set).resolve()
    # scr_file_pattern = '*4col.csv'
    scr_file_pattern = '*sorted*csv'
    scr_files = sorted(scr_dir.glob(scr_file_pattern))

    # ss = ['ADRP_6W02_A_1_H',
    #       'NSP10-16_6W61_AB_1_F',
    #       'NSP10-16_6W61_AB_2_F']

    # def fnc(f):
    #     for s in ss:
    #         if s in str(f):
    #             return True
    #     return False

    # scr_files = [f for f in scr_files if fnc(f)]

    # ========================================================
    # Load features
    # ------------------------------
    dd_names = None
    fps_names = None
    img_names = None

    if (args.dd_fpath is not None) and (args.dd_fpath.lower() != 'none'):
        dd_fea = load_data(args.dd_fpath)
        dd_names = dd_fea[ID].tolist()
        dd_fea = dd_fea.drop(columns='SMILES')
        # tmp = dd_fea.isna().sum(axis=0).sort_values(ascending=False)
        dd_fea = dd_fea.fillna(0)
    else:
        dd_fea = None
        dd_names = None

    if (args.fps_fpath is not None) and (args.fps_fpath.lower() != 'none'):
        fps_fea = load_data(args.fps_fpath)
        fps_names = fps_fea[ID].tolist()
        fps_fea = fps_fea.drop(columns='SMILES')
        # tmp = fps_fea.isna().sum(axis=0).sort_values(ascending=False)
        fps_fea = fps_fea.fillna(0)
    else:
        fps_fea = None
        fps_names = None

    if (args.img_fpath is not None) and (args.img_fpath.lower() != 'none'):
        # TODO
        pass
    else:
        img_fea = None
        img_names = None

    # ========================================================
    # Get the common samples (by ID)
    # ------------------------------
    """
    For each feature type (descriptors, fps, images), obtain the list
    of drug names for which the features are available. Also, get the
    intersect of drug names across the feature types. This is required
    for multimodal learning (we want to make sure that we have all the
    feature types for a compound).
    """
    # Union of TITLE names across all features types
    all_names = []
    for ii in [dd_names, fps_names, img_names]:
        if ii is not None:
            all_names.extend(list(ii))
    print_fn(f'Union of titles across all feature types: {len(set(all_names))}')

    # Intersect of TITLE names across all features types
    common_names = None
    for ii in [dd_names, fps_names, img_names]:
        if (common_names is not None) and (ii is not None):
            common_names = set(common_names).intersection(set(ii))
        elif (common_names is None) and (ii is not None):
            common_names = ii
    print_fn(f'Intersect of titles across all feature types: {len(set(common_names))}')

    # Get TITLEs that are not available across all feature types
    bb_names = list(set(all_names).difference(set(common_names)))
    if len(bb_names) > 0:
        # TODO consider to dump these titles!
        print_fn(f'Difference of titles across all feature types: {len(set(bb_names))}')

    # Retain the common samples in fea dfs
    if dd_fea is not None:
        dd_fea = dd_fea[dd_fea[ID].isin(common_names)]  # .reset_index(drop=True)
    if fps_fea is not None:
        fps_fea = fps_fea[fps_fea[ID].isin(common_names)]  # .reset_index(drop=True)

    # ========================================================
    kwargs = {'common_samples': common_names,
              # 'fea_type': fea_type,
              # 'drg_set': drg_set,
              'dd_fea': dd_fea,
              'fps_fea': fps_fea,
              'img_fea': img_fea,
              'print_fn': print_fn,
              'outdir': outdir,
              'outfigs': outfigs,
              'baseline': args.baseline,
              'n_samples': args.n_samples,
              'n_top': args.n_top,
              # 'flatten': args.flatten,
              'sampling': args.sampling,
              }

    if par_jobs > 1:
        results = Parallel(n_jobs=par_jobs, verbose=20)(
            delayed(gen_ml_data)(fpath=f, **kwargs) for f in scr_files)
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
