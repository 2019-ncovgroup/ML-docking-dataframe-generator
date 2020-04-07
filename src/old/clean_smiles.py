"""
This code takes the original smiles file ena+db.smi, cleans data, and saves.
The original file is not overwritten.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time
import pandas as pd

filepath = Path(__file__).resolve().parent

# Utils
sys.path.append( filepath/'../utils' )
# from utils.classlogger import Logger
# from utils.smiles import canon_single_smile, canon_df
from classlogger import Logger
from smiles import canon_single_smile, canon_df

datadir = Path('/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/raw')
# outdir  = Path('/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/processed/descriptors')
outdir  = Path('/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/processed/descriptors/ena_db')
os.makedirs(outdir, exist_ok=True)


t0 = time()
lg = Logger(outdir/'clean.smiles.log')
print_fn = lg.logger.info

print_fn('\nPython filepath    {}'.format( filepath ))
print_fn('Original data dir {}'.format( datadir ))
print_fn('Output data dir   {}'.format( outdir ))

# File names
in_fname = 'ena+db.smi'
in_fpath  = datadir / in_fname
out_fpath = outdir / (in_fname+'.can.csv')

# Original non-canonical SMILES of the combined library of Enamine Diversity
# and DrugBank. (Rick added IDs to the drugbank entries db-x).
print_fn('\nLoad {}'.format( in_fpath ))
smi = pd.read_csv(in_fpath, header=None, names=['smiles', 'name'],
                  sep=' |\t') # use multiple separators(!!)
smi['smiles'] = smi['smiles'].map(lambda x: x.strip())
print_fn( smi.shape )
print_fn('Total duplicates (do not remove): {}'.format( sum(smi.duplicated()) ))

# Drop duplicates (all)
# cnt0 = smi.shape[0]; print('\nCount: ', cnt0)
# smi = smi.drop_duplicates().reset_index(drop=True)
# cnt1 = smi.shape[0]; print('Count: ', cnt1); print('Dropped duplicates: ', cnt0-cnt1)

# Drop duplicates (consider only smiles)
# cnt0 = smi.shape[0]; print('\nCount: ', cnt0)
# smi = smi.drop_duplicates(subset=['smiles']).reset_index(drop=True)
# cnt1 = smi.shape[0]; print('Count: ', cnt1); print('Dropped duplicates: ', cnt0-cnt1)

# Canonicalize
print_fn('\nCanonicalize ...')
# smi_vec = smi[['smiles']].copy()
smi_can = canon_df(smi, smi_name='smiles')
# smi['smiles'] = smi_vec

smi_can = smi_can[ ~smi_can['smiles'].isna() ] # keep good smiles
smi_can = smi_can.reset_index( drop=True )

# Save
print_fn('\nSave ...')
print_fn( smi_can.shape )
# smi = smi.reset_index(drop=True)
smi_can.to_csv(out_fpath, index=False)

print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
print_fn('Done.')
lg.kill_logger()


