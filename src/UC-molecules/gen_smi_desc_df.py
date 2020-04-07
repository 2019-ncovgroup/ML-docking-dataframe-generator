"""
This script takes SMILES from the oath below, canonicalize, and generate mordred descriptors.
/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/raw/UC-molecules 

The UC-molecules are drugs that UC chemists have made in the past that we want to screen,
so they are not in any set yet. These SMILES are not related to any drugs we have screened.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

from rdkit import Chem
from mordred import Calculator, descriptors

filepath = Path(__file__).resolve().parent

# Utils
sys.path.append( os.path.abspath(filepath/'../utils') )
# from utils.classlogger import Logger
# from utils.smiles import canon_single_smile, canon_df
from classlogger import Logger
from smiles import canon_single_smile, canon_df

datadir = Path('/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/raw/UC-molecules')
outdir  = Path('/vol/ml/apartin/projects/covid-19/ML-docking-dataframe-generator/data/processed/descriptors/UC-molecules')
os.makedirs(outdir, exist_ok=True)


t0 = time()
lg = Logger(outdir/'gen.desc.log')
print_fn = lg.logger.info

print_fn('\nPython filepath   {}'.format( filepath ))
print_fn('Original data dir {}'.format( datadir ))
print_fn('Output data dir   {}'.format( outdir ))

# File names
in_fname = 'UC.smi'
in_fpath  = datadir / in_fname
out_smi_fpath = outdir / 'UC.smi.can'
out_desc_fpath = outdir / 'UC.smi.desc'

# Load
print_fn('\nLoad {}'.format( in_fpath ))
smi = pd.read_csv(in_fpath, sep='\t', header=None, names=['smiles', 'NA'])
smi['smiles'] = smi['smiles'].map(lambda x: x.strip())
print_fn( smi.shape )
print_fn( 'Unique smiles (original): {}'.format( len(smi['smiles'].unique()) ))

# ----------------------
#   Canonicalize
# ----------------------
print_fn( '\nCanonicalize ...' )
smi_can = canon_df(smi, smi_name='smiles')
print_fn( 'Unique smiles (canonicalized): {}'.format( len(smi_can['smiles'].unique()) ))

smi_can = smi_can[ ~smi_can['smiles'].isna() ] # keep good smiles
smi_can = smi_can.reset_index( drop=True )

print_fn( '\nDrop duplicates ...' )
print_fn( smi_can.shape )
smi_can = smi_can.drop_duplicates( subset=['smiles'] )
smi_can = smi_can.reset_index( drop=True )
print_fn( smi_can.shape )

smi_can.to_csv( str(out_smi_fpath)+'.csv', index=False)

# ----------------------
#   Mordred
# ----------------------
# Create descriptor calculator with all descriptors
calc = Calculator(descriptors, ignore_3D=True)
# print_fn( len(calc.descriptors) )
# print_fn( len(Calculator(descriptors, ignore_3D=True, version="1.0.0")) )

# SMILES to Mordred
mols = [Chem.MolFromSmiles(smi) for smi in smi_can['smiles'].values]
dsc = calc.pandas( mols ) # convert to dataframe
dsc = pd.concat([smi_can['smiles'], dsc], axis=1) 

# Cast columns (descriptors)
print_fn('\nCast descriptors to float ...')
dsc = dsc.astype({c: np.float32 for c in dsc.columns[1:]})

# Impute missing values
print_fn('\nImpute NaNs ...')
# dsc = dsc.reset_index(drop=True)
# dsc = impute_values(dsc, print_fn=print_fn) # ap's approach
dsc = dsc.fillna(0.0)
print_fn('Total NaNs: {}.'.format( dsc.isna().values.flatten().sum() ))

# Prefix desc names
# dsc.columns = ['name'] + ['mod.'+c for c in dsc.columns[1:]]
dsc.columns = ['smiles'] + ['mod.'+c for c in dsc.columns[1:]]

# Save
print_fn('\nSave ...')
print_fn( dsc.shape )
dsc = dsc.reset_index(drop=True)
dsc.to_parquet( str(out_desc_fpath)+'.parquet' )

print_fn('\nRuntime {:.2f} mins'.format( (time()-t0)/60 ))
print_fn('Done.')
lg.kill_logger()


