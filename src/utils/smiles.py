from time import time
import numpy as np


def canon_single_smile(smi):
    """ ... """
    from rdkit import Chem
    try:
        # canon = Chem.MolFromSmiles(smi)
        # if canon is not None:
        #     return Chem.MolToSmiles(canon, canonical=True)
        # else:
        #     print(smi)
        # (ap)
        mol = Chem.MolFromSmiles(smi)
        can_smi = Chem.MolToSmiles(mol, canonical=True)
    except:
        # print('error', smi)
        print(f'Error: smiles={smi}')
        can_smi = np.nan
    return can_smi


def canon_df(df, smi_name='smiles'):
    """ ... """
    smi_vec = []
    t0 = time()
    # for i, s in enumerate(smi_vec[smi_name]):
    for i, s in enumerate(df[smi_name].values):
        if i%50000==0:
            print('{}: {:.2f} sec'.format(i, time()-t0))
            t0 = time()
        can_smi = canon_single_smile(s)
        smi_vec.append( can_smi ) # TODO: consider return this, instead of modifying df
        # df.loc[i, 'smiles'] = can_smi
        
    df.loc[:, 'smiles'] = smi_vec
    return df


# def modred_single_smile(smi):
#     """ ... """
#     # from rdkit import Chem
#     # from mordred import Calculator, descriptors
#     try:
#         mol = Chem.MolFromSmiles(smi)
#         # can_smi = Chem.MolToSmiles(canon, canonical=True)
#     except:
#         # print('error', smi)
#         print(f'Error: smiles={smi}')
#         mol = np.nan
#     return mol


# def smi_to_mordred(df, smi_name='smiles'):
#     """ ... """    
#     from rdkit import Chem
#     from mordred import Calculator, descriptors
#     t0 = time()
#     # create descriptor calculator with all descriptors
#     calc = Calculator(descriptors, ignore_3D=True)
#     # print( len(calc.descriptors) )
#     # print( len(Calculator(descriptors, ignore_3D=True, version="1.0.0")) )
    
#     mol_vec = []
#     for i, s in enumerate(df[smi_name]):
#         if i%50000==0:
#             print('{}: {:.2f} sec'.format(i, time()-t0))
#             t0 = time()
#         df.loc[i, 'smiles'] = canon_single_smile(s)
        
#         mol = canon_single_smile(s)
#         df.loc[i, 'smiles'] = can_smi
#     df = calc.pandas(mols)
#     return df

