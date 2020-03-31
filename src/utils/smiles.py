from time import time
from rdkit import Chem


def canon_single_smile(smi):
    try:
        cannon = Chem.MolFromSmiles(smi)
        if cannon is not None:
            return Chem.MolToSmiles(cannon, canonical=True)
        else:
            print(smi)
    except:
        print('error', smi)
    return smi


def canon_df(df, smi_name='smiles'):
    t0 = time()
    for i, s in enumerate(df[smi_name]):
        if i%50000==0:
            print('{}: {:.2f} sec'.format(i, time()-t0))
            t0 = time()
        df.loc[i, 'smiles'] = canon_single_smile(s)
    return df


