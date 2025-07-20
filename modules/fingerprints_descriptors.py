# Importing Libraries
import pandas as pd
import numpy as np
import warnings

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, PandasTools, MACCSkeys, AtomPairs, rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem.rdmolops import PatternFingerprint
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprintAsBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors

pd.set_option('display.max_rows', None)

from tqdm import tqdm

warnings.filterwarnings("ignore")


class FingerprintGenerator:
    def __init__(self, mol_column):
        # self.smiles_list = smiles_list
        # self.molecules = [Chem.MolFromSmiles(smile) for smile in smiles_list]
        self.molecules = mol_column

    def generate_maccs(self):
        maccs_fp = []
        for mol in tqdm(self.molecules, desc='Generating MACCS'):
            maccs_bitvector = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(maccs_bitvector, arr)
            maccs_fp.append(arr)
        return pd.DataFrame(maccs_fp)

    def generate_pattern(self):
        pattern_fp = []
        for mol in tqdm(self.molecules, desc='Generating Pattern'):
            pf_bitvector = Chem.PatternFingerprint(mol)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(pf_bitvector, arr)
            pattern_fp.append(arr)
        return pd.DataFrame(pattern_fp)

    def generate_morgan(self, radius=1, nBits=2048):
        morgan_fp = []
        for mol in tqdm(self.molecules, desc='Generating Morgan'):
            mf_bitvector = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(mf_bitvector, arr)
            morgan_fp.append(arr)
        return pd.DataFrame(morgan_fp)

    def generate_ecfp(self, radius=2, nBits=2048):
        ecfp_fp = []
        for mol in tqdm(self.molecules, desc='Generating ECFP'):
            ecfp_bitvector = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(ecfp_bitvector, arr)
            ecfp_fp.append(arr)
        return pd.DataFrame(ecfp_fp)

    def generate_avalon(self, nBits=512):
        avalon_fp = []
        for mol in tqdm(self.molecules, desc='Generating Avalon'):
            af_bitvector = pyAvalonTools.GetAvalonFP(mol, nBits=nBits)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(af_bitvector, arr)
            avalon_fp.append(arr)
        return pd.DataFrame(avalon_fp)

    def generate_rdkit(self, fpSize=4096):
        rdik_fp = []
        rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpSize)
        for mol in tqdm(self.molecules, desc='Generating RDKit'):
            rdk_bitvector = rdkgen.GetFingerprint(mol)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(rdk_bitvector, arr)
            rdik_fp.append(arr)
        return pd.DataFrame(rdik_fp)

    def generate_atom_pair(self, fpSize=4096):
        atom_pair_fp = []
        apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fpSize)
        for mol in tqdm(self.molecules, desc='Generating Atom Pair'):
            apf_bitvector = apgen.GetFingerprint(mol)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(apf_bitvector, arr)
            atom_pair_fp.append(arr)
        return pd.DataFrame(atom_pair_fp)

    def generate_topological_torsion(self, fpSize=2048):
        topological_torsion_fp = []
        ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fpSize)
        for mol in tqdm(self.molecules, desc='Generating Topological Torsion'):
            ttf_bitvector = ttgen.GetFingerprint(mol)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(ttf_bitvector, arr)
            topological_torsion_fp.append(arr)
        return pd.DataFrame(topological_torsion_fp)



class DescriptorGenerator:
    def __init__(self, mol_column):
        self.df = df
        self.molecules = mol_column

    def generate_rdkit_descriptors(self):
        Desc_list_func = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        names = Desc_list_func.GetDescriptorNames()
        Des_func = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        rdkit_descriptors = []

        for mol in tqdm(self.molecules, desc='Generating RDKit Descriptors'):
            rdkit_descriptors.append(Des_func.CalcDescriptors(mol))

        return pd.DataFrame(rdkit_descriptors, columns=names)
