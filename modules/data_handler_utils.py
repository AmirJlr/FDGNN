import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles
from torch_geometric.utils import degree

import os
from tqdm.notebook import tqdm

import deepchem as dc

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split

from molfeat.calc import FPCalculator, RDKitDescriptors2D, Pharmacophore2D, Pharmacophore3D, RDKitDescriptors3D
import datamol as dm
from molfeat.trans import MoleculeTransformer

from sklearn.decomposition import PCA

import signal

from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict



class DTsetBasicMulti(InMemoryDataset):
    def __init__(self, root, filename, smiles_column,
                 ECFP, Topological, MACCS, EState, Rdkit2D, Phar2D):
        self.filename = filename
        self.smiles_column = smiles_column

        self.ECFP = ECFP
        self.Topological = Topological
        self.MACCS = MACCS
        self.EState = EState
        self.Rdkit2D = Rdkit2D
        self.Phar2D = Phar2D

        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.filename]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass  # Implement download logic if needed

    def process(self):
        # Load raw data
        data_path = os.path.join(self.raw_dir, self.filename)
        df = pd.read_csv(data_path)


        graph_list = []
        for i, smiles in tqdm(enumerate(df[self.smiles_column])):

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue  # Skip invalid SMILES strings

            g = from_smiles(smiles)
            g.x = g.x.float()
            g.y = torch.tensor(np.array(df.loc[i].drop(self.smiles_column).values, dtype=np.float32), dtype=torch.float).view(1, -1)

            g.ECFP = torch.tensor(self.ECFP[i], dtype=torch.float).view(1, -1)
            g.Topological = torch.tensor(self.Topological[i], dtype=torch.float).view(1, -1)
            g.MACCS = torch.tensor(self.MACCS[i], dtype=torch.float).view(1, -1)
            g.EState = torch.tensor(self.EState[i], dtype=torch.float).view(1, -1)
            g.Rdkit2D = torch.tensor(self.Rdkit2D[i], dtype=torch.float).view(1, -1)
            g.Phar2D = torch.tensor(self.Phar2D[i], dtype=torch.float).view(1, -1)

            # g.Phar3D = torch.tensor(self.Phar3D[i], dtype=torch.float).view(1, -1)
            # g.Rdkit3D = torch.tensor(self.Rdkit3D[i], dtype=torch.float).view(1, -1)

            graph_list.append(g)

        data_list = graph_list

        # Apply pre-filter and pre-transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data
        self.save(data_list, self.processed_paths[0])

# dataset_64 = DTsetBasic(root='data/basic-64', filename='tox21_cleaned.csv', smiles_column='smiles',
#     ECFP=ecfp_reduced, Topological=topological_reduced, MACCS=maccs_reduced,
#     EState=estate_reduced, Rdkit2D=rdkit2D_reduced, Phar2D=phar2D_reduced)


class DTsetBasicExtended(Dataset):
    def __init__(self, root, filename, smiles_column, label_column, 
    ECFP, Topological, MACCS, EState, MordredD, Phar2D, Phar3D, Rdkit3D):
        self.filename = filename
        self.smiles_column = smiles_column
        self.label_column = label_column

        self.ECFP = ECFP
        self.Topological = Topological
        self.MACCS = MACCS
        self.EState = EState
        self.Rdkit2D = Rdkit2D
        self.Phar2D = Phar2D
        self.Phar3D = Phar3D
        self.Rdkit3D = Rdkit3D

        super(DTsetBasicExtended, self).__init__(root)

    @property
    def raw_file_names(self):
        return [self.filename]

    @property
    def processed_file_names(self):
        data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in data.index]

    def download(self):
        pass

    def process(self):

        data_path = os.path.join(self.raw_dir, self.filename)
        df = pd.read_csv(data_path)

        for idx, smiles in tqdm(enumerate(df[self.smiles_column])):
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                continue  # Skip invalid SMILES strings
            
            node_feats = self._get_node_features(mol)
            edge_feats = self._get_edge_features(mol)
            edge_index = self._get_adjacency_info(mol)
            label = torch.tensor(df[self.label_column][idx], dtype=torch.float).view(1, -1)

            ECFP = torch.tensor(self.ECFP[idx], dtype=torch.float).view(1, -1)
            Topological = torch.tensor(self.Topological[idx], dtype=torch.float).view(1, -1)
            MACCS = torch.tensor(self.MACCS[idx], dtype=torch.float).view(1, -1)
            EState = torch.tensor(self.EState[idx], dtype=torch.float).view(1, -1)
            Rdkit2D = torch.tensor(self.Rdkit2D[idx], dtype=torch.float).view(1, -1)
            Phar2D = torch.tensor(self.Phar2D[idx], dtype=torch.float).view(1, -1)
            Phar3D = torch.tensor(self.Phar3D[idx], dtype=torch.float).view(1, -1)
            Rdkit3D = torch.tensor(self.Rdkit3D[idx], dtype=torch.float).view(1, -1)
            
            data = Data(
                x = node_feats,
                edge_index = edge_index,
                edge_attr = edge_feats,
                y = label,
                smiles=smiles,
                ECFP = ECFP, 
                Topological = Topological, 
                MACCS = MACCS,
                EState = EState,
                Rdkit2D = Rdkit2D,
                Phar2D = Phar2D,
                Phar3D = Phar3D,
                Rdkit3D = Rdkit3D
            )

           # Save processed data
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))


    def _get_node_features(self, mol):
        """Returns a matrix of shape [Number of Nodes, Node Feature size]."""
        all_node_feats = []
        for atom in mol.GetAtoms():
            node_feats = [
                atom.GetAtomicNum(),  # Atomic number
                atom.GetDegree(),  # Degree
                atom.GetFormalCharge(),  # Formal charge
                int(atom.GetHybridization()),  # Hybridization
                atom.GetIsAromatic(),  # Aromaticity
                atom.GetTotalNumHs(),  # Total number of Hs
                atom.GetNumRadicalElectrons(),  # Radical Electrons
                atom.IsInRing(),  # In Ring
                int(atom.GetChiralTag()),  # Chirality
                atom.GetMass(),  # Atomic mass
                atom.GetExplicitValence(),  # Explicit valence
                atom.GetImplicitValence(),  # Implicit valence
                atom.GetTotalValence(),  # Total valence
                atom.GetIsotope()  # Isotope
            ]
            all_node_feats.append(node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        """Returns a matrix of shape [Number of edges, Edge Feature size]."""
        all_edge_feats = []
        for bond in mol.GetBonds():
            edge_feats = [
                bond.GetBondTypeAsDouble(),  # Bond type
                bond.IsInRing(),  # In Ring
                bond.GetIsAromatic(),  # Aromaticity
                int(bond.GetBondDir()),  # Bond direction
                int(bond.GetStereo()),  # Stereochemistry
                bond.GetBondLength() if hasattr(bond, 'GetBondLength') else 0  # Bond length
            ]
            # Append edge features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        """Returns adjacency information for the molecule."""
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
        edge_indices = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        """Converts label to tensor."""
        return torch.tensor([label], dtype=torch.float)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data


class DTsetDeepChemFeaturizer(Dataset):
    def __init__(self, root, filename, smiles_column, label_column, featurizer, 
    ECFP, Topological, MACCS, EState, Rdkit2D, Phar2D, Phar3D, Rdkit3D, test=False):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.filename = filename
        self.smiles_column = smiles_column
        self.label_column = label_column
        self.featurizer = featurizer
        self.test = test

        ### prepare FP, DES :
        self.ECFP = ECFP
        self.Topological = Topological
        self.MACCS = MACCS
        self.EState = EState
        self.Rdkit2D = Rdkit2D
        self.Phar2D = Phar2D
        self.Phar3D = Phar3D
        self.Rdkit3D = Rdkit3D

        super(DTsetDeepChemFeaturizer, self).__init__(root)

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered."""
        return [self.filename]

    @property
    def processed_file_names(self):
        """If these files are found in raw_dir, processing is skipped."""
        data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in data.index]

    def download(self):
        pass  # Implement download logic if needed

    def process(self):
        # Load raw data
        data_path = os.path.join(self.raw_dir, self.filename)
        df = pd.read_csv(data_path)

        # Process each SMILES string
        for idx, smiles in tqdm(enumerate(df[self.smiles_column])):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue  # Skip invalid SMILES strings
            
            ECFP = torch.tensor(self.ECFP[idx], dtype=torch.float).view(1, -1)
            Topological = torch.tensor(self.Topological[idx], dtype=torch.float).view(1, -1)
            MACCS = torch.tensor(self.MACCS[idx], dtype=torch.float).view(1, -1)
            EState = torch.tensor(self.EState[idx], dtype=torch.float).view(1, -1)
            Rdkit2D = torch.tensor(self.Rdkit2D[idx], dtype=torch.float).view(1, -1)
            Phar2D = torch.tensor(self.Phar2D[idx], dtype=torch.float).view(1, -1)
            Phar3D = torch.tensor(self.Phar3D[idx], dtype=torch.float).view(1, -1)
            Rdkit3D = torch.tensor(self.Rdkit3D[idx], dtype=torch.float).view(1, -1)

            # Featurize molecule
            f = self.featurizer._featurize(mol)
            # To pyg
            data = f.to_pyg_graph()
            data.y = torch.tensor(df[self.label_column][idx], dtype=torch.float).view(1, -1)
            data.smiles = smiles

            data.ECFP = ECFP, 
            data.Topological = Topological, 
            data.MACCS = MACCS,
            data.EState = EState,
            data.Rdkit2D = Rdkit2D,
            data.Phar2D = Phar2D,
            data.Phar3D = Phar3D,
            data.Rdkit3D = Rdkit3D

            # Save processed data
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data


# Featurizer
# featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)



class DTsetMolGraphConvFeaturizer(Dataset):
    def __init__(self, root, filename, smiles_column, label_column, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.filename = filename
        self.smiles_column = smiles_column
        self.label_column = label_column
        self.test = test
        super(DTsetMolGraphConvFeaturizer, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered."""
        return [self.filename]

    @property
    def processed_file_names(self):
        """If these files are found in raw_dir, processing is skipped."""
        data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in data.index]

    def download(self):
        pass  # Implement download logic if needed

    def process(self):
        # Load raw data
        data_path = os.path.join(self.raw_dir, self.filename)
        df = pd.read_csv(data_path)

        # Featurizer
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

        # Process each SMILES string
        for idx, smiles in tqdm(enumerate(df[self.smiles_column])):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue  # Skip invalid SMILES strings

            # Featurize molecule
            f = featurizer._featurize(mol)
            data = f.to_pyg_graph()
            data.y = torch.tensor(df[self.label_column][idx], dtype=torch.float).view(1, -1)
            data.smiles = smiles

            # Save processed data
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
