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



def generate_graph_list(df, smiles_column, target_column):
    graph_list = []

    for i, smile in tqdm(enumerate(df[smiles_column])):
        g = from_smiles(smile)
        g.x = g.x.float()
        y = torch.tensor(df[target_column][i], dtype=torch.float).view(1, -1)
        g.y = y
        graph_list.append(g)

    return graph_list



############################# General Loader : #############################

def load_and_process_data(dataset, splitter="random", test_size=0.1, batch_size=32):
    """
    Loads a dataset, splits it into train, validation, and test sets, and creates PyTorch Geometric data loaders.
    """
    if splitter == "random":
        
        data_size = len(dataset)
        train_idx, test_idx = train_test_split(list(range(data_size)), test_size=0.1)
        train_idx, valid_idx = train_test_split(train_idx, test_size = test_size)  # Split train further into train and valid

        # Create data loaders for train, validation, and test sets
        train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset[valid_idx], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[test_idx], batch_size=batch_size, shuffle=False)

    else:
        raise ValueError(f"Invalid splitter type: {splitter}. Valid options are 'random' or 'scaffold'.")

    return train_loader, val_loader, test_loader



def generate_scaffold(smiles, include_chirality=False):
    """Generate the Bemis-Murcko scaffold for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_split_indices(smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=None, include_chirality=False):
    """
    Perform scaffold splitting on a list of SMILES strings and return the indices for train, validation, and test sets.

    Args:
        smiles_list (list): List of SMILES strings.
        frac_train (float): Fraction of the dataset to use for training.
        frac_valid (float): Fraction of the dataset to use for validation.
        frac_test (float): Fraction of the dataset to use for testing.
        seed (int): Random seed for shuffling the scaffolds.
        include_chirality (bool): Whether to include chirality in scaffold generation.

    Returns:
        dict: Dictionary with train, valid, and test indices as torch tensors.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0, err_msg="The fractions must sum to 1.")
    
    # Set random seed for reproducibility
    rng = np.random.RandomState(seed)
    
    # Group SMILES by their scaffold
    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality)
        scaffolds[scaffold].append(ind)
    
    # Get scaffold keys and shuffle them
    scaffold_keys = list(scaffolds.keys())
    rng.shuffle(scaffold_keys)
    
    # Compute the number of samples for each set
    n_total = len(smiles_list)
    n_total_valid = int(np.floor(frac_valid * n_total))
    n_total_test = int(np.floor(frac_test * n_total))
    
    train_index = []
    valid_index = []
    test_index = []
    
    # Distribute the scaffold sets into train, valid, and test sets
    for scaffold_key in scaffold_keys:
        scaffold_set = scaffolds[scaffold_key]
        if len(valid_index) + len(scaffold_set) <= n_total_valid:
            valid_index.extend(scaffold_set)
        elif len(test_index) + len(scaffold_set) <= n_total_test:
            test_index.extend(scaffold_set)
        else:
            train_index.extend(scaffold_set)
    
    # Return indices as torch tensors in a dictionary
    return {
        'train': torch.tensor(train_index, dtype=torch.long),
        'valid': torch.tensor(valid_index, dtype=torch.long),
        'test': torch.tensor(test_index, dtype=torch.long)
    }
    
    
class FingerprintsDescriptorsCalculator:
    def __init__(self, smiles_column):
        self.smiles_column = smiles_column
        
        self.valid_molecules = []
        self.valid_smiles = []
        self.invalid_indices = []

        for index, smiles in tqdm(enumerate(self.smiles_column)) :
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print('******* Invalid Mol !!!!!!!')
                self.invalid_indices.append(index)
            else :
                self.valid_smiles.append(smiles)


        self.calc_ecfp = FPCalculator("ecfp")
        self.calc_topological = FPCalculator("topological")
        self.calc_maccs = FPCalculator("maccs")
        self.calc_estate = FPCalculator("estate")
        self.calc_rdkit2D = RDKitDescriptors2D(replace_nan=True)
        self.calc_phar2D = Pharmacophore2D()
      

        self.featurizer_ecfp = MoleculeTransformer(self.calc_ecfp, dtype=np.float64)
        self.featurizer_topological = MoleculeTransformer(self.calc_topological, dtype=np.float64)
        self.featurizer_maccs = MoleculeTransformer(self.calc_maccs, dtype=np.float64)
        self.featurizer_estate = MoleculeTransformer(self.calc_estate, dtype=np.float64)
        self.featurizer_rdkit2D = MoleculeTransformer(self.calc_rdkit2D, dtype=np.float64)
        self.featurizer_phar2D = MoleculeTransformer(self.calc_phar2D, dtype=np.float64)
        

    def calculate_ecfp(self):
        with dm.without_rdkit_log():
            return self.featurizer_ecfp(self.valid_smiles)

    def calculate_topological(self):
        with dm.without_rdkit_log():
            return self.featurizer_topological(self.valid_smiles)

    def calculate_maccs(self):
        with dm.without_rdkit_log():
            return self.featurizer_maccs(self.valid_smiles)

    def calculate_estate(self):
        with dm.without_rdkit_log():
            return self.featurizer_estate(self.valid_smiles)

    def calculate_rdkit2D(self):
        with dm.without_rdkit_log():
            return self.featurizer_rdkit2D(self.valid_smiles)

    def calculate_phar2D(self):
        with dm.without_rdkit_log():
            return self.featurizer_phar2D(self.valid_smiles)
    
    def get_invalid_indices(self):
        return self.invalid_indices

    def get_valid_smiles(self):
        return self.valid_smiles


# Usage Example :
# df = pd.read_csv('/content/bace.csv')
# smiles_column = df['mol'].values

# calculator = FingerprintsDescriptorsCalculator(smiles_column)

# ecfp = calculator.calculate_ecfp()
# topological = calculator.calculate_topological()
# maccs = calculator.calculate_maccs()
# estate = calculator.calculate_estate()
# rdkit2D = calculator.calculate_rdkit2D()
# phar2D = calculator.calculate_phar2D()

# phar3D = calculator.calculate_phar3D()
# rdkit3D = calculator.calculate_rdkit3D()
# invalid_indices = calculator.get_invalid_indices()


class FingerprintsDescriptorsCalculator2:
    def __init__(self, smiles_column):
        self.smiles_column = smiles_column
        
        self.valid_smiles = []
        self.invalid_indices = []

        for index, smiles in tqdm(enumerate(self.smiles_column)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f'******* Invalid Mol at index {index} !!!!!!')
                self.invalid_indices.append(index)
            else:
                self.valid_smiles.append(smiles)

        self.calc_ecfp = FPCalculator("ecfp")
        self.calc_topological = FPCalculator("topological")
        self.calc_maccs = FPCalculator("maccs")
        self.calc_estate = FPCalculator("estate")
        self.calc_rdkit2D = RDKitDescriptors2D(replace_nan=True)
        self.calc_phar2D = Pharmacophore2D(replace_nan=True)

        self.featurizer_ecfp = MoleculeTransformer(self.calc_ecfp, dtype=np.float64)
        self.featurizer_topological = MoleculeTransformer(self.calc_topological, dtype=np.float64)
        self.featurizer_maccs = MoleculeTransformer(self.calc_maccs, dtype=np.float64)
        self.featurizer_estate = MoleculeTransformer(self.calc_estate, dtype=np.float64)
        self.featurizer_rdkit2D = MoleculeTransformer(self.calc_rdkit2D, dtype=np.float64)
        # self.featurizer_phar2D = MoleculeTransformer(self.calc_phar2D, dtype=np.float64)

    def calculate_phar2D(self, timeout=20):
        def timeout_handler(signum, frame):
            raise TimeoutError("Phar2D calculation timed out")

        signal.signal(signal.SIGALRM, timeout_handler)

        results = []
        remaining_smiles = []
        for index, smiles in tqdm(enumerate(self.valid_smiles)):
            signal.alarm(timeout)
            try:
                with dm.without_rdkit_log():
                    result = self.calc_phar2D(smiles)
                results.append(result)
                remaining_smiles.append(smiles)
            except TimeoutError:
                print(f"Phar2D calculation timed out for index {index}, smiles: {smiles}")
                self.invalid_indices.append(index)
            finally:
                signal.alarm(0)

        self.valid_smiles = remaining_smiles
        return np.array(results, dtype=np.float64)

    def calculate_ecfp(self):
        with dm.without_rdkit_log():
            return self.featurizer_ecfp(self.valid_smiles)

    def calculate_topological(self):
        with dm.without_rdkit_log():
            return self.featurizer_topological(self.valid_smiles)

    def calculate_maccs(self):
        with dm.without_rdkit_log():
            return self.featurizer_maccs(self.valid_smiles)

    def calculate_estate(self):
        with dm.without_rdkit_log():
            return self.featurizer_estate(self.valid_smiles)

    def calculate_rdkit2D(self):
        with dm.without_rdkit_log():
            return self.featurizer_rdkit2D(self.valid_smiles)

    def get_invalid_indices(self):
        return self.invalid_indices

    def get_valid_smiles(self):
        return self.valid_smiles


# calculator = FingerprintsDescriptorsCalculator2(smiles_column)

# phar2D = calculator.calculate_phar2D()
# ecfp = calculator.calculate_ecfp()
# topological = calculator.calculate_topological()
# maccs = calculator.calculate_maccs()
# estate = calculator.calculate_estate()
# rdkit2D = calculator.calculate_rdkit2D()
# invalid_indices = calculator.get_invalid_indices()



class PCAReducer:
    def __init__(self, n_components=64):
        self.n_components = n_components
        self.pca_ecfp = PCA(n_components=self.n_components)
        self.pca_topological = PCA(n_components=self.n_components)
        self.pca_maccs = PCA(n_components=self.n_components)
        self.pca_estate = PCA(n_components=self.n_components)
        self.pca_rdkit2D = PCA(n_components=self.n_components)
        self.pca_phar2D = PCA(n_components=self.n_components)
        # self.pca_phar3D = PCA(n_components=self.n_components)
        # self.pca_rdkit3D = PCA(n_components=self.n_components)


    def reduce_ecfp(self, ecfp_data):
        return self.pca_ecfp.fit_transform(ecfp_data)

    def reduce_topological(self, topological_data):
        return self.pca_topological.fit_transform(topological_data)

    def reduce_maccs(self, maccs_data):
        return self.pca_maccs.fit_transform(maccs_data)

    def reduce_estate(self, estate_data):
        return self.pca_estate.fit_transform(estate_data)

    def reduce_rdkit2D(self, rdkit2D_data):
        return self.pca_rdkit2D.fit_transform(rdkit2D_data)

    def reduce_phar2D(self, phar2D_data):
        return self.pca_phar2D.fit_transform(phar2D_data)

    def reduce_phar3D(self, phar3D_data):
        return self.pca_phar3D.fit_transform(phar3D_data)

    def reduce_rdkit3D(self, rdkit3D_data):
        return self.pca_rdkit3D.fit_transform(rdkit3D_data)

# Usage Example :
# N_COMPONENTS = 64
# reducer = PCAReducer(n_components=N_COMPONENTS)

# ecfp_reduced = reducer.reduce_ecfp(ecfp)
# topological_reduced = reducer.reduce_topological(topological)
# maccs_reduced = reducer.reduce_maccs(maccs)
# estate_reduced = reducer.reduce_estate(estate)
# rdkit2D_reduced = reducer.reduce_rdkit2D(rdkit2D)
# phar2D_reduced = reducer.reduce_phar2D(phar2D)

# phar3D_reduced = reducer.reduce_phar3D(phar3D)
# rdkit3D_reduced = reducer.reduce_rdkit3D(rdkit3D)


class DTsetBasic(InMemoryDataset):
    def __init__(self, root, filename, smiles_column, label_column,
                 ECFP, Topological, MACCS, EState, Rdkit2D, Phar2D):
        self.filename = filename
        self.smiles_column = smiles_column
        self.label_column = label_column

        self.ECFP = ECFP
        self.Topological = Topological
        self.MACCS = MACCS
        self.EState = EState
        self.Rdkit2D = Rdkit2D
        self.Phar2D = Phar2D

        # self.Phar3D = Phar3D
        # self.Rdkit3D = Rdkit3D

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
            g.y = torch.tensor(df[self.label_column][i], dtype=torch.float).view(1, -1)

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

# dataset_64 = DTsetBasic(root='basic-64', filename='bace.csv', smiles_column='mol', label_column='Class',
#     ECFP=ecfp_reduced, Topological=topological_reduced, MACCS=maccs_reduced,
#     EState=estate_reduced, Rdkit2D=rdkit2D_reduced, Phar2D=phar2D_reduced)


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
