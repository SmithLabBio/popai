
from tensorflow.keras.utils import to_categorical
import pickle
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import glob
import re
import os
from typing import List


class PopaiDataset(Dataset):
    def __init__(self, paths:List[str]):
        self.paths = paths
        self.path_ixs = [] # Lookup for path index for __getitem__ index
        self.sim_ixs = [] # Lookup for simulation index for __getitem__ index
        self.labels = []
        for i in range(0, len(paths)):
            modno = paths[i].split('_')[-1].split('.')[0]
            with open(paths[i], "rb") as fh:
                p = pickle.load(fh)
                for j in range(0, len(p)):
                    self.labels.append(int(modno))
                    self.path_ixs.append(i)
                    self.sim_ixs.append(j)
        self.n_classes = len(set(self.labels))
        self.encoded_labels = to_categorical(self.labels)

    def __len__(self):
        return len(self.path_ixs)

    def __getitem__(self, ix):
        path = self.paths[self.path_ixs[ix]]
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return data[self.sim_ixs[ix]], self.encoded_labels[ix] 

class PopaiDatasetLowMem(Dataset):
    def __init__(self, paths:List[str]):
        self.paths = paths
        self.path_ixs = [] # Lookup for path index for __getitem__ index
        self.sim_ixs = [] # Lookup for simulation index for __getitem__ index
        self.labels = []
        for i in range(0, len(paths)):
            modno = paths[i].split('_')[-1].split('.')[0]
            with open(paths[i], "rb") as fh:
                p = pickle.load(fh)
                for j in range(0, len(p)):
                    self.labels.append(int(modno))
                    self.path_ixs.append(i)
                    self.sim_ixs.append(j)
        self.n_classes = len(set(self.labels))
        self.encoded_labels = to_categorical(self.labels)

    def __len__(self):
        return len(self.path_ixs)

    def __getitem__(self, ix):
        path = self.paths[self.path_ixs[ix]]
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return data[self.sim_ixs[ix]], self.encoded_labels[ix] 


def human_sort_key(s):
    """
    Key function for human sorting.
    Splits the string into parts and converts numeric parts to integers.
    """
    return [int(part) if part.isdigit() else part for part in re.split('([0-9]+)', s)]

class PopaiTrainingData:
    """
    Container for training datasets and data loaders. 
    """
    def __init__(self, dir:str, pattern:str, seed:int, low_mem:bool=False):
        pattern_path = os.path.join(dir, pattern)
        paths = glob.glob(pattern_path)
        sorted_paths = sorted(paths, key=human_sort_key)
        if len(sorted_paths) == 0:
            raise FileNotFoundError(f"No files found matching pattern: {pattern_path}") 
        if low_mem:
            self.dataset = PopaiDatasetLowMem(sorted_paths)  
        else:
            self.dataset = PopaiDataset(sorted_paths)  
        rng = np.random.default_rng(seed)
        train_test_seed = rng.integers(2**32, size=1)[0]
        train_ixs, test_ixs = train_test_split(np.arange(len(self.dataset)), test_size=0.2, 
                random_state=train_test_seed, stratify=self.dataset.labels)
        self.train_dataset = Subset(self.dataset, train_ixs)
        self.test_dataset =  Subset(self.dataset, test_ixs)
        self.train_loader = DataLoader(self.train_dataset, batch_size=10, shuffle=True)
        self.test_loader =  DataLoader(self.test_dataset,  batch_size=10, shuffle=False)