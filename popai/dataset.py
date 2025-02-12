
from torch.utils.data import Dataset
from tensorflow.keras.utils import to_categorical
import pickle

class PopaiDataset(Dataset):
    def __init__(self, paths):
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

# import glob
# from sklearn.model_selection import train_test_split
# import numpy as np

# # paths = [
# #     "../popai-tutorial/simulated/simulated_mSFS_0.pickle",
# #     "../popai-tutorial/simulated/simulated_mSFS_1.pickle"
# #     # "../popai-tutorial/simulated/simulated_2dSFS_0.pickle",
# #     # "../popai-tutorial/simulated/simulated_2dSFS_1.pickle"
# #     # "../popai-tutorial/simulated/simulated_arrays_0.pickle",
# #     # "../popai-tutorial/simulated/simulated_arrays_1.pickle"
# # ]

# paths = glob.glob("../../popai-tutorial/simulated/simulated_mSFS_*.pickle")
# d = PopaiDataset(paths)
# train, test = train_test_split(np.arange(len(d)), test_size=0.2, stratify=d.labels)



# train_loader = DataLoader(train, batch_size=1, shuffle=True)
# test_loader = DataLoader(test, batch_size=1, shuffle=True)
