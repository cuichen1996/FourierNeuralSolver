# %%
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, N, G, b, f):
        self.N = N
        self.G = G
        self.b = b
        self.f = f

    def __len__(self):
        return self.b.shape[0]

    def __getitem__(self, idx):
        b = self.b[idx].view(1, self.N+1, self.N+1)
        f = self.f[idx].view(1, self.N+1, self.N+1)
        G = self.G.view(1, self.N+1, self.N+1)
        return G, b, f

def load_data(data_dir, batch_size, N):

    G = torch.from_numpy(np.load(data_dir+"/G.npy"))
    f = torch.from_numpy(np.load(data_dir+"/f.npy"))
    b = torch.from_numpy(np.load(data_dir+"/b.npy"))

    G0 = torch.from_numpy(np.load(data_dir+"/G0.npy"))
    f0 = torch.from_numpy(np.load(data_dir+"/f0.npy"))
    b0 = torch.from_numpy(np.load(data_dir+"/b0.npy"))
    train_dataset = MyDataset(N, G, b, f)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataset = MyDataset(N, G0, b0, f0)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=8)

    return train_loader, test_loader