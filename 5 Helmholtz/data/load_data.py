import numpy as np
import torch
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from data.data import KappaDataGenerator

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x 
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx] 
        y = self.y[idx]
        return x, y
    

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, p, data1, data2, data3):
        self.p = p
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3


    def __getitem__(self, index):
        f = self.data1[index, :]
        kappa = self.data2[index, :]
        u = self.data3[index, :]
        return f, kappa, u

    def __len__(self):
        return self.p

def CreateTrainLoader(config):
    # N = 128
    kappa = torch.from_numpy(np.load("/random/128/data_kappa128.npy"))
    f = torch.randn_like(kappa, dtype=torch.cfloat)
    # f[:,:, 64, 64] = 128**2
    u = torch.from_numpy(np.load("/random/128/data_u128.npy"))
    # f = torch.from_numpy(np.load("/random/128/data_f128.npy"))
    data_num = len(kappa)
    print("Number of training data: ", data_num)

    trainset = CreateDataset(data_num, f, kappa, u)
    train_loader128 = DataLoader(
        trainset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    # N = 256
    # kappa = torch.from_numpy(np.load("/random/256/data_kappa256.npy"))
    # u = torch.from_numpy(np.load("/random/256/data_u256.npy"))
    # f = torch.from_numpy(np.load("/random/256/data_f256.npy"))
    # data_num = len(kappa)
    # print("Number of training data: ", data_num)

    # trainset = CreateDataset(data_num, f, kappa, u)
    # train_loader256 = DataLoader(
    #     trainset,
    #     batch_size=config["batch_size"]//2,
    #     shuffle=True
    # )

    # # N = 512
    # kappa = torch.from_numpy(np.load("/random/512/data_kappa512.npy"))
    # u = torch.from_numpy(np.load("/random/512/data_u512.npy"))
    # f = torch.from_numpy(np.load("/random/512/data_f512.npy"))
    # data_num = len(kappa)
    # print("Number of training data: ", data_num)

    # trainset = CreateDataset(data_num, f, kappa, u)
    # train_loader512 = DataLoader(
    #     trainset,
    #     batch_size=config["batch_size"]//4,
    #     shuffle=True
    # )
    return train_loader128


def CreateTestLoader(N):
    dataset = KappaDataGenerator(N+1, N+1)
    dataset.load_data(data="cifar10")
    kappa = dataset.generate_kappa().to(torch.float)
    # kappa = torch.ones_like(kappa)
    b = torch.zeros_like(kappa, dtype=torch.cfloat)
    b[:,:, N//2, N//2] = 1.0 * N**2
    u = torch.zeros_like(b, dtype=torch.cfloat)
    print(kappa.shape, b.shape, u.shape)
    trainset = CreateDataset(1, b, kappa, u)
    train_loader = DataLoader(
        trainset,
        batch_size=1,
        shuffle=False
    )
    return train_loader

