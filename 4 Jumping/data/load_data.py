import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np

class CreateDataset(data.Dataset):
    def __init__(self, p, data1, data2):
        self.p = p
        self.data1 = data1
        self.data2 = data2

    def __getitem__(self, index):
        f = self.data1[index]
        a = self.data2[index]
        return f, a

    def __len__(self):
        return self.p


def set_bd(x):
    n = x.size(-1)
    mask = torch.ones(1, 1, n, n)
    mask[:, :, 0, :]  = 0   
    mask[:, :, -1, :] = 0  
    mask[:, :, :, 0]  = 0   
    mask[:, :, :, -1] = 0  
    x *= mask 
    return x

def CreateTrainLoader(config, N):
    pde_num = config["pde_num"]
    h = 1/N
    dataa = []
    dataf = []
    number = 4
    for i in range(pde_num):
        a = torch.randn(number, number)
        c = torch.pow(10, -3*(torch.rand(1)+1))
        for i in range(number):
            for j in range(number):
                if a[i,j] > 0:
                    a[i,j] = 1
                else:
                    a[i,j] = c
        a = F.interpolate(a.unsqueeze(0).unsqueeze(0), N-1)

        f = torch.randn([1, 1, N+1, N+1], dtype=torch.float64)*h**2
        f = set_bd(f)
        dataa.append(a)
        dataf.append(f)

    f = torch.cat(dataf, dim=0)
    a = torch.cat(dataa, dim=0)
    total_num = f.shape[0]
    dataset = CreateDataset(total_num, f, a)
    dataLoader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    return dataLoader


def CreateTestLoader(config, N=None):
    pde_num = config["test_num"]
    if N is None:
        N = config["N"]
    h = 1/N

    dataa = []
    dataf = []
    number = 4
    rect1_top, rect1_bottom = 25, 35
    rect1_left, rect1_right = 30, 70
    rect2_top, rect2_bottom = 65, 75
    rect2_left, rect2_right = 30, 70
    for i in range(pde_num):
        # * choice 1
        a = torch.randn(number, number)
        c = torch.pow(10, -3*(torch.rand(1)+1))
        print(c)
        for i in range(number):
            for j in range(number):
                if a[i,j] > 0:
                    a[i,j] = 1
                else:
                    a[i,j] = c
        # * choice 2
        # a = np.flipud(np.load("data/shandong.npy"))
        # a = torch.from_numpy(a.copy())
        # * choice 3
        # a = torch.ones(100, 100)
        # c = -np.random.uniform(0,5)
        # a[rect1_top:rect1_bottom, rect1_left:rect1_right] = 10**c
        # a[rect2_top:rect2_bottom, rect2_left:rect2_right] = 10**c

        #  * interpolate
        a = F.interpolate(a.unsqueeze(0).unsqueeze(0), N-1)

        f = torch.ones([1, 1, N+1, N+1], dtype=torch.float64)*h**2
        f = set_bd(f)
        dataa.append(a)
        dataf.append(f)

    f = torch.cat(dataf, dim=0)
    a = torch.cat(dataa, dim=0)
    dataset = CreateDataset(pde_num, f, a)
    dataLoader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataLoader
