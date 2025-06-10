import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from scipy.fftpack import idctn


class CreateDataset(data.Dataset):
    def __init__(self, p, data1, data2, label):
        self.p = p
        self.data1 = data1
        self.data2 = data2
        self.label = label

    def __getitem__(self, index):
        f = self.data1[index]
        a = self.data2[index]
        u = self.label[index]
        return f, a, u

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


def CreateTrainLoader(config, N=None, pde_num=None):
    ctype = config["ctype"]
    if pde_num is None:
        pde_num = config["pde_num"]
    num = config["every_num"]
    if N is None:
        N = config["N"]
    h = 1/(N-1)
    
    alpha = 2
    tau = 3
    k1 = range(0,N-1,1)
    k2 = range(0,N-1,1)
    K1, K2 = np.meshgrid(k1,k2)
    kernel1 = torch.tensor([[[[-1/6, 2/3], [-1/3, -1/6]]]], dtype=torch.float64)
    kernel2 = torch.tensor([[[[2/3, -1/6], [-1/6, -1/3]]]], dtype=torch.float64)
    kernel3 = torch.tensor([[[[-1/6, -1/3], [2/3, -1/6]]]], dtype=torch.float64)
    kernel4 = torch.tensor([[[[-1/3, -1/6], [-1/6, 2/3]]]], dtype=torch.float64)

    kernel = torch.vstack((kernel1, kernel2, kernel3, kernel4)).permute(1, 0, 2, 3)

    datau = []
    dataa = []
    dataf = []
    for i in range(pde_num):
        xi = np.random.normal(0, 1, (N-1, N-1))
        coef = tau**(alpha-1)*(np.pi**2*(K1*K1+K2*K2) + tau**2)**(-alpha/2)
        L = N*coef*xi
        L[0,0] = 0
        norm_a = idctn(L,norm='ortho')
        if ctype == "log":
            lognorm_a = np.exp(norm_a)
            a = torch.tensor(lognorm_a, dtype=torch.float64)
        elif ctype == "jump":
            thresh_a = np.zeros((N-1,N-1))
            thresh_a[norm_a >= 0] = torch.pow(10, torch.rand(1)*5)
            thresh_a[norm_a < 0] = 1
            a = torch.tensor(thresh_a, dtype=torch.float64)
        a = a.unsqueeze(0).unsqueeze(0)
        # a = F.interpolate(a, N-1, mode="nearest")

        u = torch.randn(num, 1, N, N, dtype=torch.float64)
        u1 = u[:,:,1:N,:N-1] * a
        u2 = u[:,:,1:N,1:] * a
        u3 = u[:,:,:N-1,1:] * a
        u4 = u[:,:,:N-1,:N-1] * a
        U = torch.cat((u1, u2, u3, u4), dim=1)
        a = a.repeat(num, 1, 1, 1)  #* a
        f = F.conv2d(U, kernel, padding=0)
        f = F.pad(f, pad=(1,1,1,1), mode='constant', value=0)  #* f

        f = torch.ones_like(u, dtype=torch.float64)*h**2
        f = set_bd(f)
        datau.append(u)
        dataa.append(a)
        dataf.append(f)

    f = torch.cat(dataf, dim=0)
    a = torch.cat(dataa, dim=0)
    u = torch.cat(datau, dim=0)
    total_num = f.shape[0]
    dataset = CreateDataset(total_num, f, a, u)
    dataLoader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    return dataLoader


def CreateTestLoader(config, N=None):
    ctype = config["ctype"]
    num = 1
    pde_num = config["test_num"]
    if N is None:
        N = config["N"]
    h = 1/(N-1)
    alpha = 2
    tau = 3
    k1 = range(0,N-1,1)
    k2 = range(0,N-1,1)
    K1, K2 = np.meshgrid(k1,k2)
    kernel1 = torch.tensor([[[[-1/6, 2/3], [-1/3, -1/6]]]], dtype=torch.float64)
    kernel2 = torch.tensor([[[[2/3, -1/6], [-1/6, -1/3]]]], dtype=torch.float64)
    kernel3 = torch.tensor([[[[-1/6, -1/3], [2/3, -1/6]]]], dtype=torch.float64)
    kernel4 = torch.tensor([[[[-1/3, -1/6], [-1/6, 2/3]]]], dtype=torch.float64)

    kernel = torch.vstack((kernel1, kernel2, kernel3, kernel4)).permute(1, 0, 2, 3)

    datau = []
    dataa = []
    dataf = []
    for i in range(pde_num):
        u = torch.randn(num, 1, N, N).double()  #* u
        xi = np.random.normal(0, 1, (N-1, N-1))
        coef = tau**(alpha-1)*(np.pi**2*(K1*K1+K2*K2) + tau**2)**(-alpha/2)
        L = N*coef*xi
        L[0,0] = 0
        norm_a = idctn(L,norm='ortho')
        if ctype == "log":
            lognorm_a = np.exp(norm_a)
            a = torch.tensor(lognorm_a, dtype=torch.float64)
        elif ctype == "jump":
            thresh_a = np.zeros((N-1,N-1))
            thresh_a[norm_a >= 0] = 10
            thresh_a[norm_a < 0] = 1
            a = torch.tensor(thresh_a, dtype=torch.float64)
        a = a.unsqueeze(0).unsqueeze(0)

        f = torch.ones_like(u, dtype=torch.float64)*h**2
        f = set_bd(f)

        datau.append(u)
        dataa.append(a)
        dataf.append(f)

    f = torch.cat(dataf, dim=0)
    a = torch.cat(dataa, dim=0)
    u = torch.cat(datau, dim=0)
    total_num = f.shape[0]
    dataset = CreateDataset(total_num, f, a, u)
    dataLoader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataLoader