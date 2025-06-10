import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from torch.utils.data.distributed import DistributedSampler


Kernelxx = torch.tensor([[[[-1./6., 1./3., -1./6.], [-2./3., 4./3., -2./3.], [-1./6., 1./3., -1./6.]]]])  
Kernelxy = torch.tensor([[[[-1./4., 0., 1./4.], [0., 0., 0.], [1./4., 0., -1./4.]]]])   
Kernelyx = torch.tensor([[[[-1./4., 0., 1./4.], [0., 0., 0.], [1./4., 0., -1./4.]]]])    
Kernelyy = torch.tensor([[[[-1./6., -2./3., -1./6.], [1./3., 4./3., 1./3.], [-1./6., -2./3., -1./6.]]]])    

def CreateKernelA(epsilons, data_num=100, repeat=True):
    m = epsilons.shape[0]
    if repeat:
        KernelA = torch.zeros([m*data_num, 1, 3, 3]).to(torch.float64)
        for i in range(m):
            K = epsilons[i, 0]*Kernelxx + \
                (epsilons[i, 1] + epsilons[i, 2]) * \
                Kernelxy + epsilons[i, 3]*Kernelyy
            for j in range(data_num):
                KernelA[i*data_num+j] = K
    else:
        KernelA = torch.zeros([m, 1, 3, 3])
        for i in range(m):
            KernelA[i] = epsilons[i, 0]*Kernelxx + \
                (epsilons[i, 1] + epsilons[i, 2]) * \
                Kernelxy + epsilons[i, 3]*Kernelyy

    return KernelA

# %%
class CreateMetaDataset(data.Dataset):
    def __init__(self, p, data1, data2, label, train=0):
        self.p = p
        self.train = train
        self.data1 = data1
        self.data2 = data2
        self.label = label

    def __getitem__(self, index):
        f = self.data1[index]
        kernelA = self.data2[index]
        u = self.label[index]
        return f, kernelA, u

    def __len__(self):
        return self.p

def BatchConv2d(inputs, Kernels, stride=1, padding=1):
    batch_size = inputs.shape[0]
    du = F.conv2d(inputs.permute(1,0,2,3), Kernels.to(inputs.dtype), stride=stride, padding=padding, bias=None, groups=batch_size)
    return du.permute(1,0,2,3)

def CreateTrainLoader(config, N):
    h = 1/(N+1)
    epsilons_train = torch.pow(0.1, torch.rand(config["pde_num"])*6)
    thetas_train = torch.rand(config["pde_num"])*torch.pi
    print(epsilons_train)
    etas_train = torch.zeros(config["pde_num"], 4)
    c = torch.cos(thetas_train)
    s = torch.sin(thetas_train)
    c2 = c*c
    cs = c*s
    s2 = s*s
    etas_train[:, 0] = c2 + epsilons_train*s2
    etas_train[:, 1] = cs*(1-epsilons_train)
    etas_train[:, 2] = cs*(1-epsilons_train)
    etas_train[:, 3] = s2 + epsilons_train*c2

    kernelAs = CreateKernelA(etas_train, data_num=config["every_num"], repeat=True)
    total_num = config["pde_num"]*config["every_num"]
    if config["loss"] == 'u':
        gu = torch.randn([total_num, 1, N, N], dtype=torch.float64)
        gf = BatchConv2d(gu, kernelAs)
        # gf = torch.ones_like(gu)*torch.rand(total_num, 1,1,1)
        # gf = torch.ones([total_num, 1, N, N], dtype=torch.float64)*h**2

    else:
        gf = torch.randn([total_num, 1, N, N], dtype=torch.float64)
        gu = torch.zeros([total_num, 1, N, N], dtype=torch.float64)

    dataset = CreateMetaDataset(total_num, gf, kernelAs, gu)
    dataLoader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    return dataLoader

def CreateLFALoader(config, N):
    h = 1/(N+1)
    datalfa = np.load(f"data/data_lfa{N}.npy")
    data_kernel = np.load(f"data/data_kernelA{N}.npy")
    # datau = np.load(f"data/data_u{N}.npy")
    dataf = np.ones((datalfa.shape[0],1,N,N))*h**2
    lfa = torch.from_numpy(datalfa)
    kernelA = torch.from_numpy(data_kernel)
    # u = torch.from_numpy(datau)
    f = torch.from_numpy(dataf)
    train_num = f.shape[0]
    print(f.shape, kernelA.shape, lfa.shape)
    trainset = CreateMetaDataset(train_num, f, kernelA, lfa)

    if N == 31:
        batchsize = 80
    elif N == 63:
        batchsize = 40
    elif N == 127:
        batchsize = 30
    else:
        batchsize = 4
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
    # train_loader = torch.utils.data.DataLoader(
    #     trainset,
    #     batch_size=config['batch_size'],
    #     shuffle=False,
    #     pin_memory=True,
    #     sampler=DistributedSampler(trainset),
    # )
    return train_loader

from math import pi

def CreateTestLoader(config, N):
    h = 1/(N+1)
    epsilons = torch.tensor(config["test_epsilons"]).to(torch.float64)
    print(epsilons)
    test_num = epsilons.shape[0]
    theta = pi*torch.tensor(config["test_theta"]).to(torch.float64)
    # print(theta)

    eta = torch.zeros([test_num, 4], dtype=torch.float64)
    c = torch.cos(theta)
    s = torch.sin(theta)
    c2 = c*c
    cs = c*s
    s2 = s*s
    eta[:,0] = c2 + epsilons*s2
    eta[:,1] = cs*(1-epsilons)
    eta[:,2] = cs*(1-epsilons)
    eta[:,3] = s2 + epsilons*c2
    data_num = test_num 
    kernelA = CreateKernelA(eta, data_num=data_num, repeat=False)
    gu = torch.randn([data_num,1, N, N], dtype=torch.float64)
    gf = torch.ones([data_num,1, N, N]).to(torch.float64)*h**2
    dataset = CreateMetaDataset(data_num, gf, kernelA, gu)
    dataLoader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataLoader, epsilons

