# -*- coding: utf-8 -*-
# @Author: Chen Cui
# @Date:   2022-04-04 11:52:32
# @Last Modified by:   Your name
# @Last Modified time: 2022-08-30 07:27:35
# %%
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from scipy.sparse import spdiags
torch.set_default_dtype(torch.float64)

def CreateKernelA(k, h, data_num=100):
    m = len(k)
    KernelA = torch.zeros([m*data_num, 1, 3, 3], dtype=torch.float64)
    for i in range(m):
        K = torch.tensor([[[[0, -1, 0], [-1, 4 - (k[i]**2) * (h**2), -1], [0, -1, 0]]]], dtype=torch.float64)
        for j in range(data_num):
            KernelA[i*data_num+j] = K
    return KernelA

def gen_single_data(k, N, num):
    h = 1/N
    f = torch.ones(num, 1, N-1, N-1)
    u = torch.zeros(num, 1, N-1, N-1)

    KernelA = torch.tensor(
        [[[[0, -1, 0], [-1, 4 - (k**2) * (h**2), -1], [0, -1, 0]]]], dtype=torch.float64)
    A = KernelToMatrix(KernelA, N-1)
    A = torch.tensor(A, dtype=torch.float64)
    for i in range(num):
        idx1 = np.random.rand(1)
        # idx1 = 1
        f[i, 0, :, :] = f[i, 0, :, :] * idx1
        ff = f[i, 0, :, :].flatten()*h**2
        # uu = torch.linalg.solve(A, ff)
        uu = torch.rand(N-1, N-1)
        f[i, 0, :, :] = ff.reshape(N-1, N-1)
        u[i, 0, :, :] = uu.reshape(N-1, N-1)
    return u, f


def KernelToMatrix(kernel, m):
    n = m*m
    batchSize = kernel.shape[0]
    if batchSize > 1:
        MM = torch.zeros([batchSize, n, n])
    for b in range(batchSize):
        K = kernel[b].view([9, 1]).cpu().numpy()
        B = K*np.ones(n)
        M = spdiags(B, [-m-1, -m, -m+1, -1, 0, 1, m-1, m, m+1], n, n)
        M = M.toarray()
        for i in range(m-1):
            M[i*m, (i+1)*m-1] = 0
            M[(i+1)*m, (i+1)*m-1] = 0
            M[(i+1)*m-1, (i+1)*m] = 0
            M[(i+1)*m-1, i*m] = 0
        M[n-m, n-1] = 0
        M[n-1, n-m] = 0
        for i in range(1, m-1):
            M[(i+1)*m, i*m-1] = 0
            M[i*m-1, (i+1)*m] = 0
        if batchSize == 1:
            return M
        else:
            MM[b] = M
    return MM

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


def BatchConv2d(inputs, kernels, stride=1, padding=1):
    batch_size = inputs.shape[0]
    m1 = inputs.shape[2]
    m2 = inputs.shape[3]
    out = F.conv2d(inputs.view(1, batch_size, m1, m2), kernels,
                   stride=stride, padding=padding, bias=None, groups=batch_size)
    return out.view(batch_size, 1, m1, m2)

def CreateTrainLoader(config):
    N = config["N"]+1
    k_train = [random.uniform(config["kmin"],config["kmax"]) for i in range(config["pde_num"])]
    gu = []
    gf = []
    for i in range(config["pde_num"]):
        print("k_train[%d] = %f" % (i, k_train[i]))
        u, f = gen_single_data(k_train[i], N, config["every_num"])
        gu.append(u)
        gf.append(f)
    gu = torch.cat(gu, dim=0)
    gf = torch.cat(gf, dim=0)

    kernelAs = CreateKernelA(k_train, h=1/N, data_num=config["every_num"])
    total_num = config["pde_num"]*config["every_num"]
    dataset = CreateMetaDataset(total_num, gf, kernelAs, gu)
    dataLoader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    return dataLoader

#%%

def CreateTestLoader(config):
    N = config["N"]+1
    k_test = torch.tensor(config["test_k"])
    test_num = k_test.shape[0]
    gu = []
    gf = []
    for i in range(test_num):
        u, f = gen_single_data(k_test[i], N, 1)
        gu.append(u)
        gf.append(f)
    gu = torch.cat(gu, dim=0)
    gf = torch.cat(gf, dim=0)
    kernelA = CreateKernelA(k_test, h=1/N, data_num=1)

    dataset = CreateMetaDataset(test_num, gf, kernelA, gu)
    dataLoader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataLoader


#%%
# config = {}
# config["pde_num"]    = 20
# config["every_num"]  = 100
# config["loss"]       = 'u' # 'u' or 'r'
# config['batch_size'] = 1
# config["N"]          = 15
# config["kmin"]       = 5
# config["kmax"]       = 20
# config["test_k"]     = [30]
# train_loader = CreateTrainLoader(config)
  
# # %%
# N = 16
# x = np.linspace(0, 1, N+1)[1:-1]
# y = np.linspace(0, 1, N+1)[1:-1]
# X, Y = np.meshgrid(x, y)
# from scipy.sparse.linalg import gmres
# import matplotlib.pyplot as plt
# for f, kernelA, u in train_loader:
#     A = KernelToMatrix(kernelA, 15)
#     f = f.numpy().flatten()
#     u = u.numpy().flatten()
#     plt.pcolor(X, Y, u.reshape(N-1, N-1))
#     plt.colorbar()
#     x = np.zeros_like(f)
#     res = 1
#     k = 0
#     ress = [res]
#     while res > 1e-6 and k < 4000:
#         x, _ = gmres(A, f, x0=x, maxiter=1, tol=1e-6)
#         res = np.linalg.norm(x-u)/np.linalg.norm(u)
#         print(res)
#         ress.append(res)
#         k += 1
#     break
# plt.semilogy(ress)
# %%
