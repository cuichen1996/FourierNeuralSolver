# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2022-09-11 05:47:33
# @Last Modified by:   Your name
# @Last Modified time: 2022-09-11 05:56:10
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
# %%
# Kernel of prolongation and restriction
KernelP = torch.tensor([[[[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]]])
KernelR = torch.tensor([[[[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]]])

def BatchConv2d(inputs, kernels, stride=1, padding=1):
    batch_size = inputs.shape[0]
    m1 = inputs.shape[2]
    m2 = inputs.shape[3]
    out = F.conv2d(inputs.view(1, batch_size, m1, m2), kernels, stride=stride, padding=padding, bias=None, groups=batch_size)
    return out.view(batch_size, 1, m1, m2)

def BatchConv2d_multichannel(inputs, Kernels, stride=1, padding=1):
    batch_size = inputs.shape[0]
    du = F.conv2d(inputs.permute(1,0,2,3), Kernels, stride=stride, padding=padding, bias=None, groups=batch_size)
    return du.permute(1,0,2,3)

def getActivationFunction(
    act_function_name: str, features=None, end=False
) -> nn.Module:
    if act_function_name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif act_function_name.lower() == "celu":
        return nn.CELU(inplace=True)
    elif act_function_name.lower() == "relu_batchnorm":
        if end:
            return nn.ReLU(inplace=True)
        else:
            return nn.Sequential(nn.ReLU(inplace=True), nn.BatchNorm2d(features))
        return nn.CELU(inplace=True)
    elif act_function_name.lower() == "tanh":
        return nn.Tanh()
    elif act_function_name.lower() == "prelu":
        return nn.PReLU()
    elif act_function_name.lower() == "gelu":
        return nn.GELU()
    elif act_function_name.lower() == "tanhshrink":
        return nn.Tanhshrink()
    elif act_function_name.lower() == "softplus":
        return nn.Softplus()
    elif act_function_name.lower() == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    else:
        err = "Unknown activation function {}".format(act_function_name)
        raise NotImplementedError(err)

##* smoother \Phi
class ChebySemi(nn.Module):
    def __init__(self, alpha, niters=15):
        super(ChebySemi, self).__init__()
        self.niters = niters
        self.alpha = alpha
        self.roots = [np.cos((np.pi*(2*i+1)) / (2*self.niters)) for i in range(self.niters)]

    def forward(self, x, f, kernelA):
        with torch.no_grad():
            u = torch.randn_like(f).double()
            for i in range(20):
                y = BatchConv2d(u, kernelA)      
                m = torch.max(torch.max(torch.abs(y),dim=2).values, dim=2).values.reshape(x.shape[0], 1, 1, 1)
                u = y / m
            taus = [2 / (m + m/self.alpha - (m/self.alpha - m) * r) for r in self.roots]
        for k in range(self.niters):
            Ax = BatchConv2d(x, kernelA) 
            x = x + taus[k]*(f - Ax)
        return x
    
class Jacobi(nn.Module):
    def __init__(self, w):
        super(Jacobi, self).__init__()
        self.w = w
    def forward(self, x, f, kernelA):
        weight = self.w/kernelA[:,:,1,1]
        weight = weight.unsqueeze(1).unsqueeze(1)
        Ax = BatchConv2d(x, kernelA) 
        x = x + weight*(f - Ax)
        return x

class MetaScSmoother(nn.Module):
    def __init__(self, mL=3, kernelSize=7):
        super(MetaScSmoother, self).__init__()
        self.mL = mL
        self.L = 2*mL+1
        self.kernelSize = kernelSize
        self.fc1 = nn.Sequential(nn.Linear(9, 200), nn.ReLU(inplace=True),
                                nn.Linear(200, mL*kernelSize*kernelSize))
        self.fc2 = nn.Sequential(nn.Linear(9, 200), nn.ReLU(inplace=True),
                                nn.Linear(200, mL*mL*kernelSize*kernelSize))

    def forward(self, x, f, kernelA):
        r = f - BatchConv2d(x, kernelA)
        batchSize = r.shape[0]
        mu=r.shape[2]
        weights1 = self.fc1(kernelA.view(batchSize, 9)).view(batchSize, self.mL, 1, self.kernelSize, self.kernelSize)
        weights2 = self.fc2(kernelA.view(batchSize, 9)).view(batchSize, self.mL, self.mL, self.kernelSize, self.kernelSize)
        
        G1 = torch.zeros(batchSize, self.mL, mu, mu, dtype=r.dtype, device=r.device)
        G2 = torch.zeros(batchSize, self.mL, mu, mu, dtype=r.dtype, device=r.device)

        for i in range(batchSize):
            G1[i] = F.conv2d(r[i:i+1], weights1[i], padding = self.kernelSize//2)
        for i in range(batchSize):
            G2[i] = F.conv2d(G1[i:i+1], weights2[i], padding = self.kernelSize//2)
        G = torch.cat((r, G1, G2), dim = 1)
        S = BatchConv2d_multichannel(G, kernelA).view(-1, self.L, mu*mu)
        lr = r.view(-1, mu*mu, 1)
        M = torch.matmul(S, S.permute(0,2,1))
        b = torch.matmul(S, lr)
        K = torch.linalg.solve(M, b)
        x = x + torch.matmul(K.permute((0,2,1)), G.view(-1, self.L, mu*mu)).view(-1, 1, mu, mu)
        return x 

#* H with Hadamard product
class PoissonH(nn.Module):
    def __init__(self, mid_chanel, act, N):
        super(PoissonH, self).__init__()
        self.meta = nn.Sequential(
            nn.ConvTranspose2d(1,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),           
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),           
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(mid_chanel,2,3,stride=2,padding=2),
            nn.AdaptiveAvgPool2d(N)).double()
        
    def forward(self, r, kernelA):
        batch_size, N = r.shape[0], r.shape[2]
        weights = self.meta(kernelA).view(batch_size, 1, N, N, 2)
        weights = torch.view_as_complex(weights)
        r_hat = torch.fft.fft2(r, norm='ortho')
        out_ft = r_hat * weights
        e = torch.fft.irfft2(out_ft, dim=(2, 3), s=(r.size(-2), r.size(-1)), norm='ortho')  
        return e
    
#* H with matrix multiplication
class KernelH(nn.Module):
    def __init__(self, mid_chanel, act, N):
        super(KernelH, self).__init__()
        self.meta = nn.Sequential(
            nn.ConvTranspose2d(1,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),           
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),           
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(mid_chanel,2,3,stride=2,padding=2),
            nn.AdaptiveAvgPool2d(N)).double()

    def compl_mul2d(self, input, weights):
        return torch.einsum("bcxy,bcyz->bcxz", input, weights)

    def forward(self, r, kernelA):
        batch_size, N = r.shape[0], r.shape[2]
        weights = self.meta(kernelA).view(batch_size, 1, N, N, 2)
        weights = torch.view_as_complex(weights)
        
        r_hat = torch.fft.fft2(r, norm='ortho')
        out_ft = self.compl_mul2d(r_hat, weights)
        e = torch.fft.irfft2(out_ft, dim=(2, 3), s=(r.size(-2), r.size(-1)), norm='ortho')  
        return e
#%%
class HyperFNS(nn.Module):
    def __init__(self, config):
        super(HyperFNS, self).__init__()
        self.N = config["N"]
        self.smoother_times = config["smoother_times"]
        self.smoother = ChebySemi(config["alpha"], config["m"])
        self.H = PoissonH(config["mid_chanel"], config["act"], self.N).double() 
        
        self.K = config["K"]
        self.error_threshold = config["error_threshold"]
        self.max_iter_num = config["max_iter_num"]

        self.xavier_init = config["xavier_init"]
        if self.xavier_init > 0:
            self._reset_parameters()

    def forward(self, x, f, kernelA):
        for i in range(self.K):
            for j in range(self.smoother_times):
                x = self.smoother(x, f, kernelA)
            r = f - BatchConv2d(x, kernelA)
            e = self.H(r, kernelA)
            x = x + e
        r = f - BatchConv2d(x, kernelA)
        res = torch.norm(r) 
        return res

    def test(self, x, f, kernelA):
        res = 1
        i = 1
        while res > self.error_threshold and i < self.max_iter_num:
            for j in range(self.smoother_times):
                x = self.smoother(x, f, kernelA)
            r = f - BatchConv2d(x, kernelA)
            e = self.H(r, kernelA)
            x = x + e
            r = f - BatchConv2d(x, kernelA)
            res = torch.norm(r) / torch.norm(f)
            i = i + 1
            print(f"res:{res.item()}")
        return x, i

    def _reset_parameters(self):
        for param in self.H.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
            else:
                constant_(param, 1e-2)       
                
        for param in self.smoother.parameters():
            if param.ndim > 2:
                xavier_uniform_(param, gain=self.xavier_init)
            else:
                constant_(param, 1e-2)    
                
