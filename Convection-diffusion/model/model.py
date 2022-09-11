import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

# %%
# Kernel of prolongation and restriction
KernelP = torch.tensor([[[[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]]])
KernelR = torch.tensor([[[[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]]])

def BatchConv2d(inputs, kernels, stride=1, padding=0):
    batch_size = inputs.shape[0]
    m1 = inputs.shape[2]
    m2 = inputs.shape[3]
    out = F.conv2d(inputs.view(1, batch_size, m1, m2), kernels, stride=stride, padding=padding, bias=None, groups=batch_size)
    return out.view(batch_size, 1, m1-2, m2-2)

def BatchConv2d_multichannel(inputs, Kernels, stride=1, padding=0):
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

##* smoother
class ChebySemi(nn.Module):
    def __init__(self, niters=15, alpha=0.5):
        super(ChebySemi, self).__init__()
        self.niters = niters
        self.alpha = alpha
        self.roots = [np.cos((np.pi*(2*i+1)) / (2*self.niters)) for i in range(self.niters)]

    def forward(self, x, f, kernelA):
        with torch.no_grad():
            u = torch.randn_like(f).double()
            for i in range(20):
                u = F.pad(u,(1,0,1,0), "constant", 0)
                u = F.pad(u,(0,1,0,1), "constant", 1)
                y = BatchConv2d(u, kernelA)      
                m = torch.max(torch.max(torch.abs(y),dim=2).values, dim=2).values.reshape(x.shape[0], 1, 1, 1)
                u = y / m
            taus = [2 / (m + m*self.alpha - (m*self.alpha - m) * r) for r in self.roots]
        for k in range(self.niters):
            x_pad = F.pad(x,(1,0,1,0), "constant", 0)
            x_pad = F.pad(x_pad,(0,1,0,1), "constant", 1)
            Ax = BatchConv2d(x_pad, kernelA) 
            x = x + taus[k]*(f - Ax)
        return x

class Jacobi(nn.Module):
    def __init__(self, w, k):
        super(Jacobi, self).__init__()
        self.w = w
        self.k = k
    def forward(self, x, f, kernelA):
        weight = self.w/kernelA[:,:,1,1]
        weight = weight.unsqueeze(1).unsqueeze(1)
        for i in range(self.k):
            x_pad = F.pad(x,(1,0,1,0), "constant", 0)
            x_pad = F.pad(x_pad,(0,1,0,1), "constant", 1)
            Ax = BatchConv2d(x_pad, kernelA) 
            x = x + weight*(f - Ax)
        return x
 
class MetaConvSmoother(nn.Module):
    def __init__(self, act, mL=3, kernelSize=7):
        super(MetaConvSmoother, self).__init__()
        self.mL = mL
        self.kernelSize = kernelSize
        self.fc1 = nn.Sequential(nn.Linear(9, 100), 
                                 getActivationFunction(act),
                                 nn.Linear(100, mL*kernelSize*kernelSize)).double()
        self.fc2 = nn.Sequential(nn.Linear(9, 100), 
                                 getActivationFunction(act),
                                 nn.Linear(100, mL*kernelSize*kernelSize)).double()

    def forward(self, x, f, kernelA):
        batch_size, N = x.shape[0], x.shape[-1]
        
        weight = torch.flatten(kernelA, 1)
        weight1 = self.fc1(weight).view([batch_size,self.mL,1,self.kernelSize,self.kernelSize])
        weight2 = self.fc2(weight).view([batch_size,1,self.mL,self.kernelSize,self.kernelSize])

        G2 = torch.zeros(batch_size, 1, N, N, dtype=x.dtype, device=x.device)
        x_pad = F.pad(x,(1,0,1,0), "constant", 0)
        x_pad = F.pad(x_pad,(0,1,0,1), "constant", 1)
        Ax = BatchConv2d(x_pad, kernelA) 
        r = f - Ax
        for i in range(batch_size):
            tmp = F.conv2d(r[i:i+1], weight1[i], padding=self.kernelSize//2)
            G2[i] = F.conv2d(tmp, weight2[i], padding=self.kernelSize//2)
        x = x + G2
        return x
        
class MetaScSmoother(nn.Module):
    def __init__(self, mL = 3, kernelSize = 7):
        super(MetaScSmoother, self).__init__()
        self.mL = mL
        self.L = 2*mL+1
        self.kernelSize = kernelSize
        self.fc1 = nn.Sequential(nn.Linear(9, 200), nn.ReLU(inplace=True),
                                nn.Linear(200, mL*kernelSize*kernelSize))
        self.fc2 = nn.Sequential(nn.Linear(9, 200), nn.ReLU(inplace=True),
                                nn.Linear(200, mL*mL*kernelSize*kernelSize))

    def forward(self, r, kernelA):
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
        return torch.matmul(K.permute((0,2,1)), G.view(-1, self.L, mu*mu)).view(-1, 1, mu, mu)


#*CNN
class CNNFNS(nn.Module):
    def __init__(self, mid_chanel, act):
        super(CNNFNS, self).__init__()
        self.meta = nn.Sequential(
            nn.ConvTranspose2d(1,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),           
            nn.ConvTranspose2d(mid_chanel,mid_chanel,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(mid_chanel,2,3,stride=2,padding=2)).double()
        
    def forward(self, r, kernelA):
        batch_size, N = r.shape[0], r.shape[2]
        weights = self.meta(kernelA).view(batch_size, 1, N, N, 2)
        weights = torch.view_as_complex(weights)
        
        r_hat = torch.fft.fft2(r, norm='ortho')
        out_ft = r_hat * weights
        e = torch.fft.irfft2(out_ft, dim=(2, 3), s=(r.size(-2), r.size(-1)), norm='ortho')  
        return e
    

#%%
class HyperFNS(nn.Module):
    def __init__(self, config):
        super(HyperFNS, self).__init__()
        self.N = config["N"]
        self.smoother = MetaConvSmoother(config["act"], config["mL"],config["kernel_size"])
        self.H = CNNFNS(config["mid_chanel"], config["act"]).double() 
    
        self.K = config["K"]
        self.error_threshold = config["error_threshold"]
        self.max_iter_num = config["max_iter_num"]

        self.xavier_init = config["xavier_init"]
        if self.xavier_init > 0:
            self._reset_parameters()

    def forward(self, x, f, kernelA):
        # r = f
        for i in range(self.K):
            x = self.smoother(x, f, kernelA)
            x_pad = F.pad(x,(1,0,1,0), "constant", 0)
            x_pad = F.pad(x_pad,(0,1,0,1), "constant", 1)           
            r = f - BatchConv2d(x_pad, kernelA)
            e = self.H(r, kernelA)
            x = x + e
        # compute loss
        x_pad = F.pad(x,(1,0,1,0), "constant", 0)
        x_pad = F.pad(x_pad,(0,1,0,1), "constant", 1)           
        r = f - BatchConv2d(x_pad, kernelA)
        res = torch.norm(r) / x.shape[0]
        return res

    def test(self, x, f, kernelA, u):
        x_pad = F.pad(x,(1,0,1,0), "constant", 0)
        x_pad = F.pad(x_pad,(0,1,0,1), "constant", 1)    
        r = f - BatchConv2d(x_pad, kernelA)              
        res1 = torch.norm(r)
        res = 1
        i = 1
        h = 1/(x.shape[-1]-1)
        residual = [1]
        while res > self.error_threshold and i < self.max_iter_num:
            x = self.smoother(x, f, kernelA)               
            x_pad = F.pad(x,(1,0,1,0), "constant", 0)
            x_pad = F.pad(x_pad,(0,1,0,1), "constant", 1)    
            r = f - BatchConv2d(x_pad, kernelA)              
            e = self.H(r, kernelA) 
            x = x + e                                       
            x_pad = F.pad(x,(1,0,1,0), "constant", 0)
            x_pad = F.pad(x_pad,(0,1,0,1), "constant", 1)   
            r = f - BatchConv2d(x_pad, kernelA)              
            res = torch.norm(r)/res1
            i = i + 1
            residual.append(res.item())
            print(f"i:{i}, res:{res.item()}")
        return res, i

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
                constant_(param, 1e-1)    
                
