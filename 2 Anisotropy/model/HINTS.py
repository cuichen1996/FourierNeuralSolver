import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from matplotlib import pyplot as plt
from model.misc import *
from model.DeepONet import DeepONet

class Anisotropy(nn.Module):
    def __init__(self, kernel):
        super(Anisotropy, self).__init__()
        self.kernel = kernel 
        self.device = kernel.device
        self.degree = 16

    def forward(self, x):
        if x.dim() == 4:
            batch_size = x.shape[0]
            y = F.conv2d(x.permute(1,0,2,3), self.kernel.to(x.device,x.dtype), stride=1, padding=1, bias=None, groups=batch_size)
            return y.permute(1,0,2,3)
        elif x.dim() == 1:
            n = int(np.sqrt(x.shape[-1]))
            x = x.reshape(1,1,n,n)
            y = F.conv2d(x, self.kernel.to(x.device,x.dtype), stride=1, padding=1, bias=None)
            return y.flatten()

    def Anorm(self, x):
        Ax = self.forward(x)
        l = torch.inner(x.flatten(1), Ax.flatten(1)).diag()
        return l

    def jacobi(self, x, f, w, M):
        taus = w/self.kernel[:, :, 1:2,1:2]
        for i in range(M):
            x = x + taus*(f - self.forward(x))
        return x

    def power_method(self, x):
        with torch.no_grad():
            b_k = torch.randn_like(x, device=self.device, dtype=x.dtype)
            for i in range(50):
                b_k1 = self.forward(b_k)
                b_k = b_k1 / torch.norm(b_k1)
            mu = torch.inner(
                self.forward(b_k).flatten(), b_k.flatten()
            ) / torch.inner(b_k.flatten(), b_k.flatten())
        return mu.item()
    
    def chebysemi(self, x, f, alpha, M):
        lam_max = self.power_method(x)
        lam_min = lam_max / alpha
        roots = [np.cos((np.pi * (2 * i + 1)) / (2 * self.degree)) for i in range(self.degree)]
        good_perm_even = leb_shuffle_2n(self.degree)
        taus = torch.tensor([2 / (lam_max + lam_min - (lam_min - lam_max) * r) for r in roots]).to(self.device, x.dtype)
        for i in range(M):
            r = f - self.forward(x)
            x = x + taus[good_perm_even[i]].unsqueeze(-1).unsqueeze(-1) * r
        return x

    def fns(self, x, f, LFA, H, smoother):
        if smoother == "jacobi":
            x = self.jacobi(x, f, w=0.5, M=10) #* M times
        elif smoother == "chebyshev":
            x = self.chebysemi(x, f, alpha=3, M=5)
        elif smoother == "none":
            x = x
        else:
            raise ValueError
        r = f - self.forward(x)
        e = H(LFA, r.float())
        x = x + e.double()
        return x

  

class HyperFNS(nn.Module):
    def __init__(self, config):
        super(HyperFNS, self).__init__()
        self.smoother = config["smoother"]
        self.H = DeepONet(config)

        self.dir = config["prediction_folder"]
        self.error_threshold = config["error_threshold"]
        self.max_iter_num = config["max_iter_num"]
        self.xavier_init = config["xavier_init"]
        if self.xavier_init > 0:
            self._reset_parameters()

    def forward(self, f, kernelA, u, epoch):
        K = (epoch-1) // 100 + 1
        A = Anisotropy(kernelA)
        LFA = self.ComputeSmootherFactor(kernelA, f.shape[-1])
        x = torch.zeros_like(f, dtype=f.dtype, device=f.device)
        for i in range(K):
            x = A.fns(x, f, LFA, self.H, self.smoother)
            r = f - A(x)
        loss = torch.norm(r) / torch.norm(f)
        return loss

    def ComputeSmootherFactor(self, KernelA, N):
        device = KernelA.device
        w = 1/2
        h = 1/(N+1)
        p1 = range(-N//2, N//2)
        p2 = range(-N//2, N//2)
        P1, P2 = np.meshgrid(p1, p2)
        P1 = torch.from_numpy(P1)
        P2 = torch.from_numpy(P2)
        P1, P2 = P1.to(device), P2.to(device)
        theta1 = 2j*np.pi*P1*h
        theta2 = 2j*np.pi*P2*h
        Y = torch.ones([KernelA.shape[0], 1, N, N], device=device)
        for i in range(KernelA.shape[0]):
            k1, k2, k3, k4, k5, k6, k7, k8, k9 = KernelA[i].flatten()[:]
            taus = [w/KernelA[i, 0, 1, 1]]*5
            y = 1
            for j in range(len(taus)):
                y *= 1-taus[j] * (k1*torch.exp(-theta1)*torch.exp(-theta2)+k2*torch.exp(-theta2)+k3*torch.exp(theta1)*torch.exp(-theta2)+k4*torch.exp(-theta1)+k5+k6*torch.exp(theta1)+k7*torch.exp(-theta1)*torch.exp(theta2)+k8*torch.exp(theta2)+k9*torch.exp(theta1)*torch.exp(theta2))
            Y[i, 0, :, :] = torch.abs(y)
        return Y
    
    def test(self, f, kernelA, epoch):
        res = 1
        i = 1
        N = f.shape[-1]
        lfa = self.ComputeSmootherFactor(kernelA, N)
        A = Anisotropy(kernelA)
        normf = torch.norm(f, p=2, dim=(2, 3))
        res = normf / normf
        ress = [res]
        f1 = f[:,:,1:-1,1:-1]
        x1 = self.H(lfa, f1.float())
        x = x1.double()
        while torch.max(res).item() > self.error_threshold and i < self.max_iter_num:
            x = A.fns(x, f, lfa, self.H, self.smoother)
            r = f - A(x)
            res = torch.norm(r, p=2, dim=(2, 3)) / normf
            i = i + 1
            ress.append(res)
            print(f"res:{res[0][0]}")
        result_ress = [[tensor[i].item() for tensor in ress] for i in range(f.shape[0])]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5)) 
        im = ax1.imshow(x1[0,0].cpu().detach().numpy(), cmap="jet")
        ax1.set_title('initial', fontsize=20)
        cbar = fig.colorbar(im, ax=ax1) 
        im = ax2.imshow(x[0,0].cpu().detach().numpy(), cmap="jet", extent=[0,1,0,1])
        ax2.set_title('Solution', fontsize=20)
        cbar = fig.colorbar(im, ax=ax2) 
        im = ax3.semilogy(result_ress[0], "-o")
        ax3.set_title('error', fontsize=20)
        ax3.set_xlabel('Iterations', fontsize=20)
        ax3.set_ylabel('Relative residual', fontsize=20)
        ax3.grid()
        plt.title(r"$N={}$".format(N), fontsize=20)
        plt.savefig(self.dir+"/results_{}_{}.png".format(N, epoch), dpi=300, bbox_inches='tight')
        plt.close()
        print(len(ress)-1)
        return x, result_ress

    def _reset_parameters(self):
        for param in self.H.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
            else:
                constant_(param, 1e-2)
 