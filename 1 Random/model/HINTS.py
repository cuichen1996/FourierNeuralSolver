import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from matplotlib import pyplot as plt
from model.misc import *
from model.DeepONet import DeepONet

class DarcyFlow(nn.Module):
    def __init__(self, a):
        super(DarcyFlow, self).__init__()
        kernel1 = torch.tensor([[[[-1 / 6, 2 / 3], [-1 / 3, -1 / 6]]]])
        kernel2 = torch.tensor([[[[2 / 3, -1 / 6], [-1 / 6, -1 / 3]]]])
        kernel3 = torch.tensor([[[[-1 / 6, -1 / 3], [2 / 3, -1 / 6]]]])
        kernel4 = torch.tensor([[[[-1 / 3, -1 / 6], [-1 / 6, 2 / 3]]]])
        self.kernel = (
            torch.vstack((kernel1, kernel2, kernel3, kernel4))
            .permute(1, 0, 2, 3)
            .to(a.device, dtype=a.dtype)
        )
        diag_kernal = 2/3*torch.tensor([[[[1, 1], [1, 1]]]]).to(a.device, dtype=a.dtype)
        self.dinv = 1 / F.conv2d(a, diag_kernal, padding=0)
        self.a = a  # diffusion coefficient
        self.device = a.device
        self.degree = 16

    def forward(self, x):
        xdim = x.dim()
        if xdim == 1:
            n = int(np.sqrt(x.shape[-1]))
            x = x.reshape(1, 1, n, n)
        N = x.shape[-1]
        x1 = x[:, :, 1:N, : N - 1] * self.a
        x2 = x[:, :, 1:N, 1:] * self.a
        x3 = x[:, :, : N - 1, 1:] * self.a
        x4 = x[:, :, : N - 1, : N - 1] * self.a
        X = torch.cat((x1, x2, x3, x4), dim=1)
        Ax = F.conv2d(X, self.kernel, padding=0)
        Ax = F.pad(Ax, pad=(1, 1, 1, 1), mode="constant", value=0)
        if xdim == 1:
            return Ax.flatten()
        elif x.dim() == 4:
            return Ax

    def Richardson(self, x, f, w, M):
        for i in range(M):
            x = x + w * (f - self.forward(x))
        return x
    
    def Jacobi(self, x, f, w, M):
        for i in range(M):
            r = f - self.forward(x)
            x[:,:,1:-1,1:-1] = x[:,:,1:-1,1:-1] + w * self.dinv * r[:,:,1:-1,1:-1]
        return x
    
    def power_method(self, x):
        with torch.no_grad():
            b_k = torch.randn_like(x, device=self.device, dtype=x.dtype)
            for i in range(50):
                b_k1 = self.forward(b_k)
                b_k = b_k1 / torch.norm(b_k1)
            mu = torch.inner(self.forward(b_k).flatten(), b_k.flatten()) / torch.inner(
                b_k.flatten(), b_k.flatten()
            )
        return mu.item()

    def chebysemi(self, x, f, alpha, M):
        lam_max = self.power_method(x)
        lam_min = lam_max / alpha
        roots = [
            np.cos((np.pi * (2 * i + 1)) / (2 * self.degree))
            for i in range(self.degree)
        ]
        good_perm_even = leb_shuffle_2n(self.degree)
        taus = torch.tensor(
            [2 / (lam_max + lam_min - (lam_min - lam_max) * r) for r in roots]
        ).to(self.device, x.dtype)
        for i in range(M):
            r = f - self.forward(x)
            x = x + taus[good_perm_even[i]].unsqueeze(-1).unsqueeze(-1) * r
        return x

    def fns(self, x, f, H, smoother):
        h = 1 / (x.shape[-1])
        if smoother == "Richardson":
            x = self.Richardson(x, f, w=40*h, M=10)  # * M times
        elif smoother == "Jacobi":
            x = self.Jacobi(x, f, w=3/4, M=10)
        elif smoother == "Chebyshev":
            x = self.chebysemi(x, f, alpha=30, M=10)
        elif smoother == "NoSmoother":
            x = x
        else:
            raise ValueError
        r = f - self.forward(x)
        r1 = r[:,:,1:-1,1:-1]
        e = H(self.a.float(), r1.float())
        e = F.pad(e, pad=(1, 1, 1, 1), mode="constant", value=0)
        alpha = 1.0
        x = x + alpha*e.double()
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

    def forward(self, f, coef, u, epoch):
        # K = (epoch-1) // 100 + 1
        K = 10
        A = DarcyFlow(coef)
        x = torch.zeros_like(f, dtype=f.dtype, device=f.device)
        for i in range(K):
            x = A.fns(x, f, self.H, self.smoother)
            r = f - A(x)
        loss = torch.norm(r) / torch.norm(f)
        return loss
    
    def test(self, f, coef, epoch):
        # print(f.shape, coef.shape)
        res = 1
        i = 1
        N = f.shape[-1]
        A = DarcyFlow(coef)
        normf = torch.norm(f, p=2, dim=(2, 3))
        res = normf / normf
        ress = [res]
        f1 = f[:,:,1:-1,1:-1]
        x1 = self.H(coef.float(), f1.float())
        x = F.pad(x1, pad=(1, 1, 1, 1), mode="constant", value=0)
        while torch.max(res).item() > self.error_threshold and i < self.max_iter_num:
            x = A.fns(x, f, self.H, self.smoother)
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
 