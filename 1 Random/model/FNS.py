import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from matplotlib import pyplot as plt
from model.misc import *
from model.unet import MetaUNet


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

    def Anorm(self, x):
        Ax = self.forward(x)
        l = torch.inner(x.flatten(1), Ax.flatten(1)).diag()
        return l

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

    def compute_alpha(self, r, e):
        Ae = self.forward(e)
        alpha = (
            torch.inner(r.flatten(1), e.flatten(1)).diag()
            / torch.inner(Ae.flatten(1), e.flatten(1)).diag()
        )
        return alpha.view(-1, 1, 1, 1)

    def fns(self, x, f, H, weights, smoother):
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
        e = H(r1.float(), weights)
        e = F.pad(e, pad=(1, 1, 1, 1), mode="constant", value=0).double()
        alpha = self.compute_alpha(r, e)
        x = x + alpha*e
        return x


#*********************************************
#*                Meta network               #
#*********************************************
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act):
        super(ResNetBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=1
            ),
            nn.BatchNorm2d(in_channels),
            getActivationFunction(act),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1
            ),
        )
        self.activation = nn.Sequential(
            nn.BatchNorm2d(out_channels), getActivationFunction(act)
        )

    def forward(self, x):
        out = self.layers(x) + x
        out = self.activation(out)
        return out

class Meta_T(nn.Module):
    def __init__(self, in_channels, out_channels, act):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            getActivationFunction(self.act),
            ResNetBlock(4, 4, 3, self.act),
            nn.Conv2d(4, 8, 3),
            getActivationFunction(self.act),
            ResNetBlock(8, 8, 3, self.act),
            nn.Conv2d(8, in_channels*out_channels, 3),
            getActivationFunction(self.act),
            nn.AdaptiveAvgPool2d(3),
        )

        self.fnn = nn.Sequential(
            nn.Linear(in_channels*out_channels*9, 200),
            getActivationFunction(self.act),
            nn.Linear(200, 200),
            getActivationFunction(self.act),
            nn.Linear(200, 200),
            getActivationFunction(self.act),
            nn.Linear(200, 2*in_channels*out_channels*9)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.fnn(x)
        x_real = x[:, :self.in_channels*self.out_channels*9].view(x.shape[0], self.out_channels, self.in_channels, 3, 3)
        x_imag = x[:, self.in_channels*self.out_channels*9:].view(x.shape[0], self.out_channels, self.in_channels, 3, 3)
        return x_real + 1j*x_imag
  
  
#*********************************************
#*       FNS based on spectral transform     #
#*********************************************
class FNS(nn.Module):
    def __init__(self):
        super(FNS, self).__init__()

    def forward(self, r, weights):
        N1 = r.shape[-1]
        rsym,ik2 = self.expand(r)
        N2 = rsym.shape[-1]
        pad_left = N2//4+1
        pad_right = N2 - N2//2 - pad_left
        padding = (pad_left, pad_right, pad_left, pad_right)

        weights1, weights2, weights3, weights_theta = weights

        r_hat = torch.fft.ifft2(rsym, dim=(2, 3))
        r_hat = torch.fft.fftshift(r_hat, dim=(2, 3))[:,:,N2//2-N2//4:N2//2+N2//4+1, N2//2-N2//4:N2//2+N2//4+1]
        r_hat = self.transition(r_hat, [weights1, weights2, weights3])
        out_ft = r_hat*weights_theta*ik2
        out_ft = self.transition(out_ft, [torch.transpose(weights3, -4, -3).transpose(-2, -1).conj(), torch.transpose(weights2, -4, -3).transpose(-2, -1).conj(), torch.transpose(weights1, -4, -3).transpose(-2, -1).conj()])
        out_ft = F.pad(out_ft, padding)
        out_ft = torch.fft.ifftshift(out_ft, dim=(2, 3))
        e = torch.fft.fft2(out_ft, dim=(2, 3)).real
        return e[:,:,:N1,:N1]  

    def expand(self, r):
        B, C, M, N = r.shape
        rsym = torch.zeros(B, C, 2*(M+1), 2*(N+1), dtype=r.dtype, device=r.device)
        rsym[:, :, 1:M+1, 1:N+1] = r
        rsym[:, :, M+2:, 1:N+1] = -torch.flip(r, dims=(2,)) 
        rsym[:, :, 1:M+1, N+2:] = -torch.flip(r, dims=(3,)) 
        rsym[:, :, M+2:, N+2:] = torch.flip(torch.flip(r, dims=(3,)), dims=(2,))

        N2 = rsym.shape[-1]
        kg = torch.arange(-M-1, N+1) * torch.tensor(np.pi) 
        kx, ky = torch.meshgrid(kg, kg)
        ik2 = 1 / (kx**2 + ky**2)
        ik2[M+1,N+1] = 1
        ik2 = ik2[N2//2-N2//4:N2//2+N2//4+1, N2//2-N2//4:N2//2+N2//4+1].unsqueeze(0).unsqueeze(0).to(r.device)
        return rsym, ik2 
    
    def transition(self, r, weights):
        for i in range(len(weights)):
            r = self.multi_channel_conv(r, weights[i])
        return r

    def multi_channel_conv(self, x, weights):
        conv_weights = weights.flatten(start_dim=0, end_dim=1)
        dump_result = F.conv2d(x, conv_weights, stride=1, padding=1)
        n, n_mul_cin, h, w = dump_result.size()
        dump_result = dump_result.view(n, n, n_mul_cin // n, h, w)
        x = torch.diagonal(dump_result, dim1=0, dim2=1)
        return x.permute(3, 0, 1, 2)
        

class HyperFNS(nn.Module):
    def __init__(self, config):
        super(HyperFNS, self).__init__()

        self.smoother = config["smoother"]
        self.meta1 = Meta_T(1, 4, config["act"])
        self.meta2 = Meta_T(4, 4, config["act"])
        self.meta3 = Meta_T(4, 1, config["act"])
        self.meta_type = config["Meta"]

        if config["Meta"] == "UNet":
            self.meta = MetaUNet(config)
        elif config["Meta"] == "FNO":
            self.meta = sFNO_epsilon_v2(config, 3, 2)

        self.H = FNS()

        self.dir = config["prediction_folder"]
        self.error_threshold = config["error_threshold"]
        self.max_iter_num = config["max_iter_num"]
        self.xavier_init = config["xavier_init"]
        if self.xavier_init > 0:
            self._reset_parameters()

    def forward(self, f, coef, u, epoch):
        K = (epoch-1) // 100 + 1
        A = DarcyFlow(coef)
        weights = self.setup(coef)
        x = torch.zeros_like(f, dtype=f.dtype, device=f.device)
        for i in range(K):
            x = A.fns(x, f, self.H, weights, self.smoother)
            r = f - A(x)
        loss = torch.norm(r) / torch.norm(f)
        return loss


    def setup(self, coef):
        N = coef.shape[-1] + 1
        weights1 = self.meta1(coef.float())
        weights2 = self.meta2(coef.float())
        weights3 = self.meta3(coef.float())
        if self.meta_type == "FNO":
            gridx, gridy = get_grid2D(coef.shape, coef.device)
            no_input = torch.cat((coef, gridx, gridy), 1)
            weiths_theta = self.meta(no_input.float(), N)
        elif self.meta_type == "UNet":
            weiths_theta = self.meta(coef.float(), N)
        weights = [weights1, weights2, weights3, weiths_theta]
        return weights
    
    
    def test(self, f, coef, epoch):
        res = 1
        i = 1
        N = f.shape[-1]
        weights = self.setup(coef)
        A = DarcyFlow(coef)
        normf = torch.norm(f, p=2, dim=(2, 3))
        res = normf / normf
        ress = [res]
        f1 = f[:,:,1:-1,1:-1]
        x1 = self.H(f1.float(), weights)
        plt.imshow(x1[0,0].cpu().detach().numpy(), cmap="jet")
        plt.colorbar()
        plt.savefig(self.dir + "/x1_{}_{}.png".format(N, epoch), dpi=300, bbox_inches="tight")
        plt.close()

        x = torch.zeros_like(f, dtype=f.dtype, device=f.device)
        while torch.max(res).item() > self.error_threshold and i < self.max_iter_num:
            x = A.fns(x, f, self.H, weights, self.smoother)
            r = f - A(x)
            res = torch.norm(r, p=2, dim=(2, 3)) / normf
            i = i + 1
            ress.append(res)
            print(f"res:{res[0][0]}")
        result_ress = [[tensor[i].item() for tensor in ress] for i in range(f.shape[0])]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5)) 
        im = ax1.imshow(np.abs(weights[-1][0,0].cpu().detach().numpy()), cmap="jet")
        ax1.set_title('lambda_inv', fontsize=20)
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
        return x, result_ress

    def _reset_parameters(self):
        for param in self.meta.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
            else:
                constant_(param, 1e-2)

        for param in self.meta1.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
            else:
                constant_(param, 1e-2)

        for param in self.meta2.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
            else:
                constant_(param, 1e-2)

        for param in self.meta3.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
            else:
                constant_(param, 1e-2)

 