import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from matplotlib import pyplot as plt
from model.misc import *
from model.unet import MetaUNet
from model.sFNO import sFNO_epsilon_v2
from model.torch_fgmres import fgmres_res


class Helmholtz(nn.Module):
    def __init__(self, kappa: torch.Tensor,  omega: float) -> None:
        super().__init__()
        self.degree = 16
        self.kappa = kappa
        self.omega = omega
        self.h = 1 / (kappa.shape[-2] - 1)
        self.device = kappa.device
        self.laplace_kernel = (1.0 / (self.h**2)) * torch.tensor(
            [[[[0, -1.0, 0], [-1, 4, -1], [0, -1, 0]]]]
        )

    def generate_helmholtz_matrix(self):
        return self.omega**2 * self.kappa**2 

    def generate_shifted_laplacian(self, alpha=1.0, beta=0.5):
        return self.generate_helmholtz_matrix() * (alpha + beta*1j) 

    def robin_bc(self, x):
        n = x.shape[2]
        m = x.shape[3]
        xx = torch.zeros((x.shape[0], 1, n + 2, m + 2), dtype=x.dtype, device=x.device)
        xx[:, :, 1:n + 1, 1:m + 1] = x

        xx[:, :, 0, 1:m + 1] = x[:, :, 1, :] + 2*self.h*(1j*self.omega*x[:, :, 0, :])
        xx[:, :, 1:n + 1, 0] = x[:,:, :, 1] + 2*self.h*(1j*self.omega*x[:,:, :, 0])
        xx[:, :, -1, 1:m + 1] = x[:, :, -2, :] + 2*self.h*(1j*self.omega*x[:, :, -1, :])
        xx[:, :, 1:n + 1, -1] = x[:,:, :, -2] + 2*self.h*(1j*self.omega*x[:,:, :, -1])
        return xx
    
    def matvec(self, x: torch.Tensor, SL=False) -> torch.Tensor:
        X = self.robin_bc(x)
        Dx = F.conv2d(X, self.laplace_kernel.to(X.device, X.dtype), padding=0)
        if SL:
            Rx = self.generate_shifted_laplacian()*x
        else:
            Rx = self.generate_helmholtz_matrix()*x
        return Dx - Rx

    def matvec_conj(self, x: torch.Tensor, SL=False) -> torch.Tensor:
        X = self.robin_bc(x) 
        Dx = F.conv2d(X, self.laplace_kernel.to(X.device, X.dtype))
        if SL:
            Rx = self.generate_shifted_laplacian().conj()*x
        else:
            Rx = self.generate_helmholtz_matrix().conj()*x
        return Dx - Rx
    

    def forward(self, x, SL=False):
        batch, c, m, n = self.kappa.shape
        dims = x.dim()
        if dims == 2:
            x = x.view(m, n, c, batch).permute(3, 2, 0, 1)
        elif dims == 1:
            x = x.view(1, 1, m, n)
        if SL:
            y = self.matvec(x, SL)
        else:
            y = self.matvec(x)

        if dims == 2:
            return y.permute(2, 3, 1, 0).reshape(-1, batch)
        elif dims == 1:
            return y.flatten()
        else:
            return y

    def helm_normal(self, x):
        x = self.matvec_conj(self.matvec(x))
        return x

    def power_method(self):
        with torch.no_grad():
            b_k = torch.randn(self.kappa.shape, dtype=torch.cfloat, device=self.device)
            for i in range(50):
                b_k1 = self.helm_normal(b_k)
                b_k = b_k1 / torch.norm(b_k1)
            mu = torch.inner(
                self.helm_normal(b_k).flatten(), b_k.flatten()
            ) / torch.inner(b_k.flatten(), b_k.flatten())
        return mu.item().real

    def chebysemi(self, x, b, alpha, M):
        lam_max = self.power_method()
        lam_min = lam_max / alpha
        roots = [np.cos((np.pi * (2 * i + 1)) / (2 * self.degree)) for i in range(self.degree)]
        good_perm_even = leb_shuffle_2n(self.degree)
        taus = [2 / (lam_max + lam_min - (lam_min - lam_max) * r) for r in roots]
        b = self.matvec_conj(b)
        for i in range(M):
            r = b - self.helm_normal(x)
            x = x + taus[good_perm_even[i]].unsqueeze(-1).unsqueeze(-1) * r
        return x

    def jacobi(self, x, b, w, M=1, SL=False):
        D = 4 / self.h**2
        if SL:
            R = self.omega**2 * self.kappa**2 *(1+0.5j)
        else:
            R = self.omega**2 * self.kappa**2
        Dinv = 1 / (D - R)
        for i in range(M):
            r = b - self.forward(x, SL)
            x = x + w * Dinv * r
        return x

    def fns(self, x, f, H, weights, smoother, SL=False):
        h = 1 / (x.shape[-1])
        if smoother == "Jacobi":
            x = self.jacobi(x, f, w=2/3, M=3, SL=SL)
        elif smoother == "Chebyshev":
            x = self.chebysemi(x, f, alpha=30, M=10)
        elif smoother == "NoSmoother":
            x = x
        else:
            raise ValueError
        r = f - self.forward(x, SL)
        e = H(r, weights)
        x = x + e
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
#                    FNS                     #
#*********************************************
class FNS(nn.Module):
    def __init__(self):
        super(FNS, self).__init__()

    def forward(self, r, weights):
        N1 = r.shape[-1]
        rsym = self.expand(r)
        N2 = rsym.shape[-1]
        pad_left = N2//4+1
        pad_right = N2 - N2//2 - pad_left
        padding = (pad_left, pad_right, pad_left, pad_right)

        weights1, weights2, weights3, weights_theta = weights

        r_hat = torch.fft.ifft2(rsym, dim=(2, 3))
        r_hat = torch.fft.fftshift(r_hat, dim=(2, 3))[:,:,N2//2-N2//4:N2//2+N2//4+1, N2//2-N2//4:N2//2+N2//4+1]
        r_hat = self.transition(r_hat, [weights1, weights2, weights3])
        out_ft = r_hat*weights_theta
        out_ft = self.transition(out_ft, [torch.transpose(weights3, -4, -3).transpose(-2, -1).conj(), torch.transpose(weights2, -4, -3).transpose(-2, -1).conj(), torch.transpose(weights1, -4, -3).transpose(-2, -1).conj()])
        out_ft = F.pad(out_ft, padding)
        out_ft = torch.fft.ifftshift(out_ft, dim=(2, 3))
        e = torch.fft.fft2(out_ft, dim=(2, 3))
        return e[:,:,:N1,:N1]  

    def expand(self, r):
        B, C, M, N = r.shape
        rsym = torch.zeros(B, C, 2*(M+1), 2*(N+1), dtype=r.dtype, device=r.device)
        rsym[:, :, 1:M+1, 1:N+1] = r
        rsym[:, :, M+2:, 1:N+1] = -torch.flip(r, dims=(2,)) #上下翻转
        rsym[:, :, 1:M+1, N+2:] = -torch.flip(r, dims=(3,)) #左右翻转
        rsym[:, :, M+2:, N+2:] = torch.flip(torch.flip(r, dims=(3,)), dims=(2,))
        return rsym
    
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

        if config["Meta_Type"] == "sFNO":
            self.meta = sFNO_epsilon_v2(config, 3, 2)
        elif config["Meta_Type"] == "UNet":
            self.meta = MetaUNet(config)

        self.H = FNS()

        self.dir = config["prediction_folder"]
        self.error_threshold = config["error_threshold"]
        self.max_iter_num = config["max_iter_num"]
        self.xavier_init = config["xavier_init"]
        if self.xavier_init > 0:
            self._reset_parameters()

    def forward(self, f, kappa, u, epoch):
        B, N = kappa.shape[0], kappa.shape[-2]
        sl = False
        if N == 128:
            omega = 20 * np.pi
        elif N == 256:
            omega = 40 * np.pi
        else:
            omega = 80 * np.pi

        A = Helmholtz(kappa, omega)
        weights = self.setup(kappa, omega)

        x = torch.zeros_like(f, dtype=f.dtype, device=f.device)
        K = (epoch-1) // 40 + 1
        for i in range(K):
            x = A.fns(x, f, self.H, weights, smoother=self.smoother, SL=sl)
        res = torch.norm(f - A(x, sl), p=2, dim=(2, 3))  
        return torch.mean(res) / torch.mean(torch.norm(f, p=2, dim=(2, 3)))


    def setup(self, kappa, omega):
        N = kappa.shape[-1]
        kappa = F.interpolate(kappa, size=(128,128), mode='nearest')
        gridx, gridy = get_grid2D(kappa.shape, kappa.device)
        no_input = torch.cat((kappa, gridx, gridy), 1)
        weights1 = self.meta1(kappa)
        weights2 = self.meta2(kappa)
        weights3 = self.meta3(kappa)
        T = self.meta(no_input, 129)
        T_low, T1 = T[:,:1,:,:],T[:,1:,:,:]

        weights_theta0 = torch.fft.fftshift(torch.fft.ifft2(T_low,dim=(2, 3)), dim=(2, 3))
        T0 = self.constant_T(kappa.shape[-1]+1).to(kappa.device)
        lambda_inv = torch.exp(-1j*omega*T0*T1)
        weights_theta1 = torch.fft.fftshift(torch.fft.ifft2(lambda_inv, dim=(2, 3)), dim=(2, 3))
        weights_theta = weights_theta0 + weights_theta1
        weights_theta = F.interpolate(weights_theta.real, size=(N+1, N+1), mode='bilinear', align_corners=True) + 1j*F.interpolate(weights_theta.imag, size=(N+1, N+1), mode='bilinear', align_corners=True)
        weights = [weights1, weights2, weights3, weights_theta]

        return weights
    
    def constant_T(self, n):
        h = 1 / (n-1)
        src = [n//2, n//2]
        source1 = (src[0]) * h
        source2 = (src[1]) * h
        X1, X2 = torch.meshgrid(torch.arange(0, n) * h - source1, torch.arange(0, n) * h - source2)
        # print(n, X1.shape)
        T = torch.sqrt(X1**2 + X2**2)
        return T.unsqueeze(0).unsqueeze(0)
    
    def test(self, f, kappa, epoch):
        res = 1
        i = 1
        N = kappa.shape[-2]
        if N == 128:
            omega = 20 * np.pi
        elif N == 256:
            omega = 40 * np.pi
        else:
            omega = 80 * np.pi

        A = Helmholtz(kappa, omega)
        weights = self.setup(kappa, omega)
        
        normf = torch.norm(f, p=2, dim=(2, 3))
        res = normf / normf
        ress = [res]
        x1 = self.H(f, weights)
        plt.imshow(x1[0,0].cpu().detach().numpy().real, cmap="jet")
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

        print("GMRES test")
        def M(v):
            f = v.view(1,1,N,N)
            x = torch.zeros_like(f, dtype=f.dtype, device=f.device)
            x = A.fns(x, f, self.H, weights, self.smoother, SL=False)
            return x.flatten()
        tic = time.time()
        haha = fgmres_res(A, f.flatten(), rel_tol=1e-6, max_restarts=20, max_iter=100, flexible=True, precond=M)
        ress_gmres = [1.] + haha.residual_norms
        sols = haha.solution.view(N,N)
        PGMRES_time = time.time() - tic
        print("PGMRES test finished in {:.2f} s".format(PGMRES_time))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5)) 
        im = ax1.imshow(np.abs(weights[-1][0,0].cpu().detach().numpy()), cmap="jet")
        ax1.set_title('lambda_inv', fontsize=20)
        cbar = fig.colorbar(im, ax=ax1) 
        im = ax2.imshow(sols.cpu().numpy().real, cmap="jet", extent=[0,1,0,1])
        ax2.set_title('Solution', fontsize=20)
        cbar = fig.colorbar(im, ax=ax2) 
        ax3.semilogy(result_ress[0], "-o", label="stand-alone")
        ax3.semilogy(ress_gmres, "-o", label="accelerated")
        ax3.set_title('error', fontsize=20)
        ax3.set_xlabel('Iterations', fontsize=20)
        ax3.set_ylabel('Relative residual', fontsize=20)
        ax3.grid()
        ax3.legend(fontsize=20)
        plt.title(r"$N={}$".format(N), fontsize=20)
        plt.savefig(self.dir+"/results_{}_{}.png".format(N, epoch), dpi=300, bbox_inches='tight')
        plt.close()
        return x, result_ress[0]


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