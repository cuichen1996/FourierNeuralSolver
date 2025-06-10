import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from matplotlib import pyplot as plt
from model.misc import *
from model.sFNO import sFNO_epsilon_v2
from model.unet import MetaUNet


class Advection(nn.Module):
    def __init__(self, kernel):
        super(Advection, self).__init__()
        self.kernel = kernel #[B, 1, 3, 3] SUPG stencil for advection-diffusion equation on structured grid
        self.device = kernel.device
        self.degree = 64

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

    def conv_smoother(self, x, f, weight1, weight2):
        weight1, weight2 = weight1.to(x.device, x.dtype), weight2.to(x.device, x.dtype)
        kernelsize1 = weight1.shape[-1]
        kernelsize2 = weight2.shape[-1]
        r = f - self.forward(x)
        e = torch.zeros_like(r, dtype=r.dtype, device=r.device)
        for i in range(x.shape[0]):
            tmp = F.conv2d(r[i:i+1], weight1[i], padding=kernelsize1//2)
            e[i] = F.conv2d(tmp, weight2[i], padding=kernelsize2//2)
        x = x + e
        return x
    
    def fns(self, x, f, H, weights, smoother):
        if smoother == "jacobi":
            x = self.jacobi(x, f, w=1/2, M=20) #* M times
        elif smoother == "chebyshev":
            x = self.chebysemi(x, f, alpha=30, M=20)
        elif smoother == "cnn":
            x = self.conv_smoother(x, f, weights[-2], weights[-1])
        else:
            raise ValueError
        r = f - self.forward(x)
        if smoother == "jacobi":
            e = H(r.float(), weights).to(x.dtype)
        elif smoother == "cnn":
            e = H(r.float(), weights[:4]).to(x.dtype) 
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
            nn.Conv2d(1, 4, 3, stride=1, padding=1),
            getActivationFunction(self.act),
            ResNetBlock(4, 4, 3, self.act),
            nn.Conv2d(4, 8, 3, stride=1, padding=1),
            getActivationFunction(self.act),
            ResNetBlock(8, 8, 3, self.act),
            nn.Conv2d(8, in_channels*out_channels, 3, stride=1, padding=1),
            getActivationFunction(self.act)
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
  
class MetaConvSmoother(nn.Module):
    def __init__(self, act, mL=8, kernelSize=3):
        super(MetaConvSmoother, self).__init__()
        self.mL = mL
        self.kernelSize = kernelSize
        self.fc1 = nn.Sequential(nn.Linear(9, 100), 
                                 getActivationFunction(act),
                                 nn.Linear(100, mL*kernelSize*kernelSize))
        self.fc2 = nn.Sequential(nn.Linear(9, 100), 
                                 getActivationFunction(act),
                                 nn.Linear(100, mL*kernelSize*kernelSize))

    def forward(self, kernelA):
        batch_size= kernelA.shape[0]
        
        weight = torch.flatten(kernelA, 1)
        weight1 = self.fc1(weight).view([batch_size,self.mL,1,self.kernelSize,self.kernelSize])
        weight2 = self.fc2(weight).view([batch_size,1,self.mL,self.kernelSize,self.kernelSize])
        return weight1, weight2
    
#*********************************************
#*       FNS based on spectral transform     #
#*********************************************
class FNS(nn.Module):
    def __init__(self):
        super(FNS, self).__init__()

    def forward(self, r, weights):
        N1 = r.shape[-1]
        rsym = self.expand(r)
        weights1, weights2, weights3, weights_theta = weights

        r_hat = torch.fft.ifft2(rsym, dim=(2, 3))
        r_hat = torch.fft.fftshift(r_hat, dim=(2, 3))
        r_hat = self.transition(r_hat, [weights1, weights2, weights3])
        out_ft = r_hat*weights_theta
        out_ft = self.transition(out_ft, [torch.transpose(weights3, -4, -3).transpose(-2, -1).conj(), torch.transpose(weights2, -4, -3).transpose(-2, -1).conj(), torch.transpose(weights1, -4, -3).transpose(-2, -1).conj()])
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
        if self.smoother == "cnn":
            self.meta_smoother = MetaConvSmoother(config["act"], mL=8, kernelSize=config["kernel_size"])

        self.meta1 = Meta_T(1, 4, config["act"])
        self.meta2 = Meta_T(4, 4, config["act"])
        self.meta3 = Meta_T(4, 1, config["act"])
        self.Meta_H_type = config["Meta_Type"]

        if self.Meta_H_type == "sFNO":
            self.meta = sFNO_epsilon_v2(config, 3, 2)
        elif self.Meta_H_type == "UNet":
            self.meta = MetaUNet(config)
        self.H = FNS()

        self.dir = config["prediction_folder"]
        self.error_threshold = config["error_threshold"]
        self.max_iter_num = config["max_iter_num"]
        self.xavier_init = config["xavier_init"]
        if self.xavier_init > 0:
            self._reset_parameters()


    def forward(self, f, kernelA, epoch):
        K = (epoch-1) // 40 + 1
        if K > 10:
            K = 10
        A = Advection(kernelA)
        weights = self.setup(kernelA, f.shape[-1])
        x = torch.zeros_like(f, dtype=f.dtype, device=f.device)
        for i in range(K):
            x = A.fns(x, f, self.H, weights, self.smoother)
            r = f - A(x)
        loss = torch.norm(r) / torch.norm(f)
        return loss

    
    def setup(self, kernelA, N):

        weights1 = self.meta1(kernelA.float())
        weights2 = self.meta2(kernelA.float())
        weights3 = self.meta3(kernelA.float())
        weights = [weights1, weights2, weights3]

        if self.smoother == "jacobi":
            lfa = self.ComputeSmootherFactor(kernelA, N, 1/2)
            LFA = self.ComputeSmootherFactor(kernelA, 2*(N+1), 1/2)
            theta = self.meta(lfa.float(), 2*(N+1)) * LFA
            weiths_theta = theta[:, 0:1, :, :] + 1j*theta[:, 1:2, :, :]
            weights.append(weiths_theta.cfloat())
        elif self.smoother == "cnn":
            lfa = F.interpolate(kernelA, size=(N, N), mode='bilinear', align_corners=True)
            theta = self.meta(lfa.float(), 2*(N+1))
            weiths_theta = theta[:, 0:1, :, :] + 1j*theta[:, 1:2, :, :]
            weights.append(weiths_theta.cfloat())
            Sweight1, Sweight2 = self.meta_smoother(kernelA.float())
            weights.append(Sweight1)
            weights.append(Sweight2)
        return weights
    
    
    def ComputeSmootherFactor(self, KernelA, N, w):
        device = KernelA.device
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
            taus = [w/KernelA[i, 0, 1, 1]]*10
            y = 1
            for j in range(len(taus)):
                y *= 1-taus[j] * (k1*torch.exp(-theta1)*torch.exp(-theta2)+k2*torch.exp(-theta2)+k3*torch.exp(theta1)*torch.exp(-theta2)+k4*torch.exp(-theta1)+k5+k6*torch.exp(theta1)+k7*torch.exp(-theta1)*torch.exp(theta2)+k8*torch.exp(theta2)+k9*torch.exp(theta1)*torch.exp(theta2))
            Y[i, 0, :, :] = torch.abs(y)
        return Y
    
   
    def test(self, f, kernelA, epoch):
        res = 1
        i = 1
        N = f.shape[-1]
        weights = self.setup(kernelA, N)
        A = Advection(kernelA)
        normf = torch.norm(f, p=2, dim=(2, 3))
        res = normf / normf
        ress = [res]
        x1 = self.H(f.float(), weights[:4])
        plt.imshow(np.flipud(x1[0,0].cpu().detach().numpy()), cmap="jet")
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
        im = ax1.imshow(np.abs(weights[3][0,0].cpu().detach().numpy()), cmap="jet")
        ax1.set_title('lambda_inv', fontsize=20)
        cbar = fig.colorbar(im, ax=ax1) 
        im = ax2.imshow(np.flipud(x[0,0].cpu().detach().numpy()), cmap="jet", extent=[0,1,0,1])
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
        return x, ress

     
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

        if self.smoother == "cnn":
            for param in self.meta_smoother.parameters():
                if param.ndim > 1:
                    xavier_uniform_(param, gain=self.xavier_init)
                else:
                    constant_(param, 1e-2)

 