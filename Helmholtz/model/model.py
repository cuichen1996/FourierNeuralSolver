# -*- coding: utf-8 -*-
# @Author: Chen Cui
# @Date:   2022-05-24 12:27:50
# @Last Modified by:   Your name
# @Last Modified time: 2022-09-12 00:52:03

# %%
from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_


# %%
def getActivationFunction(
    act_function_name: str, features=None, end=False
) -> nn.Module:
    """Returns the activation function module given
    the name

    Args:
        act_function_name (str): Name of the activation function, case unsensitive

    Raises:
        NotImplementedError: Raised if the activation function is unknown

    Returns:
        nn.Module
    """
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


class OutConv(nn.Module):
    """Outconvolution, consisting of a simple 2D convolution layer with kernel size 1"""

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => actFunction) * 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels=None,
        activation_fun="relu",
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            getActivationFunction(activation_fun, mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_features: int,
        state_size=2,
        activation_function="prelu",
        use_state=False,
        domain_size=0,
    ):
        super().__init__()
        self.state_size = state_size
        self.use_state = use_state
        self.domain_size = domain_size
        self.num_features = num_features

        # Define the two double_conv layers
        self.conv_signal = DoubleConv(
            self.num_features + self.state_size * self.use_state,
            self.num_features,
            activation_fun=activation_function,
        )

        #  Downward path
        self.down = nn.Conv2d(
            self.num_features, self.num_features, kernel_size=8, padding=3, stride=2
        )
        if self.use_state:
            self.conv_state = DoubleConv(
                self.num_features + self.state_size,
                self.state_size,
                activation_fun=activation_function,
            )

        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def clear_state(self, x):
        self.state = torch.zeros(
            [x.shape[0], 2, self.domain_size, self.domain_size], device=x.device
        )

    def forward(self, x):
        # self.clear_state(x)
        if self.use_state:
            if self.state is None:
                raise ValueError(
                    "You must set or clear the state before using this module"
                )
            x_and_state = torch.cat([x, self.state], 1)
            output = self.conv_signal(x_and_state)
            self.state = self.conv_state(torch.cat([output, self.state], 1))
            # self.state = self.conv_state(output, self.state)
        else:
            output = self.conv_signal(x)
        return output, self.down(output)


class HybridNet(nn.Module):
    def __init__(
        self,
        activation_function: str,
        depth: int,   # Number of encoder levels
        domain_size: int, # discretization size
        features: int,  # Number of output channels of DoubleConv and input channels of EncoderBlock
        inchannels: int, # Number of input channels of DoubleConv
        state_channels: int,
        state_depth: int,
    ):
        super().__init__()
        # Hyperparameters
        self.activation_function = activation_function
        self.depth = depth
        self.domain_size = domain_size
        self.features = features
        self.inchannels = inchannels
        self.state_channels = state_channels
        self.state_depth = state_depth

        #  Define states boundaries for packing and unpacking
        self.init_by_size()

        # Input layer
        self.inc = DoubleConv(
            self.inchannels, self.features, activation_fun=self.activation_function
        )

        # Encoding layer
        self.enc = nn.ModuleList(
            [
                EncoderBlock(
                    self.features,
                    state_size=self.state_channels,
                    activation_function=self.activation_function,
                    use_state=False,  # Use state only
                    domain_size=self.states_dimension[d],
                )
                for d in range(self.depth)
            ]
        )

        # Decode path
        self.decode = nn.ModuleList(
            [
                DoubleConv(
                    self.features + self.features * (i < self.depth),
                    self.features,
                    activation_fun=self.activation_function,
                )
                for i in range(self.depth + 1)
            ]
        )

        # Upsampling
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    self.features,
                    self.features,
                    kernel_size=8,
                    padding=3,
                    output_padding=0,
                    stride=2,
                )
                for i in range(self.depth)
            ]
        )

        # Output layer
        self.outc = OutConv(self.features, 2)

    def init_by_size(self):
        # This helps to reshape the state to the correct dimensions
        self.states_dimension = [self.domain_size // 2 ** x for x in range(self.depth)]
        self.total_state_length = sum(map(lambda x: x ** 2, self.states_dimension))
        self.state_boundaries = []
        for d in range(self.depth):
            if d == 0:
                self.state_boundaries.append([0, self.states_dimension[d] ** 2])
            else:
                self.state_boundaries.append(
                    [
                        self.state_boundaries[-1][-1],
                        self.state_boundaries[-1][-1] + self.states_dimension[d] ** 2,
                    ]
                )

    def forward(self, x):

        # First feature transformation
        x = self.inc(x)

        # Downsampling tree and extracting new states
        inner_signals = []
        for d in range(self.depth):
            # Encode signal
            inner, x = self.enc[d](x)
            # Store signal
            inner_signals.append(inner)

        # Upscaling
        x = self.decode[-1](x)
        for d in range(self.depth - 1, -1, -1):
            # Upscale
            x = self.up[d](x)
            # Concatenate inner path
            x = torch.cat([x, inner_signals[d]], 1)
            # Decode
            x = self.decode[d](x)

        # Output layer
        out = self.outc(x)

        return out[:,:1,:,:], out[:,1:,:,:]

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

##* smoother
class ChebySemi(nn.Module):
    def __init__(self, niters=64, factor=3.0):
        super(ChebySemi, self).__init__()
        self.niters = niters
        self.factor = factor
        self.roots = [np.cos((np.pi*(2*i+1)) / (2*self.niters)) for i in range(self.niters)]
        # self.good_term = [i for i in range(self.niters)]

    def forward(self, x, f, kernelA):
        # with torch.no_grad():
        #     u = torch.randn_like(f).double()
        #     for i in range(20):
        #         y = BatchConv2d(u, kernelA)      
        #         m = torch.max(torch.max(torch.abs(y),dim=2).values, dim=2).values.reshape(x.shape[0], 1, 1, 1)
        #         u = y / m
        #     taus = [2 / (m + m/self.factor - (m/self.factor - m) * r) for r in self.roots]

        for k in range(1):
            Ax = BatchConv2d(x, kernelA) 
            # x = x + taus[k]*(f - Ax)
            x = x + 1/6*(f - Ax)
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
        mu = r.shape[2]
        weights1 = self.fc1(kernelA.view(batchSize, 9)).view(
            batchSize, self.mL, 1, self.kernelSize, self.kernelSize)
        weights2 = self.fc2(kernelA.view(batchSize, 9)).view(
            batchSize, self.mL, self.mL, self.kernelSize, self.kernelSize)

        G1 = torch.zeros(batchSize, self.mL, mu, mu,
                         dtype=r.dtype, device=r.device)
        G2 = torch.zeros(batchSize, self.mL, mu, mu,
                         dtype=r.dtype, device=r.device)

        for i in range(batchSize):
            G1[i] = F.conv2d(r[i:i+1], weights1[i], padding=self.kernelSize//2)
        for i in range(batchSize):
            G2[i] = F.conv2d(G1[i:i+1], weights2[i],
                             padding=self.kernelSize//2)
        G = torch.cat((r, G1, G2), dim=1)
        S = BatchConv2d_multichannel(G, kernelA).view(-1, self.L, mu*mu)
        lr = r.view(-1, mu*mu, 1)
        M = torch.matmul(S, S.permute(0, 2, 1))
        b = torch.matmul(S, lr)
        K = torch.linalg.solve(M, b)
        x = x + torch.matmul(K.permute((0, 2, 1)),
                             G.view(-1, self.L, mu*mu)).view(-1, 1, mu, mu)
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
        Ax = BatchConv2d(x, kernelA) 
        r = f - Ax
        for i in range(batch_size):
            tmp = F.conv2d(r[i:i+1], weight1[i], padding=self.kernelSize//2)
            G2[i] = F.conv2d(tmp, weight2[i], padding=self.kernelSize//2)
        x = x + G2
        return x

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
        # out_ft = torch.fft.fftshift(out_ft)
        # plt.pcolor(np.abs(out_ft[0,0,:,:].detach().cpu().numpy()))
        # plt.colorbar()
        # plt.savefig('/home/kaijiang/cuichen/FNSsummer/0708/expriments/Richardson_10/prediction/out_ft-6{}.png'.format(i))
        # plt.close()
        # out_ft = torch.fft.ifftshift(out_ft)
        e = torch.fft.irfft2(out_ft, dim=(2, 3), s=(r.size(-2), r.size(-1)), norm='ortho')  
        # e2 = torch.imag(torch.fft.ifft2(out_ft, norm='ortho'))
        return e
        
class _ConvNet_(nn.Module):
    """
    Huang et.al.
    """
    def __init__(self,k,kernel_size,initial_kernel):
        super(_ConvNet_, self).__init__()

        self.convLayers1 = nn.ModuleList([nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False).double()
                                          for _ in range(5)])
        self.convLayers2 = nn.ModuleList([nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False).double()
                                         for _ in range(2)])                         
        initial_weights = torch.zeros(1,1,kernel_size,kernel_size).double()
        initial_weights[0,0,kernel_size//2,kernel_size//2] = initial_kernel

        for net in self.convLayers1:
            net.weight = nn.Parameter(initial_weights)
        for net in self.convLayers2:
            net.weight = nn.Parameter(initial_weights)

    def forward(self, x):
        y1 = x
        y2 = x
        for net in self.convLayers1:
            y1 = torch.tanh(net(y1))

        for net in self.convLayers2:
            y2 = torch.tanh(net(y2))
        
        return y1+2/3*y2

#%%        
#** Hypernet **#
class MetaGFNet(nn.Module):
    def __init__(self, mid_chanel, kernelSize, N, act, modes, softshrink):
        super(MetaGFNet, self).__init__()
        self.softshrink = softshrink
        self.modes = modes
        self.up = nn.Upsample(size=(N,N), mode='nearest')
        self.meta = nn.Sequential(
            # nn.Upsample(size=(N,N), mode='nearest'),
            nn.Conv2d(1, mid_chanel, kernelSize, padding=kernelSize//2),
            getActivationFunction(act),
            nn.Conv2d(mid_chanel, mid_chanel, kernelSize, padding=kernelSize//2),
            getActivationFunction(act),
            nn.Conv2d(mid_chanel, mid_chanel, kernelSize, padding=kernelSize//2),
            getActivationFunction(act),
            nn.Conv2d(mid_chanel, 2, kernelSize, padding=kernelSize//2),
            nn.AdaptiveAvgPool2d((N,N//2+1))).double()

    def forward(self, u, r, kernelA):
        
        batch_size, N = r.shape[0], r.shape[2]

        # A = self.up(kernelA)
        # inputs = torch.cat((u, r, A), dim=1)

        weights1 = self.meta(r).view(batch_size, 1, N, N//2+1, 2)
        # weights1 = F.softshrink(weights1, lambd=self.softshrink) if self.softshrink else weights1
        weights1 = torch.view_as_complex(weights1)

        r_hat = torch.fft.rfft2(r, dim=(2, 3), norm='ortho')
        out_ft = torch.zeros(batch_size, 1,  N, N//2 + 1, dtype=torch.cdouble, device=r.device)
        
        out_ft[:, :, :self.modes[0], :self.modes[1]] = r_hat[:, :, :self.modes[0], :self.modes[1]] * weights1[:, :, :self.modes[0], :self.modes[1]]
        out_ft[:, :, -self.modes[0]:, :self.modes[1]] = r_hat[:, :, -self.modes[0]:, :self.modes[1]] * weights1[:, :, -self.modes[0]:, :self.modes[1]]
        out_ft = torch.where(abs(out_ft) > self.softshrink, out_ft, 0+0j)

        e = torch.fft.irfft2(out_ft, dim=(2, 3), s=(r.size(-2), r.size(-1)), norm='ortho')  
        return e

#%%
class HyperFNS(nn.Module):
    def __init__(self, config):
        super(HyperFNS, self).__init__()
        
        # self.smoother = MetaConvSmoother(config["act"], config["mL"],config["kernel_size"])
        self.smoother = MetaScSmoother()
        self.H = CNNFNS(config["mid_chanel"], config["act"]).double() 
        
        self.K = config["K"]
        self.error_threshold = config["error_threshold"]
        self.max_iter_num = config["max_iter_num"]

        self.xavier_init = config["xavier_init"]
        if self.xavier_init > 0:
            self._reset_parameters()

    def forward(self, x, f, kernelA, r_old):
        # loss = 0
        # r = f - BatchConv2d(x, kernelA)
        # h = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        # r_old = f - BatchConv2d(x, kernelA)
        # for i in range(self.K):
            # inputs = torch.cat([x, h, r], 1)
            # deltax, h = self.smoother(inputs)
            # x = x + deltax
        x = self.smoother(x, f, kernelA)
        # compute residual
        r = f - BatchConv2d(x, kernelA)
        e = self.H(r, kernelA)
        x = x + e
        # compute loss
        r = f - BatchConv2d(x, kernelA)
        res = torch.norm(r) / torch.norm(r_old)
            # loss += res
        return res, x

    def test(self, x, f, kernelA):
        res = 1
        i = 1
        # r = f - BatchConv2d(x, kernelA)
        # h = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        residual = [res]
        while res > self.error_threshold and i < self.max_iter_num:
            # inputs = torch.cat([x, h, r], 1)
            # deltax, h = self.smoother(inputs)
            # x = x + deltax
            x = self.smoother(x, f, kernelA)
            # compute residual
            r = f - BatchConv2d(x, kernelA)
            e = self.H(r, kernelA)
            x = x + e
            # compute loss
            r = f - BatchConv2d(x, kernelA)
            res = torch.norm(r)**2 / torch.norm(f)**2
            i = i + 1
            residual.append(res.item())
            print(f"res:{res.item()}")
            # print("k:", k)
        residual = np.asarray(residual)
        np.save("/home/kaijiang/cuichen/FNSsummer/0909/data/FNS125.npy",residual)
        return x, i

    def _reset_parameters(self):
        for param in self.H.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
            else:
                constant_(param, 1e-2)       
                
        for param in self.smoother.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
            else:
                constant_(param, 1e-1)                  
