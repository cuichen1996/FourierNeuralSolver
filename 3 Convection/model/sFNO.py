import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class fourier_conv_2d(nn.Module):
    def __init__(self, in_, out_, wavenumber1, wavenumber2):
        super(fourier_conv_2d, self).__init__()
        self.out_ = out_
        self.wavenumber1 = wavenumber1
        self.wavenumber2 = wavenumber2
        scale = (1 / (in_ * out_))
        self.weights1 = scale * torch.rand(in_, out_, wavenumber1, wavenumber2, dtype=torch.cfloat)
        self.weights2 = scale * torch.rand(in_, out_, wavenumber1, wavenumber2, dtype=torch.cfloat)
        self.weights1 = nn.Parameter(torch.view_as_real(self.weights1))
        self.weights2 = nn.Parameter(torch.view_as_real(self.weights2))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        weights1, weights2 = map(torch.view_as_complex, (self.weights1, self.weights2))
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.wavenumber1, :self.wavenumber2] = \
            self.compl_mul2d(x_ft[:, :, :self.wavenumber1, :self.wavenumber2], weights1)
        out_ft[:, :, -self.wavenumber1:, :self.wavenumber2] = \
            self.compl_mul2d(x_ft[:, :, -self.wavenumber1:, :self.wavenumber2], weights2)
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def get_grid2D(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x, endpoint=True), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y, endpoint=True), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


class IO_layer(nn.Module):
    def __init__(self,  features_, 
                        wavenumber, 
                        drop = 0.):
        super().__init__()
        self.W =  nn.Conv2d(features_, features_, 1)
        self.IO = fourier_conv_2d(features_, features_,*wavenumber)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.IO(x)+self.W(x)
        x = self.dropout(x) 
        x = self.act(x)
        return x

class IO_ResNetblock(nn.Module):
    def __init__(self,  features_, 
                      wavenumber, 
                      drop_path = 0., 
                      drop = 0.):
        super().__init__()
        self.IO = IO_layer(features_, wavenumber, drop)
        self.pwconv1 = nn.Conv2d(features_, 4* features_, 1) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 =nn.Conv2d(4 * features_, features_,1)
        self.norm1 = LayerNorm(features_, eps=1e-5,  data_format = "channels_first")
        self.norm2 = LayerNorm(features_, eps=1e-5,  data_format = "channels_first")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        skip = x
        x = self.norm1(x)
        x = self.IO(x)
        x = skip+self.drop_path(x) #NonLocal Layers
        skip = x 
        x = self.norm2(x)
        #local
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = skip + self.drop_path(x)
        return x  

class sFNO_epsilon_v2(nn.Module): 
    def __init__(self, config, in_channel, out_channel):
        super().__init__()
        self.padding = config["padding"]

        self.lifting_layers = nn.ModuleList()
        steam =  nn.Conv2d(in_channel, config["dims"][0], 1, 1)
        self.lifting_layers.append(steam)
        for i in range(len(config["dims"])-1):
            lifting_layers = nn.Sequential(
                                            LayerNorm(config["dims"][i], eps=1e-6, data_format= "channels_first"),
                                            nn.Conv2d(config["dims"][i],config["dims"][i+1], kernel_size = 1, stride = 1)
                                        )
            self.lifting_layers.append(lifting_layers)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, config["drop_path_rate"], sum(config["depths"]))]
        cur = 0
        for i in range(len(config["dims"])):
                    stage = nn.Sequential(
                        *[IO_ResNetblock(features_=config["dims"][i], 
                                    wavenumber=[config["modes"][i]]*2, 
                                    drop_path=dp_rates[cur + j],
                                    drop=config["drop"]) for j in range(config["depths"][i])]
                    )
                    self.stages.append(stage)
                    cur += config["depths"][i]

        self.head = nn.Conv2d(config["dims"][-1],out_channel,1,1)
        
    def forward_features(self, x, N):
        x = self.lifting_layers[0](x)
        x = F.interpolate(x, size=(N, N), mode='bilinear', align_corners=True)
        x = F.pad(x, [0,self.padding, 0, self.padding])
        for i in range(1,len(self.lifting_layers)):
            x = self.lifting_layers[i](x)
            x = self.stages[i](x)
        x = x[..., :-self.padding, :-self.padding] 
        return x

    def forward(self, x, N):
        x = self.forward_features(x, N)
        x = self.head(x)
        return x