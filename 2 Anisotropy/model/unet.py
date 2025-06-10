import torch
import torch.nn as nn
import torch.nn.functional as F

def getActivationFunction(
    act_function_name: str, features=None, end=False
):
    if act_function_name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif act_function_name.lower() == "celu":
        return nn.CELU(inplace=True)
    elif act_function_name.lower() == "relu_batchnorm":
        if end:
            return nn.ReLU(inplace=True)
        else:
            return nn.Sequential(nn.ReLU(inplace=True), nn.BatchNorm2d(features))
    elif act_function_name.lower() == "tanh":
        return nn.Tanh()
    elif act_function_name.lower() == "elu":
        return nn.ELU()
    elif act_function_name.lower() == "prelu":
        return nn.PReLU()
    elif act_function_name.lower() == "gelu":
        return nn.GELU()
    elif act_function_name.lower() == "tanhshrink":
        return nn.Tanhshrink()
    elif act_function_name.lower() == "softplus":
        return nn.Softplus()
    elif act_function_name.lower() == "mish":
        return nn.Mish()
    elif act_function_name.lower() == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    else:
        err = "Unknown activation function {}".format(act_function_name)
        raise NotImplementedError(err)

class ConvDown(nn.Module):
    """_summary_
    通道数翻倍, 尺寸减少一半
    """
    def __init__(self, in_channels, out_channels, act):
        super().__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=5, padding=1),
            nn.BatchNorm2d(out_channels),
            getActivationFunction(act)
        )
        
    def forward(self, x):
        return self.Down(x)

# %%
class UNetUpBlock(nn.Module):
    """_summary_
    通道数减半(skip-connection 会使得通道数恢复), 尺寸翻倍,
    """
    def __init__(self, in_channels, out_channels, act):
        super().__init__()

        self.up = nn.Sequential(
            getActivationFunction(act),
            nn.ConvTranspose2d(in_channels, out_channels, 5, 2, 1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

# %%
class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels, act):
        super().__init__()

        self.up = nn.Sequential(
            getActivationFunction(act),
            nn.ConvTranspose2d(in_channels, out_channels, 5, 2, 1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x = self.up(x)
        return x

# %%
class ResNetBlock(nn.Module):
    """_summary_
    不改变通道、尺寸
    """
    def __init__(self, in_channels, out_channels, kernel_size, act):
        super(ResNetBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            getActivationFunction(act),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        )
    
        self.activation = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            getActivationFunction(act)
        )
     
    def forward(self, x):
        out = self.layers(x) + x
        out = self.activation(out)
        return out

# %%
class UNet(nn.Module):
    def __init__(self, latent_channels, kernel_size, act):
        super(UNet, self).__init__()
        self.act = act
        self.kernel_size= kernel_size
        
        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(1,4,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(4,4,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(4,8,3,stride=2,padding=1), 
            getActivationFunction(act),     
            nn.ConvTranspose2d(8,8,3,stride=2,padding=1))
        
        self.conv_down_blocks = nn.ModuleList()
        self.conv_down_blocks.append(ConvDown(8, 16, self.act))
        self.conv_down_blocks.append(ConvDown(16, 32, self.act))
        self.conv_down_blocks.append(ConvDown(32, 64, self.act))
        self.conv_down_blocks.append(ConvDown(64, 128, self.act))

        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(ResNetBlock(16,16, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(16,16, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(32,32, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(32,32, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(64,64, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(64,64, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(128,128, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(128,128, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(128,128, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(128,128, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(128,128, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(16,16, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(32,32, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(64,64, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(128,128, self.kernel_size, self.act))

        self.up_blocks = nn.ModuleList()
        self.up_blocks.append(UNetUpBlock(128, 64, self.act))
        self.up_blocks.append(UNetUpBlock(128, 32, self.act))
        self.up_blocks.append(UNetUpBlock(64, 16, self.act))
        self.up_blocks.append(nn.Sequential(
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            getActivationFunction(self.act)
        ))
        self.up_blocks.append(ConvUp(16, latent_channels, self.act))


    def forward(self, x, N):
        x = self.encoder(x)
        x = F.interpolate(x, N)
        op = self.conv_blocks[1](self.conv_blocks[0](self.conv_down_blocks[0](x)))
        x1 = self.conv_blocks[3](self.conv_blocks[2](self.conv_down_blocks[1](op)))
        x2 = self.conv_blocks[5](self.conv_blocks[4](self.conv_down_blocks[2](x1)))
        x3 = self.conv_blocks[7](self.conv_blocks[6](self.conv_down_blocks[3](x2)))

        up_x3 = self.conv_blocks[8](x3)
        up_x3 = self.conv_blocks[9](up_x3)
        up_x3 = self.conv_blocks[10](up_x3)

        # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
        up_x1 = self.conv_blocks[14](self.up_blocks[0](up_x3, x2))
        # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
        up_x2 = self.conv_blocks[13](self.up_blocks[1](up_x1, x1))
        # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
        up_x4 = self.conv_blocks[12](self.up_blocks[2](up_x2, op))
        x = self.up_blocks[3](up_x4)
        x = self.up_blocks[4](x)
        l1 = torch.fft.fft2(x)
        return l1
    
class MetaUNet(nn.Module):
    def __init__(self, kernel_size, channels, act):
        super(MetaUNet, self).__init__()
        self.act = act
        self.kernel_size= kernel_size
        self.channels = channels
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(9, 64),
        #     getActivationFunction(act),
        #     nn.Linear(64, 128),
        #     getActivationFunction(act),
        #     nn.Linear(128, 256))

        # self.lift = nn.Conv2d(1, channels[0], 1)
        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(1,4,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(4,4,3,stride=2,padding=1),
            getActivationFunction(act),
            nn.ConvTranspose2d(4,channels[0],3,stride=2,padding=1), 
            getActivationFunction(act),     
            nn.ConvTranspose2d(channels[0],channels[0],3,stride=2,padding=1))

        self.conv_down_blocks = nn.ModuleList()
        self.conv_down_blocks.append(ConvDown(channels[0], channels[1], self.act))
        self.conv_down_blocks.append(ConvDown(channels[1], channels[2], self.act))
        self.conv_down_blocks.append(ConvDown(channels[2], channels[3], self.act))

        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(ResNetBlock(channels[0], channels[0], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(channels[1], channels[1], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(channels[2], channels[2], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(channels[3], channels[3], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(channels[3], channels[3], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(channels[3], channels[3], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(channels[4], channels[4], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(channels[5], channels[5], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(channels[6], channels[6], self.kernel_size, self.act))

        self.up_blocks = nn.ModuleList()
        self.up_blocks.append(UNetUpBlock(channels[3], channels[2], self.act))
        self.up_blocks.append(UNetUpBlock(channels[4], channels[1], self.act))
        self.up_blocks.append(UNetUpBlock(channels[5], channels[0], self.act))

        self.out = nn.Sequential(
            getActivationFunction(self.act),
            nn.Conv2d(channels[6], 1, 1, 1, 0)
            # nn.ReLU()
        )

    def forward(self, x, N):
        # x = self.encoder(x.flatten(1))
        # x = x.view(-1, 1, 16, 16)
        # x = self.lift(x)
        x = self.encoder(x)
        x = F.interpolate(x, N+2)
        x1 = self.conv_blocks[0](x) 
        x2 = self.conv_blocks[1]((self.conv_down_blocks[0](x1))) # 16 N//2
        x3 = self.conv_blocks[2]((self.conv_down_blocks[1](x2))) # 32 N//4
        x4 = self.conv_blocks[3]((self.conv_down_blocks[2](x3))) # 64 N//8
        x4 = self.conv_blocks[4](x4)
        x4 = self.conv_blocks[5](x4)

        x5 = self.conv_blocks[6](self.up_blocks[0](x4, x3)) # 64 N//4
        x6 = self.conv_blocks[7](self.up_blocks[1](x5, x2)) # 32 N//2
        x7 = self.conv_blocks[8](self.up_blocks[2](x6, x1)) # 16 N
        x = self.out(x7)
        weights = torch.fft.fft2(x, dim=(2, 3)) / self.channels[-1]**2
        return torch.fft.fftshift(weights)
        # return x
 
class MetaUNetv2(nn.Module):
    def __init__(self, config):
        super(MetaUNetv2, self).__init__()
        self.act = config["act"]
        self.kernel_size= 3
        self.channels = config["channels"]
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1,4,3,stride=2,padding=1),
            getActivationFunction(self.act),
            nn.Conv2d(4,4,3,stride=2,padding=1),
            getActivationFunction(self.act),
            nn.Conv2d(4,self.channels[0],3,stride=2,padding=1), 
            getActivationFunction(self.act),     
            nn.Conv2d(self.channels[0],self.channels[0],3,stride=2,padding=1)
            )
        
        self.conv_down_blocks = nn.ModuleList()
        self.conv_down_blocks.append(ConvDown(self.channels[0], self.channels[1], self.act))
        self.conv_down_blocks.append(ConvDown(self.channels[1], self.channels[2], self.act))
        self.conv_down_blocks.append(ConvDown(self.channels[2], self.channels[3], self.act))

        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(ResNetBlock(self.channels[0], self.channels[0], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(self.channels[1], self.channels[1], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(self.channels[2], self.channels[2], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(self.channels[3], self.channels[3], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(self.channels[3], self.channels[3], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(self.channels[3], self.channels[3], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(self.channels[4], self.channels[4], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(self.channels[5], self.channels[5], self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(self.channels[6], self.channels[6], self.kernel_size, self.act))

        self.up_blocks = nn.ModuleList()
        self.up_blocks.append(UNetUpBlock(self.channels[3], self.channels[2], self.act))
        self.up_blocks.append(UNetUpBlock(self.channels[4], self.channels[1], self.act))
        self.up_blocks.append(UNetUpBlock(self.channels[5], self.channels[0], self.act))

        self.out = nn.Sequential(
            getActivationFunction(self.act),
            nn.Conv2d(self.channels[6], 2, 1, 1, 0)
        )

    def forward(self, x, N):
        x = self.encoder(x)
        x = F.interpolate(x, size=(N, N), mode='bilinear', align_corners=True)
        x1 = self.conv_blocks[0](x) 
        x2 = self.conv_blocks[1]((self.conv_down_blocks[0](x1))) # 16 N//2
        x3 = self.conv_blocks[2]((self.conv_down_blocks[1](x2))) # 32 N//4
        x4 = self.conv_blocks[3]((self.conv_down_blocks[2](x3))) # 64 N//8
        x4 = self.conv_blocks[4](x4)
        x4 = self.conv_blocks[5](x4)

        x5 = self.conv_blocks[6](self.up_blocks[0](x4, x3)) # 64 N//4
        x6 = self.conv_blocks[7](self.up_blocks[1](x5, x2)) # 32 N//2
        x7 = self.conv_blocks[8](self.up_blocks[2](x6, x1)) # 16 N
        x = self.out(x7)
        # weights = torch.fft.fft2(x, dim=(2, 3)) / self.channels[-1]**2
        # return torch.fft.fftshift(weights, dim=(2, 3))
        return x