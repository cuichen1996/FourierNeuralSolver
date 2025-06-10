import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def flatten(x):
    N = x.shape[0]
    return x.reshape(N, -1)


def relative_error(truth, pred):
    x = flatten(truth)
    y = flatten(pred)
    error = np.linalg.norm(x-y) / np.linalg.norm(x)
    return error


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight)

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples, -
                                                           1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def leb_shuffle_2n(n):
    if n == 1:
        return np.array([0], dtype=int)
    else:
        prev = leb_shuffle_2n(n // 2)
        ans = np.zeros(n, dtype=int)
        ans[::2] = prev
        ans[1::2] = n - 1 - prev
        return ans

def get_grid2D(shape, device):
    batchsize, size_x, size_y = shape[0], shape[2], shape[3]
    x = np.linspace(0, 1, size_y, endpoint=True, dtype=np.float32)  #! 1,3
    y = np.linspace(0, 1, size_x, endpoint=True, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    gridx = torch.from_numpy(X).repeat([batchsize, 1, 1, 1])
    gridy = torch.from_numpy(Y).repeat([batchsize, 1, 1, 1])
    return gridx.to(device), gridy.to(device)

def min_max_normalize(tensor):
    B, C, N, _ = tensor.shape
    x_flat = tensor.view(B, C, -1)
    min_val = x_flat.min(dim=2, keepdim=True).values
    max_val = x_flat.max(dim=2, keepdim=True).values
    normalized_tensor_flat = (x_flat - min_val) / (max_val - min_val + 1e-6)  # 加入小值以避免除零
    normalized_tensor = normalized_tensor_flat.view(B, C, N, N)
    return normalized_tensor

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


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
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
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
