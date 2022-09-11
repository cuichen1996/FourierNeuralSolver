import numpy as np
import os

from torch import nn


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


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
