# %%
import numpy as np
import torch

from data import KappaDataGenerator

# %%
N = 128
f = 10
gamma_value = 0.05
omega = 2 * torch.pi * f
h = 1 / (N - 1)

dataset = KappaDataGenerator(N, N)
dataset.load_data(data="cifar10")
kappa = dataset.generate_kappa().to(torch.float32)

data_kappa = np.zeros((10, 1, N, N))
for i in range(10):
    data_kappa[i, 0] = dataset.generate_kappa().to(torch.float32).reshape(N, N)

np.save(f"kappa{N}", data_kappa)


src = [N // 2, N // 2]
b = np.zeros((1, 1, N, N), dtype=np.complex64)
b[0, 0, src[0], src[1]] = 1.0 / h**2


np.save(f"b{N}", b)

# %%
N = 256
f = 20
omega = 2 * torch.pi * f
h = 1 / (N - 1)

dataset = KappaDataGenerator(N, N)
dataset.load_data(data="cifar10")
kappa = dataset.generate_kappa().to(torch.float32)

data_kappa = np.zeros((10, 1, N, N))
for i in range(10):
    data_kappa[i, 0] = dataset.generate_kappa().to(torch.float32).reshape(N, N)
np.save(f"kappa{N}", data_kappa)

src = [N // 2, N // 2]
b = np.zeros((1, 1, N, N), dtype=np.complex64)
b[0, 0, src[0], src[1]] = 1.0 / h**2

np.save(f"b{N}", b)


# %%
N = 512
f = 40
omega = 2 * torch.pi * f
h = 1 / (N - 1)

dataset = KappaDataGenerator(N, N)
dataset.load_data(data="cifar10")
kappa = dataset.generate_kappa().to(torch.float32)

data_kappa = np.zeros((10, 1, N, N))
for i in range(10):
    data_kappa[i, 0] = dataset.generate_kappa().to(torch.float32).reshape(N, N)
np.save(f"kappa{N}", data_kappa)

src = [N // 2, N // 2]
b = np.zeros((1, 1, N, N), dtype=np.complex64)
b[0, 0, src[0], src[1]] = 1.0 / h**2

np.save(f"b{N}", b)
