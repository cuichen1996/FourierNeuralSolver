# %%
import torch
import numpy as np
from pyamg.gallery import stencil_grid
import scipy.sparse.linalg as spla

torch.set_default_dtype(torch.float64)

# %%
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# %%

# %%
Kernelxx = torch.tensor([[[[-1./6., 1./3., -1./6.], [-2./3., 4./3., -2./3.], [-1./6., 1./3., -1./6.]]]])  # \partial_{xx}
Kernelxy = torch.tensor([[[[-1./4., 0., 1./4.], [0., 0., 0.], [1./4., 0., -1./4.]]]])   # \partial_{xy}
Kernelyx = torch.tensor([[[[-1./4., 0., 1./4.], [0., 0., 0.], [1./4., 0., -1./4.]]]])    # \partial_{yx}
Kernelyy = torch.tensor([[[[-1./6., -2./3., -1./6.], [1./3., 4./3., 1./3.], [-1./6., -2./3., -1./6.]]]])    # \partial_{yx}

# %%
def gen_stencil(epsilon, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    c2 = c*c
    cs = c*s
    s2 = s*s
    p1 = c2 + epsilon*s2
    p2 = cs*(1-epsilon)
    p3 = cs*(1-epsilon)
    p4 = s2 + epsilon*c2
    stencil = p1*Kernelxx + p2*Kernelxy + p3*Kernelyx + p4*Kernelyy
    return stencil.double()

# %%
def ComputeSmootherFactor(KernelA, N):
    device = KernelA.device
    w = 1/2
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
        taus = [w/KernelA[i, 0, 1, 1]]*5
        y = 1
        for j in range(len(taus)):
            y *= 1-taus[j] * (k1*torch.exp(-theta1)*torch.exp(-theta2)+k2*torch.exp(-theta2)+k3*torch.exp(theta1)*torch.exp(-theta2)+k4*torch.exp(-theta1)+k5+k6*torch.exp(theta1)+k7*torch.exp(-theta1)*torch.exp(theta2)+k8*torch.exp(theta2)+k9*torch.exp(theta1)*torch.exp(theta2))
        Y[i, 0, :, :] = torch.abs(y)
    return Y

# %%
# rhs = np.load("rhs63.npy")

# %%
N = 63
h = 1/N
n = N-1 
setup_seed(1234)
# %%
for m in range(10):
    kernels = []
    sols = []
    LFAs = []
    b = np.ones((n**2,))*h**2
    for i in range(10):
        epsilon = torch.pow(0.1, torch.rand(1)*5)
        theta1 = torch.rand(1)
        theta = theta1*np.pi
        stencil = gen_stencil(epsilon, theta)
        Y = ComputeSmootherFactor(stencil, n)
        A = stencil_grid(stencil[0,0].numpy(), (n, n), dtype=float, format='csr')
        # b = rhs[i].flatten()*h**2
        # x_star = spla.spsolve(A, b)
        # sols.append(x_star.reshape(1,n,n))
        kernels.append(stencil[0].numpy())
        LFAs.append(Y[0].numpy())
    print(m)
    # sols = np.asarray(sols)
    kernels = np.asarray(kernels)
    LFAs = np.asarray(LFAs)

    # np.save(f"lfa_data/{N}/sols{m}", sols)
    np.save(f"lfa_data/{N}/kernelA{m}", kernels)
    np.save(f"lfa_data/{N}/LFA{m}", LFAs)

# 合并
N = N-1
data_kernelA = np.empty((1, 1, 3, 3), dtype=np.float64)
data_lfa = np.empty((1, 1, N, N), dtype=np.float64)
# data_u = np.empty((1, 1, N, N), dtype=np.float64)

for i in range(10):
    kernelA = np.load(f"lfa_data/{N+1}/kernelA{i}.npy")
    lfa = np.load(f"lfa_data/{N+1}/LFA{i}.npy")
    # u = np.load(f"lfa_data/{N+1}/sols{i}.npy")
    print( lfa.shape, kernelA.shape)
    # data_u = np.concatenate((data_u, u), axis=0)
    data_lfa = np.concatenate((data_lfa, lfa), axis=0)
    data_kernelA = np.concatenate((data_kernelA, kernelA), axis=0)
# np.save(f"data_u{N}", data_u[1:, :, :, :])
np.save(f"data_lfa{N+1}", data_lfa[1:, :, :, :])
np.save(f"data_kernelA{N+1}", data_kernelA[1:, :, :, :])
