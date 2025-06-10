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

def gen_stencil(epsilon, wx, wy, h, ddtype="supg"):
    # print(epsilon, wx, wy, h)
    wlength = np.sqrt(wx**2+wy**2)
    pk = wlength*h/(2*epsilon)
    if ddtype == "supg" and pk > 1:
        delta = h/(2*wlength)*(1-1/pk)
    else:
        delta = 0

    wy = -wy
    stencil_diff = 1/3*np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

    stencil_cvec = h/12*np.array([[-wx+wy,4*wy,wx+wy],[-4*wx,0,4*wx],[-(wx+wy),-4*wy,wx-wy]])

    w2 = wx**2+wy**2
    wxy = wx*wy
    stencil_supg = delta*np.array([[-1/6*w2+1/2*wxy,1/3*wx**2-2/3*wy**2,-1/6*w2-1/2*wxy],
                                    [-2/3*wx**2+1/3*wy**2,4/3*w2,-2/3*wx**2+1/3*wy**2],
                                    [-1/6*w2-1/2*wxy,1/3*wx**2-2/3*wy**2,-1/6*w2+1/2*wxy]])
    
    stencil = epsilon*stencil_diff + stencil_cvec + stencil_supg
    return stencil.reshape(1, 1, 3, 3)

def ComputeSmootherFactor(KernelA, N):
    w = 1/2
    h = 1/(N+1)
    p1 = range(-N//2, N//2)
    p2 = range(-N//2, N//2)
    P1, P2 = np.meshgrid(p1, p2)
    P1 = torch.from_numpy(P1)
    P2 = torch.from_numpy(P2)
    theta1 = 2j*np.pi*P1*h
    theta2 = 2j*np.pi*P2*h
    Y = torch.ones([KernelA.shape[0], 1, N, N])
    for i in range(KernelA.shape[0]):
        k1, k2, k3, k4, k5, k6, k7, k8, k9 = KernelA[i].flatten()[:]
        taus = [w/KernelA[i, 0, 1, 1]]*10
        y = 1
        for j in range(len(taus)):
            y *= 1-taus[j] * (k1*torch.exp(-theta1)*torch.exp(-theta2)+k2*torch.exp(-theta2)+k3*torch.exp(theta1)*torch.exp(-theta2)+k4*torch.exp(-theta1)+k5+k6*torch.exp(theta1)+k7*torch.exp(-theta1)*torch.exp(theta2)+k8*torch.exp(theta2)+k9*torch.exp(theta1)*torch.exp(theta2))
        Y[i, 0, :, :] = torch.abs(y)
    return Y

# %%
N = 128
h = 1/N
n = N-1 
ddtype = "fem"
setup_seed(1234)
# %%
for m in range(10):
    kernels = []
    sols = []
    LFAs = []
    b = np.ones((n**2,))*h**2
    for i in range(1000):
        epsilons = torch.pow(0.001, torch.rand(1)*6)
        thetas = torch.rand(1)*torch.pi
        c = torch.cos(thetas)
        s = torch.sin(thetas)
        stencil = gen_stencil(epsilons.item(), c.item(), s.item(), h, ddtype)
        Y = ComputeSmootherFactor(stencil, n)
        # A = stencil_grid(stencil[0,0].numpy(), (n, n), dtype=float, format='csr')
        # b = rhs[i].flatten()*h**2
        # x_star = spla.spsolve(A, b)
        # sols.append(x_star.reshape(1,n,n))
        kernels.append(stencil[0])
        LFAs.append(Y[0].numpy())
    print(m)
    # sols = np.asarray(sols)
    kernels = np.asarray(kernels)
    LFAs = np.asarray(LFAs)

    # np.save(f"lfa_data/{N}/sols{m}", sols)
    np.save(f"lfa_data/{N}/kernelA{m}", kernels)
    np.save(f"lfa_data/{N}/LFA{m}", LFAs)


N = N-1
data_kernelA = np.empty((1, 1, 3, 3), dtype=np.float64)
data_lfa = np.empty((1, 1, N, N), dtype=np.float64)
# data_u = np.empty((1, 1, N, N), dtype=np.float64)

for i in range(10):
    kernelA = np.load(f"lfa_data/{N+1}/kernelA{i}.npy")
    lfa = np.load(f"lfa_data/{N+1}/LFA{i}.npy")
    # u = np.load(f"lfa_data/{N+1}/sols{i}.npy")
    print(lfa.shape, kernelA.shape)
    # data_u = np.concatenate((data_u, u), axis=0)
    data_lfa = np.concatenate((data_lfa, lfa), axis=0)
    data_kernelA = np.concatenate((data_kernelA, kernelA), axis=0)
# np.save(f"data_u{N}", data_u[1:, :, :, :])
np.save(f"data_lfa{N}", data_lfa[1:, :, :, :])
np.save(f"data_kernelA{N}", data_kernelA[1:, :, :, :])
