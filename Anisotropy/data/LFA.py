# %%
from matplotlib import pyplot as plt
import numpy as np
import torch
# %%
Kernelxx = np.array([[[[-1./6., 1./3., -1./6.], [-2./3., 4./3., -2./3.], [-1./6., 1./3., -1./6.]]]])  # \partial_{xx}
Kernelxy = np.array([[[[-1./4., 0., 1./4.], [0., 0., 0.], [1./4., 0., -1./4.]]]])  # \partial_{xy}
Kernelyx = np.array([[[[-1./4., 0., 1./4.], [0., 0., 0.], [1./4., 0., -1./4.]]]])  # \partial_{yx}
Kernelyy = np.array([[[[-1./6., -2./3., -1./6.], [1./3., 4./3.,1./3.], [-1./6., -2./3., -1./6.]]]])  # \partial_{yx}

# %%
def CreateKernelA(epsilons, data_num=100, repeat=True):
    m = epsilons.shape[0]
    if repeat:
        KernelA = torch.zeros([m*data_num, 1, 3, 3], dtype=torch.float64)
        for i in range(m):
            K = epsilons[i, 0]*Kernelxx + \
                (epsilons[i, 1] + epsilons[i, 2]) * \
                Kernelxy + epsilons[i, 3]*Kernelyy
            for j in range(data_num):
                KernelA[i*data_num+j] = K
    else:
        KernelA = torch.zeros([m, 1, 3, 3], dtype=torch.float64)
        for i in range(m):
            KernelA[i] = epsilons[i, 0]*Kernelxx + \
                (epsilons[i, 1] + epsilons[i, 2]) * \
                Kernelxy + epsilons[i, 3]*Kernelyy

    return KernelA

# %%
def ComputeSmootherFactor(KernelA, N, tol, taus):
    device = KernelA.device
    h = 1/(N+1)
    p1 = range(-N//2, N//2+1)
    p2 = range(-N//2, N//2+1)
    # p1 = np.fft.fftshift(p2)
    # p2 = np.fft.fftshift(p2)
    P1, P2 = np.meshgrid(p1, p2)
    P1 = torch.from_numpy(P1)
    P2 = torch.from_numpy(P2)
    P1, P2 = P1.to(device), P2.to(device)
    theta1 = 2*np.pi*(P1*h-P2*h)
    theta2 = 2*np.pi*P1*h
    theta3 = 2*np.pi*(P1*h+P2*h)
    theta4 = 2*np.pi*P2*h
    Y = torch.ones([KernelA.shape[0], 1, N, N], device=device)
    for i in range(KernelA.shape[0]):
        k1, k2, k3, k4, k5 = KernelA[i, 0, 0, 0], KernelA[i, 0, 0, 1], KernelA[i, 0, 0, 2], KernelA[i, 0, 1, 0], KernelA[i, 0, 1, 1]
        y = 1
        for j in range(len(taus)):
            y *= 1-taus[j] * (k1*2*torch.cos(theta3)+k2*2*torch.cos(theta4)+k3*2*torch.cos(theta1)+k4*2*torch.cos(theta2)+k5)
            ax = plt.gca()
            ax.set_aspect(1)
            plt.pcolor(P1, P2, np.abs(y.numpy()), cmap='jet')
            plt.colorbar()
            plt.xlabel(r'$p_1$', fontsize=16)
            plt.ylabel(r'$p_2$', fontsize=16)
            plt.savefig("/LFA/"+'d_1_'+str(len(taus))+'_'+str(j+1)+'.png', dpi=300)
            plt.close()
        # Y[i, 0, :, :] = y
        break
    idx = torch.where(torch.abs(Y) < tol)
    return idx

# %%
epsilons_train = torch.Tensor([1e-6, 1e-6])
thetas_train = np.pi*torch.ones(2)*0.9
etas_train = torch.zeros(2, 4)
c = torch.cos(thetas_train)
s = torch.sin(thetas_train)
c2 = c*c
cs = c*s
s2 = s*s
etas_train[:, 0] = c2 + epsilons_train*s2
etas_train[:, 1] = cs*(1-epsilons_train)
etas_train[:, 2] = cs*(1-epsilons_train)
etas_train[:, 3] = s2 + epsilons_train*c2

KernelA  = CreateKernelA(etas_train, data_num=1, repeat=True)   

N = 63
h = 1/(N+1)
L = 4
miu = L*h
m = 10
roots = [np.cos((np.pi*(2*i+1)) / (2*m)) for i in range(m)]
# taus = [2 / (L + miu - (miu - L) * r) for r in roots]
taus = [2/3/KernelA[0, 0, 1,1], 2/3/KernelA[0, 0, 1,1], 2/3/KernelA[0, 0, 1,1], 2/3/KernelA[0, 0, 1,1], 2/3/KernelA[0, 0, 1,1]]
print(taus)
idx = ComputeSmootherFactor(KernelA, N, 0.5, taus)

# %%
KernelA = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]])
taus = [2/3/KernelA[0, 0, 1,1]]
print(taus)
idx = ComputeSmootherFactor(KernelA, N, 0.5, taus)

# %%



