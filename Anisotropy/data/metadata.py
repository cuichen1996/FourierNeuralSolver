# Generate data code from http://bicmr.pku.edu.cn/~dongbin/Data/MetaMgNet.rar

import torch
import torch.nn.functional as F
import torch.utils.data as data
torch.set_default_dtype(torch.float64)

Kernelxx = torch.tensor([[[[-1./6., 1./3., -1./6.], [-2./3., 4./3., -2./3.], [-1./6., 1./3., -1./6.]]]], dtype=torch.float64)  # \partial_{xx}
Kernelxy = torch.tensor([[[[-1./4., 0., 1./4.], [0., 0., 0.], [1./4., 0., -1./4.]]]], dtype=torch.float64)   # \partial_{xy}
Kernelyx = torch.tensor([[[[-1./4., 0., 1./4.], [0., 0., 0.], [1./4., 0., -1./4.]]]], dtype=torch.float64)    # \partial_{yx}
Kernelyy = torch.tensor([[[[-1./6., -2./3., -1./6.], [1./3., 4./3., 1./3.], [-1./6., -2./3., -1./6.]]]], dtype=torch.float64)    # \partial_{yx}

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
class CreateMetaDataset(data.Dataset):
    def __init__(self, p, data1, data2, label, train=0):
        self.p = p
        self.train = train
        self.data1 = data1
        self.data2 = data2
        self.label = label

    def __getitem__(self, index):
        f = self.data1[index]
        kernelA = self.data2[index]
        u = self.label[index]
        return f, kernelA, u

    def __len__(self):
        return self.p


def BatchConv2d(inputs, kernels, stride=1, padding=1):
    batch_size = inputs.shape[0]
    m1 = inputs.shape[2]
    m2 = inputs.shape[3]
    out = F.conv2d(inputs.view(1, batch_size, m1, m2), kernels,
                   stride=stride, padding=padding, bias=None, groups=batch_size)
    return out.view(batch_size, 1, m1, m2)


def CreateMetaDatasetLoader(config):
    epsilons_train = torch.pow(0.1, torch.rand(config["pde_num"])*5)
    thetas_train = 0*torch.ones(config["pde_num"])
    etas_train = torch.zeros(config["pde_num"], 4)
    c = torch.cos(thetas_train)
    s = torch.sin(thetas_train)
    c2 = c*c
    cs = c*s
    s2 = s*s
    etas_train[:, 0] = c2 + epsilons_train*s2
    etas_train[:, 1] = cs*(1-epsilons_train)
    etas_train[:, 2] = cs*(1-epsilons_train)
    etas_train[:, 3] = s2 + epsilons_train*c2

    kernelAs = CreateKernelA(etas_train, data_num=config["every_num"], repeat=True)

    total_num = config["pde_num"]*config["every_num"]
    if config["loss"] == 'u':
        gu = torch.randn([total_num, 1, config["N"], config["N"]], dtype=torch.float64)
        gf = BatchConv2d(gu, kernelAs)
    else:
        gf = torch.randn([total_num, 1, config["N"], config["N"]], dtype=torch.float64)
        gu = torch.zeros([total_num, 1, config["N"], config["N"]], dtype=torch.float64)
    dataset = CreateMetaDataset(total_num, gf, kernelAs, gu)
    dataLoader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    return dataLoader

# config = {}
# config["pde_num"] = 20
# config["every_num"] = 100
# config["loss"] = 'u' # 'u' or 'r'
# config['batch_size'] = 100
# config["N"] = 64
# train_loader = CreateMetaDatasetLoader(config)

# #%%
# for f, kernelA, u in train_loader:
#     print(f.shape, kernelA.shape, u.shape)
#     break
# %%

# test
from math import pi

def CreateTestLoader(config):
    epsilons = torch.tensor(config["test_epsilons"])
    test_num = epsilons.shape[0]
    eta = torch.zeros([test_num, 4])
    theta = config["test_theta"]*pi*torch.ones(test_num)
    c = torch.cos(theta)
    s = torch.sin(theta)
    c2 = c*c
    cs = c*s
    s2 = s*s
    eta[:,0] = c2 + epsilons*s2
    eta[:,1] = cs*(1-epsilons)
    eta[:,2] = cs*(1-epsilons)
    eta[:,3] = s2 + epsilons*c2
    kernelA = CreateKernelA(eta, data_num=10, repeat=True)
    gu = torch.randn([10,1,config["N"], config["N"]], dtype=torch.float64)
    gf = BatchConv2d(gu, kernelA)
    # gf = torch.ones([10,1,config["N"], config["N"]], dtype=torch.float64)
    dataset = CreateMetaDataset(10, gf, kernelA, gu)
    dataLoader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataLoader

        