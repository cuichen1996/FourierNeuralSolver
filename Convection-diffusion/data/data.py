import torch
import torch.nn.functional as F
import torch.utils.data as data
torch.set_default_dtype(torch.float64)

def CreateKernelA(paras, data_num=100):
    m = paras.shape[0]
    KernelA = torch.zeros([m*data_num, 1, 3, 3], dtype=torch.float64)
    for i in range(m):
        epsilon, a, b, h = paras[i]
        stencil = torch.zeros(3, 3)
        stencil[1, 1] = 4*epsilon
        stencil[0, 1] = b*h/2 - epsilon
        stencil[1, 0] = -a*h/2 - epsilon
        stencil[1, 2] = a*h/2 - epsilon
        stencil[2, 1] = -b*h/2 - epsilon
        for j in range(data_num):
            KernelA[i*data_num+j, 0] = stencil
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

def BatchConv2d(inputs, kernels, stride=1, padding=0):
    batch_size = inputs.shape[0]
    m1 = inputs.shape[2]
    m2 = inputs.shape[3]
    out = F.conv2d(inputs.view(1, batch_size, m1, m2), kernels, stride=stride, padding=padding, bias=None, groups=batch_size)
    return out.view(batch_size, 1, m1-2, m2-2)

def CreateMetaDatasetLoader(config):
    h = 1/(config["N"]+1)
    # epsilons_train = torch.pow(0.1, torch.rand(config["pde_num"])*2)*10  
    epsilons_train = torch.pow(0.1, torch.rand(config["pde_num"])*3)*1e-2
    print(epsilons_train)
    etas_train = torch.zeros(config["pde_num"], 4)
    etas_train[:, 0] = epsilons_train
    etas_train[:, 1] = config["a"]
    etas_train[:, 2] = config["b"]
    etas_train[:, 3] = h

    kernelAs = CreateKernelA(etas_train, data_num=config["every_num"])
    total_num = config["pde_num"]*config["every_num"]
    
    if config["loss"] == 'u':
        gu = torch.randn([total_num, 1, config["N"], config["N"]], dtype=torch.float64)
        u = F.pad(gu,(1,0,1,0), "constant", 0)
        u = F.pad(u,(0,1,0,1), "constant", 1)
        gf = BatchConv2d(u, kernelAs)
    else:
        gf = torch.zeros([total_num, 1, config["N"], config["N"]], dtype=torch.float64)
        gu = torch.zeros([total_num, 1, config["N"], config["N"]], dtype=torch.float64)
        
    dataset = CreateMetaDataset(total_num, gf, kernelAs, gu)
    dataLoader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    return dataLoader

#%% test
from math import pi
def CreateTestLoader(config):
    h = 1/(config["N"]+1)
    epsilons = torch.tensor(config["test_epsilons"])
    test_num = epsilons.shape[0]
    
    eta = torch.zeros([test_num, 4])
    eta[:,0] = epsilons
    eta[:,1] = config["a"]
    eta[:,2] = config["b"]
    eta[:,3] = h
    
    kernelA = CreateKernelA(eta, data_num=1)
    gf = torch.zeros([test_num, 1, config["N"], config["N"]], dtype=torch.float64)
    gu = torch.zeros([test_num, 1, config["N"], config["N"]], dtype=torch.float64)  #* FAKE

    dataset = CreateMetaDataset(test_num, gf, kernelA, gu)
    dataLoader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataLoader