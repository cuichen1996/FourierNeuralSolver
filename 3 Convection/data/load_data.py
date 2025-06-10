import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np

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
    return stencil.reshape(1, 3, 3)


# %%
def CreateKernelA(params, data_num=100, repeat=True, ddtype="supg"):
    m = params.shape[0]
    if repeat:
        KernelA = torch.zeros([m*data_num, 1, 3, 3], dtype=torch.double)
        for i in range(m):
            epsilon, wx, wy, h = params[i, 0], params[i, 1], params[i, 2], params[i, 3]
            K = gen_stencil(epsilon, wx, wy, h, ddtype)
            for j in range(data_num):
                KernelA[i*data_num+j] = K
    else:
        KernelA = torch.zeros([m, 1, 3, 3], dtype=torch.double)
        for i in range(m):
            epsilon, wx, wy, h = params[i, 0], params[i, 1], params[i, 2], params[i, 3]
            KernelA[i] = gen_stencil(epsilon, wx, wy, h, ddtype)

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

def BatchConv2d(inputs, Kernels, stride=1, padding=1):
    batch_size = inputs.shape[0]
    du = F.conv2d(inputs.permute(1,0,2,3), Kernels.to(inputs.dtype), stride=stride, padding=padding, bias=None, groups=batch_size)
    return du.permute(1,0,2,3)

def CreateTrainLoader(config, N):
    epsilons_train = torch.pow(0.1, torch.rand(config["pde_num"])*3)
    thetas_train = torch.rand(config["pde_num"])*torch.pi
    c = torch.cos(thetas_train)
    s = torch.sin(thetas_train)
    h = 1/(N+1)

    params = torch.zeros(config["pde_num"], 4, dtype=torch.double)
    params[:, 0] = epsilons_train
    params[:, 1] = c
    params[:, 2] = s
    params[:, 3] = h

    kernelAs = CreateKernelA(params, data_num=config["every_num"], repeat=True, ddtype=config["ddtype"])
    total_num = config["pde_num"]*config["every_num"]
    gf = torch.randn([total_num, 1, N, N], dtype=torch.double)
    gu = torch.zeros([total_num, 1, N, N], dtype=torch.double)
    dataset = CreateMetaDataset(total_num, gf, kernelAs, gu)
    if N == 63:
        batchsize = 40
    elif N == 127:
        batchsize = 20
    else:
        batchsize = 10
    dataLoader = data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
    return dataLoader

def CreateFULoader(config):
    N = 63
    h = 1/(N+1)
    datalfa = np.load(f"data/data_lfa{N}.npy")
    data_kernel = np.load(f"data/data_kernelA{N}.npy")
    datau = np.load(f"data/data_u{N}.npy")
    dataf = np.ones_like(datau)*h**2
    lfa = torch.from_numpy(datalfa)
    kernelA = torch.from_numpy(data_kernel)
    # u = torch.from_numpy(datau)
    f = torch.from_numpy(dataf)
    train_num = f.shape[0]
    print(f.shape, kernelA.shape, lfa.shape)
    trainset = CreateMetaDataset(train_num, f, kernelA, lfa)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    return train_loader

def CreateLFALoader(config, N):
    h = 1/(N+1)
    datalfa = np.load(f"data/data_lfa{N}.npy")
    data_kernel = np.load(f"data/data_kernelA{N}.npy")
    # datau = np.load(f"data/data_u{N}.npy")
    dataf = np.random.randn(datalfa.shape[0],1,N,N)

    lfa = torch.from_numpy(datalfa[:100,:,:,:])
    kernelA = torch.from_numpy(data_kernel[:100,:,:,:])
    # u = torch.from_numpy(datau)
    f = torch.from_numpy(dataf[:100,:,:,:])
    train_num = f.shape[0]
    print(f.shape, kernelA.shape, lfa.shape)
    trainset = CreateMetaDataset(train_num, f, kernelA, lfa)

    if N == 31:
        batchsize = 80
    elif N == 63:
        batchsize = 40
    elif N == 127:
        batchsize = 20
    else:
        batchsize = 10
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
    # train_loader = torch.utils.data.DataLoader(
    #     trainset,
    #     batch_size=batchsize,
    #     shuffle=False,
    #     pin_memory=True,
    #     sampler=DistributedSampler(trainset),
    # )
    return train_loader


def CreateTestLoader(config, N):
    if N is None:
        N = config["N"]
    h = 1/(N+1)

    epsilons = torch.tensor(config["test_epsilons"]).to(torch.float64)
    theta = torch.tensor(config["test_theta"]).to(torch.float64)*torch.pi
    c = torch.cos(theta)
    s = torch.sin(theta)
    
    test_num = epsilons.shape[0]
    params = torch.zeros(test_num, 4, dtype=torch.double)
    params[:, 0] = epsilons
    params[:, 1] = c
    params[:, 2] = s
    params[:, 3] = h

    data_num = epsilons.shape[0]
    kernelA = CreateKernelA(params, data_num=data_num, repeat=False, ddtype=config["ddtype"])
    gf = torch.ones([data_num,1,config["N"], config["N"]], dtype=torch.double)*h**2
    gu = torch.zeros([data_num,1,config["N"], config["N"]], dtype=torch.double)
    dataset = CreateMetaDataset(data_num, gf, kernelA, gu)
    dataLoader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataLoader