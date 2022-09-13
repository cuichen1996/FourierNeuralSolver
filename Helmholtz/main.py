# -*- coding: utf-8 -*-
# @Author: Chen Cui
# @Date:   2022-05-24 12:25:09
# @Last Modified by:   Your name
# @Last Modified time: 2022-09-12 00:51:58

# %%
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import model.model as solver
from data.gen_data import CreateTestLoader, CreateTrainLoader
from utils.misc import mkdirs


# %%
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def BatchConv2d(inputs, kernels, stride=1, padding=1):
    batch_size = inputs.shape[0]
    m1 = inputs.shape[2]
    m2 = inputs.shape[3]
    out = F.conv2d(inputs.view(1, batch_size, m1, m2), kernels, stride=stride, padding=padding, bias=None, groups=batch_size)
    return out.view(batch_size, 1, m1, m2)

def train(config):
    # * data
    if config['train']:
        train_loader = CreateTrainLoader(config)
    test_loader = CreateTestLoader(config)

    # * model
    FNS = solver.HyperFNS(config)

    # load checkpoint if in post mode
    if config['restart']:
        checkpoint = config['restart_dir'] + '/model_{}.pth'.format(config['restart_epoch'])
        FNS.load_state_dict(torch.load(checkpoint))
        logging.info('Loaded pre-trained model: {}'.format(checkpoint))
        

    if config["use_cuda"]:
        FNS = FNS.cuda()

    # * opt
    optimizer = torch.optim.Adam(FNS.parameters(), lr=config['lr'])
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # * train
    epoch = 0
    if config['train']:
        for epoch in range(1, config['epoch'] + 1):
            train_loss = 0.0
            FNS.train()
            epoch_start_time = time.time()
            for f, kernelA, u in train_loader:
                x = torch.zeros_like(f)
                if config["use_cuda"]:
                    x, f, u, kernelA = x.cuda(), f.cuda(), u.cuda(), kernelA.cuda()
                r_old = f - BatchConv2d(x, kernelA)
                for i in range(config["K"]):
                    res, x = FNS(x, f, kernelA, r_old)
                    x = x.data
                    r_old = f - BatchConv2d(x, kernelA)
                    optimizer.zero_grad()
                    res.backward()
                    # torch.nn.utils.clip_grad_value_(FNS.parameters(), config['grad_clip'])
                    optimizer.step()
                train_loss += res.item() * x.shape[0]
            train_loss /= len(train_loader.dataset)
            logging.info("epoch: [{:d}/{:d}] {:.2f} sec(s) Train Loss: {:.9f} ".format(
                epoch, config['epoch'], time.time()-epoch_start_time, train_loss))
            # ExpLR.step()
            if epoch % config['ckpt_freq'] == 0:
                torch.save(FNS.state_dict(), config['checkpoints_folder'] + "/model_"+str(epoch)+".pth")

    else:
        FNS.eval()
        with torch.no_grad():
            for f, kernelA, u in test_loader:
                x = torch.zeros_like(f)
                if config["use_cuda"]:
                    x, f, u, kernelA = x.cuda(), f.cuda(), u.cuda(), kernelA.cuda()
                x, num_iter = FNS.test(x, f, kernelA)
                rerror = torch.norm(x - u) / torch.norm(u)
                logging.info("The relative error is {:.9f} after running FNS {:d} iterations.".format(rerror,num_iter))

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using GPU, %s' % torch.cuda.get_device_name(0)) if device == 'cuda' else print('Using CPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.set_default_dtype(torch.float64)
    # setup_seed(1111)
    
    # *=======================参数设置=================================
    config = {}
    
    # data
    config["kmin"]       = 100
    config["kmax"]       = 120
    config["pde_num"]    = 20
    config["every_num"]  = 100
    config['batch_size'] = 100
    config["N"]          = 255
    config["test_k"]     = [125]

    # HelmNet
    config["activation_function"] = "prelu"
    config["depth"]          = 2  # MG Level
    config["inchannels"]     = 3
    config["features"]       = 32  # hidden Features
    config["state_channels"] = 2
    config["state_depth"]    = 2

    # model
    config["mL"]              = 3
    config["kernel_size"]     = 5
    config["mid_chanel"]      = 4
    
    config["softshrink"]      = 1e-4
    config["paras_size"]      = (config["N"], config["N"]//2+1)

    config["kernelSize"]      = 7
    config["act"]             = "prelu"
    config["modes"]           = (config["N"]//4, config["N"]//4)
    config["max_iter_num"]    = 1000
    config["error_threshold"] = 1e-6
    config["K"]               = 200
    config["xavier_init"]     = 1e-2
    if device == "cuda":
        config["use_cuda"] = True
    else:
        config["use_cuda"] = False
        
    # dir
    config['run_dir'] = "/home/kaijiang/cuichen/FNSsummer/0909/expriments/Krylov/log10"
    config['checkpoints_folder'] = config['run_dir'] + '/checkpoints'
    config['prediction_folder'] = config['run_dir'] + '/prediction'
    config['restart_dir'] = '/home/kaijiang/cuichen/FNSsummer/0909/expriments/Krylov/log10/checkpoints'

    # train
    config['grad_clip']     = 20.0
    config['lr']            = 1e-4
    config['epoch']         = 20000
    config['ckpt_freq']     = 1
    config['restart_epoch'] = 21
    
    # 是否加载预训练模型
    config['restart'] = True
    # config['restart'] = False
    # 是否训练
    config['train'] = False
    # config['train'] = True

    mkdirs([config['run_dir'], config['checkpoints_folder'], config['prediction_folder']])
    # filemode='w',
    logging.basicConfig(filename=config['run_dir'] + "/train.log", format='%(asctime)s %(message)s', level=logging.INFO)
    # *=================================================================
    import time
    logging.info('Start training..............................................')
    tic = time.time()
    train(config)
    tic2 = time.time()
    logging.info("Finished training {} epochs using {} seconds"
                 .format(config['epoch'], tic2 - tic))

    logging.info("This time training parameter setting: ")
    with open(config['run_dir'] + "/train.log", 'a') as args_file:json.dump(config, args_file, indent=4)
