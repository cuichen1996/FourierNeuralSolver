# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2022-09-11 05:47:33
# @Last Modified by:   Your name
# @Last Modified time: 2022-09-11 06:05:34
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.metadata import CreateMetaDatasetLoader, CreateTestLoader
import model.model as solver
from utils.misc import mkdirs

# %%
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(config):
    # * data
    if config['train']:
        train_loader = CreateMetaDatasetLoader(config)
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
    # ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

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
                res = FNS(x, f, kernelA)
                optimizer.zero_grad()
                res.backward()
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
            iter_num = []
            for f, kernelA, u in test_loader:
                x = torch.zeros_like(f)
                if config["use_cuda"]:
                    x, f, u, kernelA = x.cuda(), f.cuda(), u.cuda(), kernelA.cuda()
                x, num_iter = FNS.test(x, f, kernelA)
                rerror = torch.norm(x - u) / torch.norm(u)
                iter_num.append(num_iter)
                logging.info("The relative error is {:.9f} after running FNS {:d} iterations.".format(rerror,num_iter))
            logging.info("The average number of iterations is {:.2f} max {:.2f}.".format(np.mean(iter_num), np.std(iter_num)))

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using GPU, %s' % torch.cuda.get_device_name(0)) if device == 'cuda' else print('Using CPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.set_default_dtype(torch.float64)
    setup_seed(1234)
    
    # *=======================参数设置=================================
    config = {}
    
    # data
    config["pde_num"]       = 20
    config["every_num"]     = 100
    config["loss"]          = 'u' # 'u' or 'r'
    config['batch_size']    = 50
    config["N"]             = 31
    config["test_epsilons"] = [1e-6]
    config["test_theta"]    = 0

    # model
    config["smoother_times"] = 1
    config["alpha"]          = 3
    config["m"]              = 10

    config["mid_chanel"] = 4
    config["act"]        = "relu"
    
    config["max_iter_num"]    = 2000
    config["error_threshold"] = 1e-6
    config["K"]               = 10
    config["xavier_init"]     = 1e-2
    
    if device == "cuda":
        config["use_cuda"] = True
    else:
        config["use_cuda"] = False
    # dir
    config['run_dir'] = "expriments/Cheby"
    config['checkpoints_folder'] = config['run_dir'] + '/checkpoints'
    config['prediction_folder'] = config['run_dir'] + '/prediction'
    config['restart_dir'] = 'expriments/Cheby/checkpoints'

    # train
    config['lr']            = 1e-4
    config['epoch']         = 20000
    config['ckpt_freq']     = 10
    config['restart_epoch'] = 330   
     
    # 是否加载预训练模型
    config['restart'] = True
    config['restart'] = False
    # 是否训练
    config['train'] = False
    config['train'] = True


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
