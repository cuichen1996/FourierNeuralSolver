import json
import logging

import model.HINTS as solver
import numpy as np
import torch
import torch.nn as nn
from data.load_data import CreateLFALoader, CreateTrainLoader, CreateTestLoader
from model.misc import *

def train(config):
    # * data
    if config['train']:
        train_loader = CreateTrainLoader(config, 63)
    test_loader, epsilons = CreateTestLoader(config, config['N'])

    # * model
    FNS = solver.HyperFNS(config)
    # FNS = nn.DataParallel(FNS)
    total_params = count_parameters(FNS)
    print("Total parameters: ", total_params)

    if config['restart']:
        checkpoint = config['restart_dir'] + '/model_{}.pth'.format(config['restart_epoch'])
        FNS.load_state_dict({k.replace('module.',''):v for k,v in torch.load(checkpoint, map_location=device).items()})
        logging.info('Loaded pre-trained model: {}'.format(checkpoint))
        
    if config["use_cuda"]:
        FNS = FNS.cuda()

    # * opt
    optimizer = torch.optim.Adam(FNS.parameters(), lr=config['lr'])

    # * train
    epoch = 0
    if config['train']:
        for epoch in range(1, config['epoch'] + 1):
            train_loss = 0.0
            FNS.train()
            epoch_start_time = time.time()
            for f, kernelA, u in train_loader:
                if config["use_cuda"]:
                    f, u, kernelA = f.float().cuda(), u.float().cuda(), kernelA.float().cuda()
                res = FNS(f, kernelA, u, epoch)
                optimizer.zero_grad()
                res.backward()
                optimizer.step()
                train_loss += res.item()
            train_loss /= len(train_loader.dataset)
            logging.info("epoch: [{:d}/{:d}] {:.2f} sec(s) Train Loss: {:.9f} ".format(
                epoch, config['epoch'], time.time()-epoch_start_time, train_loss))

            if epoch % config['ckpt_freq'] == 0:
                torch.save(FNS.state_dict(), config['checkpoints_folder'] + "/model_"+str(epoch)+".pth")

            if epoch % 10 == 0:
                with torch.no_grad():
                    for f, kernelA, u in test_loader:
                        if config["use_cuda"]:
                            f, u, kernelA = f.cuda(), u.cuda(), kernelA.cuda()
                        x, ress = FNS.test(f, kernelA, epoch)
    else:
        with torch.no_grad():
            iter_num = []
            timess = []
            for f, kernelA, u in test_loader:
                if config["use_cuda"]:
                    f, u, kernelA = f.cuda(), u.cuda(), kernelA.cuda()
                tic= time.time()
                x, ress = FNS.test(f, kernelA, 1)
                toc= time.time()
                iter_num.append(len(ress[0])-1)
                timess.append(toc-tic)
            logging.info("The test epsilons are: {}".format(epsilons))
            logging.info("The iterations are: {}".format(iter_num))
            logging.info("Grid size {}: Iterations is {:.2f} std {:.2f}. Time is {:.2f} std {:.2f}.".format(f.shape[-1], np.mean(iter_num), np.std(iter_num),np.mean(timess), np.std(timess)))

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using GPU, %s' % torch.cuda.get_device_name(0)) if device == 'cuda' else print('Using CPU')
    setup_seed(1234)

    # *=======================参数设置=================================
    config = {}
    
    # data
    config["pde_num"]       = 200
    config["every_num"]     = 50
    config["loss"]          = 'un'
    config['batch_size']    = 20
    config["N"]             = 63
    config["test_epsilons"] = 10**(-torch.rand(10) * 6)
    config["test_theta"]  = [5/12]*10

    # model
    config["smoother"]       = "jacobi"
    config["kernel_size"]    = 3
    config["act"]            = "gelu"

    config["Meta_Type"]      = "sFNO"  
    config["modes"]          = [16,16,16,16]
    config["depths"]         = [3,3,9,3]
    config["dims"]           = [32,32,32,32]
    config["drop_path_rate"] = 0.3
    config["drop"]           = 0.
    config["padding"]        = 9

    config["channels"]   = [8, 16, 32, 64, 64, 32, 16, 8, 9]

    # hints
    config["p"] = 80
    config["input_channel"] = 1

    config["max_iter_num"]    = 500
    config["error_threshold"] = 1e-6
    config["xavier_init"]     = 1e-0
    
    if device == "cuda":
        config["use_cuda"] = True
    else:
        config["use_cuda"] = False
        
    # dir
    config['run_dir'] = "expriments/HINTS"
    config['checkpoints_folder'] = config['run_dir'] + '/checkpoints'
    config['prediction_folder'] = config['run_dir'] + '/prediction'
    config['restart_dir'] = config['checkpoints_folder']

    # train
    config['grad_clip']     = 1.0
    config['lr']            = 1e-4
    config['step_size']     = 1000
    config['gamma']         = 0.5
    config['epoch']         = 10000
    config['ckpt_freq']     = 10
    config['restart_epoch'] = 600

    config['restart'] = True
    # config['restart'] = False
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
