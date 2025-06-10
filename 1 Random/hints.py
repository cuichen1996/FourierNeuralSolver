import json
import logging
import time

import numpy as np
import torch

from data.data import CreateTrainLoader, CreateTestLoader
import model.HINTS as solver
from model.misc import *

def train(config):
    # * data
    if config['train']:
        train_loader = CreateTrainLoader(config)

    test_loader = [CreateTestLoader(config)]

    # * model
    FNS = solver.HyperFNS(config)
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
            for f, coef, u in train_loader:
                if config["use_cuda"]:
                    f, coef, u = f.cuda(), coef.cuda(), u.cuda()
                res = FNS(f, coef, u, epoch)
                optimizer.zero_grad()
                res.backward()
                optimizer.step()
                train_loss += res.item()
            train_loss /= len(train_loader.dataset)
            logging.info("epoch: [{:d}/{:d}] {:.2f} sec(s) Train Loss: {:.9f} ".format(
                epoch, config['epoch'], time.time()-epoch_start_time, train_loss))

            if epoch % config['ckpt_freq'] == 0:
                torch.save(FNS.state_dict(), config['checkpoints_folder'] + "/model_"+str(epoch)+".pth")

            if epoch % 20 == 0:
                with torch.no_grad():
                    for i in range(len(test_loader)):
                        iter_num = []
                        for f, coef, u in test_loader[i]:
                            if config["use_cuda"]:
                                f, coef, u = f.cuda(), coef.cuda(), u.cuda()
                            x, ress = FNS.test(f, coef, epoch)  
                            iter_num.append(len(ress[0])-1)
                        logging.info("The average number of iterations is {:.2f} std {:.2f}.".format(np.mean(iter_num), np.std(iter_num)))
    else:
        with torch.no_grad():
            for i in range(len(test_loader)):
                iter_num = []
                timess = []
                for f, coef, u in test_loader[i]:
                    if config["use_cuda"]:
                        f, coef, u = f.cuda(), coef.cuda(), u.cuda()
                    tic= time.time()
                    x, ress = FNS.test(f, coef, 1)
                    # x, ress = FNS.test_v(f, coef, 1)
                    toc= time.time()
                    if ress[0][-1] < 1e-6:
                        iter_num.append(len(ress[0])-1)
                    timess.append(toc-tic)
                logging.info("Grid size {}: Iterations is {:.2f} std {:.2f}. Time is {:.2f} std {:.2f}.".format(f.shape[-1], np.mean(iter_num), np.std(iter_num),np.mean(timess), np.std(timess)))


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using GPU, %s' % torch.cuda.get_device_name(0)) if device == 'cuda' else print('Using CPU')
    setup_seed(12345)

    # *=======================参数设置=================================
    config = {}

    config["pde_num"]          = 10000
    config["every_num"]        = 1
    config["loss"]             = 'u'
    config['batch_size']       = 40
    config["N"]                = 65
    config["ctype"]            = "log"
    config["test_num"]         = 10

    config["smoother"]         = "Jacobi"

    # fns
    config["Meta"]             = "UNet"
    config["s_times"]          = 10
    config["act"]              = "gelu"
    config["channels"]         = [8, 16, 32, 64, 64, 32, 16, 8, 9]
    config["softshrink"]       = 1e-6
    config["modes"]            = [16, 16, 16, 16]
    config["depths"]           = [3, 3, 9, 3]
    config["dims"]             = [32, 32, 32, 32]
    config["drop_path_rate"]   = 0.3
    config["drop"]             = 0.
    config["padding"]          = 9

    # hints
    config["p"]                = 80
    config["input_channel"]    = 1

    config["max_iter_num"]     = 300
    config["error_threshold"]  = 1e-6
    config["xavier_init"]      = 1e-0

    # dir
    config['run_dir'] = "expriments/HINTS"
    config['checkpoints_folder'] = config['run_dir'] + '/checkpoints'
    config['prediction_folder']  = config['run_dir'] + '/prediction'
    config['restart_dir']      = config['checkpoints_folder']

    if device == "cuda":
        config["use_cuda"]     = True
    else:
        config["use_cuda"]     = False

    # train
    config['grad_clip']        = 1.0
    config['lr']               = 1e-4
    config['step_size']        = 1000
    config['gamma']            = 0.5
    config['epoch']            = 10000
    config['ckpt_freq']        = 10
    config['restart_epoch']    = 2150

    # 是否加载预训练模型
    config['restart']          = True
    # config['restart']        = False
    # 是否训练
    config['train']            = False
    # config['train']          = True

    mkdirs([config['run_dir'], config['checkpoints_folder'], config['prediction_folder']])
    # filemode='w',
    logging.basicConfig(filename=config['run_dir'] + "/train.log", format='%(asctime)s %(message)s', level=logging.INFO)
    # *=================================================================
    logging.info('Start training..............................................')
    tic = time.time()
    train(config)
    tic2 = time.time()
    logging.info("Finished training {} epochs using {} seconds"
                 .format(config['epoch'], tic2 - tic))

    logging.info("This time training parameter setting: ")
    with open(config['run_dir'] + "/train.log", 'a') as args_file:json.dump(config, args_file, indent=4)
