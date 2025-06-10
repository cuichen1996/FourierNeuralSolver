import json
import logging

import model.FNS as solver
import numpy as np
import torch
from data.load_data import CreateTrainLoader, CreateTestLoader
from model.misc import *

def train(config):
    # * data
    if config['train']:
        train_loader = CreateTrainLoader(config)
    test_loader = CreateTestLoader(config["N"])

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
            for f, kappa, u in train_loader:
                if config["use_cuda"]:
                    f, kappa, u = f.cfloat().cuda(), kappa.float().cuda(), u.cfloat().cuda()
                res = FNS(f, kappa, u, epoch)
                optimizer.zero_grad()
                res.backward()
                optimizer.step()
                train_loss += (res.item()*f.shape[0])
            train_loss /= len(train_loader.dataset)
            logging.info("epoch: [{:d}/{:d}] {:.2f} sec(s) Train Loss: {:.9f} ".format(
                epoch, config['epoch'], time.time()-epoch_start_time, train_loss))
            if epoch % config['ckpt_freq'] == 0:
                torch.save(FNS.state_dict(), config['checkpoints_folder'] + "/model_"+str(epoch)+".pth")

            if epoch % 10 == 0:
                with torch.no_grad():
                    for f, kappa, u in test_loader:
                        if config["use_cuda"]:
                            f, kappa, u = f.cfloat().cuda(), kappa.float().cuda(), u.cfloat().cuda()
                        x, ress = FNS.test(f, kappa, epoch)
    else:
        FNS.eval()
        with torch.no_grad():
            iter_num = []
            timess = []
            for f, kappa, u in test_loader:
                if config["use_cuda"]:
                    f, kappa, u = f.cfloat().cuda(), kappa.float().cuda(), u.cfloat().cuda()
                tic= time.time()
                x, ress = FNS.test(f, kappa, 1)
                toc= time.time()
                rerror = torch.norm(x - u) / torch.norm(u)

                timess.append(toc-tic)
                logging.info("The relative error is {:.9f} after running FNS {:d} iterations.".format(rerror,len(ress[0])-1))

            logging.info("The average number of iterations is {:.2f} std {:.2f}.".format(np.mean(iter_num), np.std(iter_num)))
            logging.info("The average time is {:.2f} std {:.2f}.".format(np.mean(timess), np.std(timess)))
            print("The average number of iterations is {:.2f} std {:.2f}.".format(np.mean(iter_num), np.std(iter_num)))

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using GPU, %s' % torch.cuda.get_device_name(0)) if device == 'cuda' else print('Using CPU')
    setup_seed(1234)

    # *=======================参数设置=================================
    config = {}

    config['batch_size']    = 20
    config["N"]             = 127

    # model
    config["smoother"]       = "Jacobi"
    config["act"]            = "gelu"
    config["Meta_Type"]      = "sFNO"  
    config["modes"]          = [16,16,16,16]
    config["depths"]         = [3,3,9,3]
    config["dims"]           = [32,32,32,32]
    config["drop_path_rate"] = 0.3
    config["drop"]           = 0.
    config["padding"]        = 9

    config["channels"]   = [8, 16, 32, 64, 64, 32, 16, 8, 9]

    config["max_iter_num"]    = 500
    config["error_threshold"] = 1e-6
    config["xavier_init"]     = 1e-1

    # dir
    config['run_dir'] = "expriments/FNS_robin"
    config['checkpoints_folder'] = config['run_dir'] + '/checkpoints'
    config['prediction_folder'] = config['run_dir'] + '/prediction'
    config['restart_dir'] = config['checkpoints_folder']
    
    if device == "cuda":
        config["use_cuda"] = True
    else:
        config["use_cuda"] = False

    # train
    config['grad_clip']     = 1.0
    config['lr']            = 1e-4
    config['step_size']     = 1000
    config['gamma']         = 0.5
    config['epoch']         = 10000
    config['ckpt_freq']     = 10
    config['restart_epoch'] = 190
    
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
