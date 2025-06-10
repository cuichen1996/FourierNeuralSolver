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
        train_loader = CreateTrainLoader(config, 63)
    test_loader = CreateTestLoader(config, config['N'])

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
                    f, kernelA = f.float().cuda(), kernelA.float().cuda()
                res = FNS(f, kernelA, epoch)
                # res = torch.mean(res)
                optimizer.zero_grad()
                res.backward()
                optimizer.step()
                train_loss += res.item()
            train_loss /= len(train_loader.dataset)
            logging.info("epoch: [{:d}/{:d}] {:.2f} sec(s) Train Loss: {:.9f} ".format(
                epoch, config['epoch'], time.time()-epoch_start_time, train_loss))
            scheduler.step()
            if epoch % config['ckpt_freq'] == 0:
                torch.save(FNS.state_dict(), config['checkpoints_folder'] + "/model_"+str(epoch)+".pth")

            if epoch % 10 == 0:
                FNS.eval()
                with torch.no_grad():
                    for i in range(len(test_loader)):
                        iter_num = []
                        for f, kernelA, u in test_loader[i]:
                            if config["use_cuda"]:
                                f, u, kernelA = f.cuda(), u.cuda(), kernelA.cuda()
                            x, ress = FNS.test(f, kernelA, epoch)
                            iter_num.append(len(ress)-1)
                        logging.info("The average number of iterations is {:.2f} std {:.2f}.".format(np.mean(iter_num), np.std(iter_num)))
    else:
        with torch.no_grad():
            iter_num = []
            timess = []
            FNS.eval()
            for f, kernelA, u in test_loader:
                print(kernelA)
                if config["use_cuda"]:
                    f, u, kernelA = f.cuda(), u.cuda(), kernelA.cuda()
                tic= time.time()
                x, ress = FNS.test(f, kernelA, 1)
                toc= time.time()
                rerror = torch.norm(x - u) / torch.norm(u)
                iter_num.append(len(ress)-1)
                timess.append(toc-tic)
                logging.info("The relative error is {:.9f} after running FNS {:d} iterations.".format(rerror,len(ress)))
            print(iter_num)
            logging.info("The average number of iterations is {:.2f} std {:.2f}.".format(np.mean(iter_num), np.std(iter_num)))
            logging.info("The average time is {:.2f} std {:.2f}.".format(np.mean(timess), np.std(timess)))
            print("The average number of iterations is {:.2f} std {:.2f}.".format(np.mean(iter_num), np.std(iter_num)))

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using GPU, %s' % torch.cuda.get_device_name(0)) if device == 'cuda' else print('Using CPU')
    setup_seed(1234)

    # *=======================参数设置=================================
    config = {}
    
    # data
    config["pde_num"]       = 1000
    config["every_num"]     = 10
    config['batch_size']    = 20
    config["N"]             = 63
    config["test_epsilons"] = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    config["test_theta"]    = [0.8]*len(config["test_epsilons"])
    config["ddtype"]        = "supg" # "supg" or "fem"

    config["smoother"] = "cnn" # "jacobi" or "cnn"

    # model
    config["kernel_size"]    = 3
    config["act"]            = "gelu"

    # config["Meta_Type"]      = "sFNO" 
    config["Meta_Type"]      = "UNet"   
    config["modes"]          = [16,16,16,16]
    config["depths"]         = [3,3,9,3]
    config["dims"]           = [32,32,32,32]
    config["drop_path_rate"] = 0.3
    config["drop"]           = 0.
    config["padding"]        = 9

    config["channels"]   = [8, 16, 32, 64, 64, 32, 16, 8, 9]

    config["max_iter_num"]    = 1000
    config["error_threshold"] = 1e-6
    config["xavier_init"]     = 1e-2
    
    if device == "cuda":
        config["use_cuda"] = True
    else:
        config["use_cuda"] = False
        
    # dir
    config['run_dir'] = "expriments/FNS_"+config["ddtype"]+"_" + config["smoother"]
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
    config['restart_epoch'] = 960
    
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
