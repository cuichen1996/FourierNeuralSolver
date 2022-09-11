import json
import logging
import os

import numpy as np
import torch

from data.data import CreateMetaDatasetLoader, CreateTestLoader
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
    optimizer = torch.optim.Adam(list(FNS.parameters()), lr=config['lr'])

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
                # torch.nn.utils.clip_grad_value_(FNS.parameters(), config['grad_clip'])
                optimizer.step()
                train_loss += res.item() * x.shape[0]
            train_loss /= len(train_loader.dataset)
            logging.info("epoch: [{:d}/{:d}] {:.2f} sec(s) Train Loss: {:.9f} ".format(
                epoch, config['epoch'], time.time()-epoch_start_time, train_loss))
            if epoch % config['ckpt_freq'] == 0:
                torch.save(FNS.state_dict(), config['checkpoints_folder'] + "/model_"+str(epoch)+".pth")
    else:
        FNS.eval()
        with torch.no_grad():
            for f, kernelA, u in test_loader:
                x = torch.zeros_like(f)
                print(kernelA)
                if config["use_cuda"]:
                    x, f, u, kernelA = x.cuda(), f.cuda(), u.cuda(), kernelA.cuda()
                rerror, num_iter = FNS.test(x, f, kernelA, u)
                # rerror = torch.norm(x - u) / torch.norm(u)
                logging.info("The relative error is {:.9f} after running FNS {:d} iterations.".format(rerror,num_iter))

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print('Using GPU, %s' % torch.cuda.get_device_name(0)) if device == 'cuda' else print('Using CPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.set_default_dtype(torch.float64)
    torch.autograd.set_detect_anomaly(True)
    setup_seed(1111)
    
    # *=======================参数设置=================================
    config = {}
    
    # data
    config["pde_num"]       = 2000
    config["every_num"]     = 1
    config["loss"]          = 'r' # 'u' or 'r'
    config['batch_size']    = 100
    config["N"]             = 63
    config["test_epsilons"] = [1e-6]
    config["a"]             = 1
    config["b"]             = 1

    # model
    config["smoother_times"]  = 1
    config["Jacobi_weight"]   = 4/5
    config["alpha"]           = 1/3
    config["niter"]           = 1
    config["mL"]              = 3
    config["kernel_size"]     = 5
    
    config["mid_chanel"]      = 4
    config["act"]             = "prelu"
    config["max_iter_num"]    = 4000
    config["error_threshold"] = 1e-6
    config["K"]               = 100
    config["xavier_init"]     = 1e-2
    if device == "cuda":
        config["use_cuda"] = True
    else:
        config["use_cuda"] = False
        
    # dir
    config['run_dir'] = "/home/kaijiang/cuichen/FNSsummer/0817/expriments/mL_{}KS_{}".format(config["mL"], config["kernel_size"])
    config['checkpoints_folder'] = config['run_dir'] + '/checkpoints'
    config['prediction_folder'] = config['run_dir'] + '/prediction'
    config['restart_dir'] = '/home/kaijiang/cuichen/FNSsummer/0817/expriments/mL_3KS_5/checkpoints'

    # train
    config['grad_clip']     = 1.0
    config['lr']            = 1e-4
    config['epoch']         = 20000
    config['ckpt_freq']     = 10
    config['restart_epoch'] = 10
    
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
