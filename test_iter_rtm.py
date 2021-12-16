import os
import io
import matplotlib as mpl
import torch
from numba.np.ufunc import parallel
from options.test_options import TestOptions
from data import create_dataset,fwi_dataset
from data.fwi_dataset import get_testloader,get_customized_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import wandb
from torch.utils.tensorboard import SummaryWriter
from matplotlib import cm, pyplot as plt
import numpy as np
import csv
import scipy.io as scio
from numba import njit,jit
import time
from mpl_toolkits import axes_grid1

def get_initial_data(vec_num, iter_num, prefix = './data/fwidata'):
    rtm_img = scio.loadmat(os.path.join(prefix,'migration',f'migration_{vec_num:04d}_iter{iter_num:04d}.mat'))['v']
    vec_img = scio.loadmat(os.path.join(prefix,'fwivel',f'fwivel_{vec_num:04d}_iter{iter_num:04d}.mat'))['v']
    true_vec = scio.loadmat(os.path.join(prefix,'vel',f'vel_{vec_num:04d}.mat'))['v']

    rtm_img = rtm_img[np.newaxis,:]
    vec_img = vec_img[np.newaxis,:]

    vec_img = (vec_img - 4034) / 514
    true_vec = (true_vec - 4034) / 514
    rtm_img = rtm_img / 0.0074


    input_data = np.concatenate([rtm_img, vec_img], axis = 0)[np.newaxis]
    return {'A':torch.from_numpy(input_data).type(torch.FloatTensor),'B':torch.from_numpy(true_vec).type(torch.FloatTensor)}


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    datamode = 'test'
    prefix_index = {'train':0,'val':150,'test':170}
    iternumber = int(opt.iternumber) if opt.iternumber.isdigit() else opt.iternumber

    use_cus = False
    metrics_values = {'PCC':[],'MAE':[]}

    test_loader = get_testloader(opt, prefix = './data/fwidata', mode = datamode, iternumber = iternumber)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create 

    if opt.eval:
        model.eval()

    save_prefix = './data/fwidata/tempoutputvec'
    rtm_saveprifix = './data/fwidata/tempoutputrtm'

    if opt.savefig:
        os.makedirs(os.path.join('result',opt.save_path,'iterationRTMresultFig'), exist_ok = True)


    headers = ['vec num', 'run iter', *metrics_values.keys()]
    values_list = []
    vec_num = 170
    run_iter = 0
    total_run_iter = 1
    initial_data = get_initial_data(170,10)
    data = initial_data
    true_vec = data['B']

    for run_iter in range(total_run_iter):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        output_vec = np.squeeze(visuals['fake_B'].cpu().float().numpy()).astype(np.float32)
        output_vec = output_vec.T.reshape(-1,)
        output_vec.tofile(os.path.join(save_prefix,f'vec{vec_num:04d}_iter{run_iter:04d}.bin'))