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
from testfwi import PCC,MAE,add_colorbar,save_metric_2_csv

def get_initial_data(vec_num, iter_num, prefix = './data/fwidata'):
    rtm_img = scio.loadmat(os.path.join(prefix,'migration',f'migration_{vec_num:04d}_iter{iter_num:04d}.mat'))['v']
    vec_img = scio.loadmat(os.path.join(prefix,'fwivel',f'fwivel_{vec_num:04d}_iter{iter_num:04d}.mat'))['v']
    true_vec = scio.loadmat(os.path.join(prefix,'vel',f'vel_{vec_num:04d}.mat'))['v']

    rtm_img = rtm_img[np.newaxis,:]
    vec_img = vec_img[np.newaxis,:]
    vec_img = (vec_img - 4034) / 514
    # true_vec = (true_vec - 4034) / 514
    rtm_img = rtm_img / 0.0074

    input_data = np.concatenate([rtm_img, vec_img], axis = 0)[np.newaxis]
    return {'A':torch.from_numpy(input_data).type(torch.FloatTensor),'B':torch.from_numpy(true_vec).type(torch.FloatTensor)}

def Savevec_per_iter(vec_img, truevec, rtm_img, vec_num, iter_number, show_fig = False, save_path = None):
    pcc_score = PCC(vec_img, truevec)
    mae_score = MAE(vec_img, truevec)
    
    vec_title = f"vec :{vec_num} run iter:{iter_number} PCC:{pcc_score:.4f} MAE:{mae_score:.4f} data range:{np.min(vec_img)}~{np.max(vec_img)}"
    im_info = {f'rtm data range:{np.min(rtm_img)}~{np.max(rtm_img)}': rtm_img, vec_title: vec_img}

    figure = plt.figure()
    figure.set_size_inches(20,9)
    loc_index = 1
    rows_number = 2
    col_number = 1
    for title,im_data in im_info.items():
        plt.subplot(rows_number, col_number, loc_index, title = title)
        im = plt.imshow(im_data)
        add_colorbar(im)
        loc_index += 1
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight',pad_inches = 0)
    if show_fig:
        plt.show()
    plt.close()
    
    return {"PCC":pcc_score, "MAE":mae_score}

def read_RTM_data_from_file(file_path, rows = 266, cols = 788):
    mig = np.fromfile(file_path, dtype = np.float32).reshape((cols,rows)).T.astype(np.float64)
    return mig

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
    iternum_choose = 10
    total_run_iter = 5
    initial_data = get_initial_data(vec_num,iternum_choose)
    data = initial_data
    true_vec = data['B'].float().numpy()

    #save initial data result
    run_iter = -1
    init_vec = scio.loadmat(os.path.join('./data/fwidata','fwivel',f'fwivel_{vec_num:04d}_iter{iternum_choose:04d}.mat'))['v']
    init_rtm = scio.loadmat(os.path.join('./data/fwidata','migration',f'migration_{vec_num:04d}_iter{iternum_choose:04d}.mat'))['v']
    data_metric = Savevec_per_iter(init_vec, true_vec, init_rtm, vec_num, run_iter, show_fig = opt.show, save_path = os.path.join('result',opt.save_path,'iterationRTMresultFig',f'vec_{vec_num}_iter{run_iter}.png') if opt.savefig else None)
    values_list.append([vec_num, run_iter, *[data_metric[keys] for keys in metrics_values.keys()]])


    for run_iter in range(total_run_iter):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        output_vec = np.squeeze(visuals['fake_B'].cpu().float().numpy())
        output_vec = output_vec[0]

        output_vec_to_bin = output_vec.astype(np.float32)
        output_vec_img = output_vec_to_bin * 514 + 4034
        output_vec_img_reshape = output_vec_img.T.reshape(-1,)
        output_vec_img_reshape.tofile(os.path.join(save_prefix,f'vec{vec_num:04d}_iter{run_iter:04d}.bin'))
        os.system(f'cd RTM && ./tti2d_acousticRTM {vec_num} 0 {run_iter}')
        
        rtm_img = read_RTM_data_from_file(os.path.join(rtm_saveprifix,f'mig_vec{vec_num:04d}_iter{run_iter:04d}.bin'))
        
        data['A'] = torch.from_numpy(np.concatenate([rtm_img[np.newaxis,:] / 0.0074, output_vec[np.newaxis,:]], axis = 0)).type(torch.FloatTensor)[np.newaxis,:]

        data_metric = Savevec_per_iter(output_vec_img, true_vec, rtm_img, vec_num, run_iter, show_fig = opt.show, save_path = os.path.join('result',opt.save_path,'iterationRTMresultFig',f'vec_{vec_num}_iter{run_iter}.png') if opt.savefig else None)
        values_list.append([vec_num, run_iter, *[data_metric[keys] for keys in metrics_values.keys()]])

    save_metric_2_csv(headers, values_list, os.path.join('result',opt.save_path,'iterationRTMresultFig',f'vec_{vec_num}iter_result.csv'))