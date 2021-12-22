import os
import io

import matplotlib as mpl
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
from numba import njit,jit
import time

from mpl_toolkits import axes_grid1

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def compare_rtm_and_sth_pcc(visuals, savepath = None, opt = None, vec_num = None, run_iter = None):
    rtm_img = np.squeeze(visuals['real_A'][:,0,:,:].cpu().float().numpy())
    itervec_img = np.squeeze(visuals['real_A'][:,1,:,:].cpu().float().numpy())
    truevec = np.squeeze(visuals['real_B'].cpu().float().numpy())
    output_vec = np.squeeze(visuals['fake_B'].cpu().float().numpy())

    if opt.output_nc == 3:
        grad_x = output_vec[1]
        grad_y = output_vec[2]
        output_vec = output_vec[0]

    if opt.input_nc == 3:
        born_img = np.squeeze(visuals['real_A'][:,2,:,:].cpu().float().numpy())
    
    vec_true_vec = truevec[0]
    vecoriginal_PCC = PCC(itervec_img,vec_true_vec)
    vecoriginal_MAE = MAE(itervec_img,vec_true_vec)
    rtm_PCC = PCC(rtm_img,vec_true_vec)
    rtm_MAE = MAE(rtm_img,vec_true_vec)
    # print(f'{np.min(rtm_img)}~{np.max(rtm_img)}')
    # print(f'{np.min(output_vec)}~{np.max(output_vec)}')

    min_val,max_val = find_min_max([itervec_img, truevec, output_vec])
    vnorm = mpl.colors.Normalize(vmin = min_val, vmax=max_val)

    title_list = {\
        f'rtm image rtm pcc:{rtm_PCC}, rtm mae:{rtm_MAE}':rtm_img,f'iter vec image PCC:{vecoriginal_PCC:.4f},MAE:{vecoriginal_MAE:.4f}':itervec_img,\
        f'output vec':output_vec,f'vec true vec {vec_num}':vec_true_vec}

    rows_number = 2
    col_number = 2

    figure = plt.figure()
    figure.set_size_inches(20, 9)
    loc_index = 1
    for title,im_data in title_list.items():
        if 'output vec' in title and vec_num is not None and run_iter is not None:
            title += ' No %4d iter %4d' % (vec_num, run_iter)
        plt.subplot(rows_number, col_number, loc_index, title = title)
        plt.grid(False)
        if im_data.shape.__len__() == 3:
            im_data = im_data.transpose(1,2,0)
        if 'vec' in title:
            im = plt.imshow(im_data, norm=vnorm)
        else:    
            im = plt.imshow(im_data)
        loc_index += 1
        add_colorbar(im)
    
    plt.tight_layout()
    if opt is not None:
        if opt.show:
            plt.show()

    if savepath is not None and opt.savefig:
        print(f'save img at {savepath}')
        plt.savefig(savepath, bbox_inches='tight',pad_inches = 0)

    plt.close()

    return {'original PCC':vecoriginal_PCC,'originial MAE':vecoriginal_MAE,'rtm PCC': rtm_PCC, 'rtm MAE': rtm_MAE}

def PCC(img1,img2):
    m1 = np.mean(img1)
    m2 = np.mean(img2)
    diffimg1 = img1 - m1
    diffimg2 = img2 - m2
    PCCnum = np.sum(diffimg1 * diffimg2) / np.sqrt(np.sum(diffimg1**2)*np.sum(diffimg2**2))
    return PCCnum

def MAE(img1,img2):
    return np.average(np.abs(img1-img2))

def find_min_max(list_data):
    min_val = []
    max_val = []
    for item in list_data:
        min_val.append(np.min(item))
        max_val.append(np.max(item))
    return np.min(min_val),np.max(max_val)

@jit(nopython=True)
def patch_pcc_and_mae(img1, img2, rows, cols):
    pccres = np.zeros((rows,cols))
    mae_res = np.zeros((rows,cols))
    for rownum in range(rows):
        for colnum in range(cols):
            m1 = np.mean(img1[rownum,colnum])
            m2 = np.mean(img2[rownum,colnum])
            diffimg1 = img1[rownum,colnum] - m1
            diffimg2 = img2[rownum,colnum] - m2
            pccres[rownum, colnum] = np.sum(diffimg1 * diffimg2) / (np.sqrt(np.sum(diffimg1**2)*np.sum(diffimg2**2)))
            mae_res[rownum, colnum] = np.mean(np.abs(img1[rownum,colnum] - img2[rownum,colnum]))
    return pccres, mae_res

@jit(nopython=True)
def patch_snr_and_mae(img1, img2, rows, cols):
    snrres = np.zeros((rows,cols))
    mae_res = np.zeros((rows,cols))
    for rownum in range(rows):
        for colnum in range(cols):
            snrres[rownum, colnum] = 10*np.log10(np.sum(img1[rownum,colnum]**2)/np.sum((img1[rownum,colnum]-img2[rownum,colnum])**2))
            mae_res[rownum, colnum] = np.mean(np.abs(img1[rownum,colnum] - img2[rownum,colnum]))
    return snrres, mae_res

def compute_patch_pcc(img1, img2, window_size = 5, compute_type = 'snr'):
    # print(img1.shape,img2.shape)
    assert img1.shape == img2.shape and window_size % 2 == 1
    rows,cols = img1.shape
    pad_width = window_size // 2 
    img1 = np.pad(img1, pad_width = pad_width, mode = 'constant')
    img2 = np.pad(img2, pad_width = pad_width, mode = 'constant')
    # out_rows = ( rows - window_size + 2 * pad_width ) + 1
    # out_cols = ( cols - window_size + 2 * pad_width ) + 1
    # img1_patch = np.lib.stride_tricks.as_strided(img1, shape = (out_rows,out_cols,window_size,window_size), strides = (*img1.strides,*img1.strides)) # or,oc,ws,ws
    # img2_patch = np.lib.stride_tricks.as_strided(img2, shape = (out_rows,out_cols,window_size,window_size), strides = (*img2.strides,*img2.strides))
    img1 = np.lib.stride_tricks.sliding_window_view(img1,(window_size,window_size))
    img2 = np.lib.stride_tricks.sliding_window_view(img2,(window_size,window_size))
    assert img1.shape[:2] == (rows,cols)
    if compute_type == 'snr':
        pccres,maeres = patch_snr_and_mae(img1, img2, rows, cols)
    elif compute_type == 'pcc':
        pccres,maeres = patch_pcc_and_mae(img1, img2, rows, cols)
    else:
        raise ValueError('unkown compute type')
    return pccres,maeres

def compute_fft_mae_and_pcc(visuals, opt = None):
    rtm_img = np.squeeze(visuals['real_A'][:,0,:,:].cpu().float().numpy())
    itervec_img = np.squeeze(visuals['real_A'][:,1,:,:].cpu().float().numpy())
    truevec = np.squeeze(visuals['real_B'].cpu().float().numpy())
    output_vec = np.squeeze(visuals['fake_B'].cpu().float().numpy())
    grad_x = None
    grad_y = None
    if opt.output_nc == 3:
        grad_x = output_vec[1]
        grad_y = output_vec[2]
        output_vec = output_vec[0]
    if len(truevec) == 3:
        truevec = truevec[0]

    if opt.input_nc == 3:
        born_img = np.squeeze(visuals['real_A'][:,2,:,:].cpu().float().numpy())
        if opt.normalize_method == 'zscore':
            born_img = born_img * 0.0714
        elif opt.normalize_method == 'minmax':
            born_img = born_img   
        else:
            raise ValueError('unkown normalize method')

    if opt.normalize_method == 'zscore':
        itervec_img = itervec_img * 514 + 4034
        truevec = truevec * 514 + 4034
        output_vec = output_vec * 514 + 4034
        rtm_img = rtm_img * 0.0074
    elif opt.normalize_method == 'minmax':
        itervec_img = itervec_img * 6605.074823646438
        truevec = truevec * 6605.074823646438
        output_vec = output_vec * 6605.074823646438
        rtm_img = rtm_img *  0.7502193450927734

    itervec_img = np.fft.fftshift(np.fft.fft2(itervec_img))
    itervec_img = np.log(np.abs(itervec_img))
    output_vec = np.fft.fftshift(np.fft.fft2(output_vec))
    output_vec = np.log(np.abs(output_vec))
    truevec = np.fft.fftshift(np.fft.fft2(truevec))
    truevec = np.log(np.abs(truevec))
 
    original_PCC = PCC(itervec_img,truevec)
    original_MAE = MAE(itervec_img,truevec)
    output_PCC = PCC(output_vec,truevec)
    output_MAE = MAE(output_vec,truevec)

    return {'original PCC':original_PCC,'originial MAE':original_MAE,'output PCC':output_PCC,'output MAE':output_MAE}


def plot_patch_pcc(visuals, savepath = None, opt = None, vec_num = None, run_iter = None, window_size = 15, compute_type = 'pcc'):

    rtm_img = np.squeeze(visuals['real_A'][:,0,:,:].cpu().float().numpy())
    itervec_img = np.squeeze(visuals['real_A'][:,1,:,:].cpu().float().numpy())
    truevec = np.squeeze(visuals['real_B'].cpu().float().numpy())
    output_vec = np.squeeze(visuals['fake_B'].cpu().float().numpy())
    grad_x = None
    grad_y = None
    
    if opt.output_nc == 3:
        grad_x = output_vec[1]
        grad_y = output_vec[2]
        output_vec = output_vec[0]

    if len(truevec) == 3:
        truevec = truevec[0]

    if opt.input_nc == 3:
        born_img = np.squeeze(visuals['real_A'][:,2,:,:].cpu().float().numpy())
        if opt.normalize_method == 'zscore':
            born_img = born_img * 0.0714
        elif opt.normalize_method == 'minmax':
            born_img = born_img   
        else:
            raise ValueError('unkown normalize method')

    if opt.normalize_method == 'zscore':
        itervec_img = itervec_img * 514 + 4034
        truevec = truevec * 514 + 4034
        output_vec = output_vec * 514 + 4034
        rtm_img = rtm_img * 0.0074
    elif opt.normalize_method == 'minmax':
        itervec_img = itervec_img * 6605.074823646438
        truevec = truevec * 6605.074823646438
        output_vec = output_vec * 6605.074823646438
        rtm_img = rtm_img *  0.7502193450927734
    
    # rtm_true_vec = truevec[1]
    # vec_true_vec = truevec[0]
    vec_true_vec= truevec
    rtm_true_vec = vec_true_vec
    
    original_PCC_patch, original_mae_patch = compute_patch_pcc(itervec_img, vec_true_vec, window_size = window_size, compute_type = compute_type )
    output_PCC_patch, output_mae_patch = compute_patch_pcc(output_vec, vec_true_vec, window_size = window_size, compute_type = compute_type )
    rtm_PCC_patch, rtm_mae_patch = compute_patch_pcc(output_vec, rtm_true_vec, window_size = window_size, compute_type = compute_type )

    vecoriginal_PCC = PCC(itervec_img,vec_true_vec)
    vecoriginal_MAE = MAE(itervec_img,vec_true_vec)
    vecoutput_PCC = PCC(output_vec,vec_true_vec)
    vecoutput_MAE = MAE(output_vec,vec_true_vec)

    rtmoutput_PCC = PCC(output_vec,rtm_true_vec)
    rtmoutput_MAE = MAE(output_vec,rtm_true_vec)

    min_val,max_val = find_min_max([itervec_img, truevec, output_vec])
    vec_vnorm = mpl.colors.Normalize(vmin = min_val, vmax=max_val)
    min_val,max_val = find_min_max([original_PCC_patch, output_PCC_patch, rtm_PCC_patch])
    pcc_vnorm = mpl.colors.Normalize(vmin = min_val, vmax=max_val)
    min_val,max_val = find_min_max([original_mae_patch, output_mae_patch, rtm_mae_patch])
    mae_vnorm = mpl.colors.Normalize(vmin = min_val, vmax=max_val)

    title_list = {'rtm image':rtm_img,'true_vec':truevec, 'grad': grad_x, \
        f'iter vec image PCC:{vecoriginal_PCC:.4f},MAE:{vecoriginal_MAE:.4f}':itervec_img, f'original patch {compute_type}': original_PCC_patch, 'original patch mae': original_mae_patch,\
        f'output vec PCC:{vecoutput_PCC:.4f},MAE:{vecoutput_MAE:.4f}':output_vec, \
        f'output patch {compute_type}':output_PCC_patch, 'output patch mae': output_mae_patch}
    
    # title_list = {'rtm image':rtm_img,f'iter vec image PCC:{vecoriginal_PCC:.4f},MAE:{vecoriginal_MAE:.4f}':itervec_img, f'original patch {compute_type}': original_PCC_patch, \
    #     f'output vec PCC:{vecoutput_PCC:.4f},MAE:{vecoutput_MAE:.4f}':output_vec, f'output patch {compute_type}':output_PCC_patch, 'output patch mae': output_mae_patch, \
    #     f'rtm true vec rtm PCC:{rtmoutput_PCC}, rtm MAE:{rtmoutput_MAE}':rtm_true_vec, f'output rtm patch {compute_type}': rtm_PCC_patch, 'output rtm patch mae': rtm_mae_patch}
    
    rows_number = 3
    col_number = 3
    figure = plt.figure()
    figure.set_size_inches(20, 9)
    loc_index = 1
    for title,im_data in title_list.items():
        if 'output vec' in title and vec_num is not None and run_iter is not None:
            title += ' No %4d iter %4d' % (vec_num, run_iter)
        plt.subplot(rows_number, col_number, loc_index)
        plt.grid(False)
        if im_data.shape.__len__() == 3:
            im_data = im_data.transpose(1,2,0)
        if 'vec' in title:
            im = plt.imshow(im_data, norm = vec_vnorm, cmap = plt.cm.gray)
        elif f'patch {compute_type}' in title:
            im = plt.imshow(im_data,cmap=plt.cm.gray, norm = pcc_vnorm)
        elif 'patch mae' in title:
            im = plt.imshow(im_data,cmap=plt.cm.gray, norm = mae_vnorm)
        elif 'rtm image' in title:
            im = plt.imshow(im_data,cmap=plt.cm.gray)
        else:
            im = plt.imshow(im_data)
        loc_index += 1
        add_colorbar(im)
    plt.tight_layout()
    if opt is not None:
        if opt.show:
            plt.show()
    if savepath is not None and opt.savefig:
        print(f'save img at {savepath}')
        plt.savefig(savepath, bbox_inches='tight',pad_inches = 0)
    plt.close()

def plot_image(visuals, savepath = None, opt = None, vec_num = None, run_iter = None, show_grad = False, show_residual = False):
    rtm_img = np.squeeze(visuals['real_A'][:,0,:,:].cpu().float().numpy())
    itervec_img = np.squeeze(visuals['real_A'][:,1,:,:].cpu().float().numpy())
    truevec = np.squeeze(visuals['real_B'].cpu().float().numpy())
    output_vec = np.squeeze(visuals['fake_B'].cpu().float().numpy())
    rtm_vec_num = 170

    if opt.output_nc == 3:
        grad_x = output_vec[1]
        grad_y = output_vec[2]
        output_vec = output_vec[0]
        # truevec = truevec[0]
    # output_vec_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(output_vec))))

    if opt.input_nc == 3:
        born_img = np.squeeze(visuals['real_A'][:,2,:,:].cpu().float().numpy())
        if opt.normalize_method == 'zscore':
            born_img = born_img * 0.0714
        elif opt.normalize_method == 'minmax':
            born_img = born_img   
        else:
            raise ValueError('unkown normalize method')

    if opt.normalize_method == 'zscore':
        itervec_img = itervec_img * 514 + 4034
        truevec = truevec * 514 + 4034
        output_vec = output_vec * 514 + 4034
        rtm_img = rtm_img * 0.0074

    elif opt.normalize_method == 'minmax':
        itervec_img = itervec_img * 6605.074823646438
        truevec = truevec * 6605.074823646438
        output_vec = output_vec * 6605.074823646438
        rtm_img = rtm_img *  0.7502193450927734
    
    rtm_true_vec = truevec[1]
    vec_true_vec = truevec[0]
    
    vecoriginal_PCC = PCC(itervec_img,vec_true_vec)
    vecoriginal_MAE = MAE(itervec_img,vec_true_vec)
    vecoutput_PCC = PCC(output_vec,vec_true_vec)
    vecoutput_MAE = MAE(output_vec,vec_true_vec)

    # rtmoriginal_PCC = PCC(itervec_img,vec_true_vec)
    # rtmoriginal_MAE = MAE(itervec_img,vec_true_vec)
    rtmoutput_PCC = PCC(output_vec,rtm_true_vec)
    rtmoutput_MAE = MAE(output_vec,rtm_true_vec)


    min_val,max_val = find_min_max([itervec_img, truevec, output_vec])
    vnorm = mpl.colors.Normalize(vmin = min_val, vmax=max_val)

    title_list = {\
        f'rtm image velocity number {rtm_vec_num}':rtm_img,f'iter vec image PCC:{vecoriginal_PCC:.4f},MAE:{vecoriginal_MAE:.4f}':itervec_img,\
        f'output vec PCC:{vecoutput_PCC:.4f},MAE:{vecoutput_MAE:.4f}':output_vec,\
        f'vec true vec {vec_num}':vec_true_vec,f'rtm true vec {rtm_vec_num} PCC:{rtmoutput_PCC:.4f}, MAE:{rtmoutput_MAE:.4f}': rtm_true_vec} if not opt.input_nc == 3 else {\
        'rtm image':rtm_img,f'iter vec image PCC:{vecoriginal_PCC:.4f},MAE:{vecoriginal_MAE:.4f}':itervec_img,\
        'vec true_vec':vec_true_vec, f'output vec PCC:{vecoutput_PCC:.4f},MAE:{vecoutput_MAE:.4f}':output_vec,\
        'born image':born_img}
    
    # if show_grad:
    #     title_list.update({'grad x':grad_x,'grad y':grad_y})
    
    if show_residual:
        title_list.pop('rtm image')
        title_list.update({'residual': truevec - output_vec})

    if opt.input_nc == 3 and show_grad:
        rows_number = 3
        col_number = 3
    elif opt.input_nc == 3 and not show_grad:
        rows_number = 3
        col_number = 2
    elif not opt.input_nc == 3 and show_grad:
        rows_number = 3
        col_number = 2
    else:
        rows_number = 2
        col_number = 2

    figure = plt.figure()
    figure.set_size_inches(20, 9)
    loc_index = 1
    for title,im_data in title_list.items():
        if 'output vec' in title and vec_num is not None and run_iter is not None:
            title += ' No %4d iter %4d' % (vec_num, run_iter)
        plt.subplot(rows_number, col_number, loc_index, title = title)
        plt.grid(False)
        if im_data.shape.__len__() == 3:
            im_data = im_data.transpose(1,2,0)
        if 'vec' in title:
            im = plt.imshow(im_data, norm=vnorm, cmap = plt.cm.gray)
        elif 'rtm' in title:
            wstd = np.std(im_data)
            rtm_norm = mpl.colors.Normalize(vmin = -2.0*wstd, vmax = 2.0*wstd)
            im = plt.imshow(im_data, norm = rtm_norm, cmap = plt.cm.gray)
        else:
            im = plt.imshow(im_data, cmap = plt.cm.gray)
        loc_index += 1
        # plt.colorbar(fraction=0.046, pad=0.04)
        add_colorbar(im)
    
    plt.tight_layout()
    # plt.get_current_fig_manager().window.state('withdraw')
    if opt is not None:
        if opt.show:
            plt.show()

    if savepath is not None and opt.savefig:
        print(f'save img at {savepath}')
        plt.savefig(savepath, bbox_inches='tight',pad_inches = 0)

    plt.close()

    return {'original PCC':vecoriginal_PCC,'originial MAE':vecoriginal_MAE,'output PCC':vecoutput_PCC,'output MAE':vecoutput_MAE, 'rtm output PCC':rtmoutput_PCC, 'rtm output MAE': rtmoutput_MAE}

def savedata(visuals, savepath):
    output_vec = np.squeeze(visuals['fake_B'].cpu().float().numpy())
    np.save(savepath, output_vec)

def save_metric_2_csv(headers, values, savepath):
    with open(savepath,'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(values)

def plot_single_image(visuals, savepath = None):
    rtm_img = np.squeeze(visuals['real_A'][:,0,:,:].cpu().float().numpy())
    itervec_img = np.squeeze(visuals['real_A'][:,1,:,:].cpu().float().numpy())
    truevec = np.squeeze(visuals['real_B'].cpu().float().numpy())
    output_vec = np.squeeze(visuals['fake_B'].cpu().float().numpy())
    print(f'out vec shape{output_vec.shape}')
    title_list = {'rtm image':rtm_img,'iter vec image':itervec_img,'true_vec':truevec,'output vec':output_vec}
    for title,im_data in title_list.items():
        # plt.subplot(2, 2, loc_index, title = title)
        plt.imshow(im_data)
        plt.grid(False)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        # loc_index += 1
        plt.colorbar()
        plt.show()
        plt.close()

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    datamode = 'test'
    prefix_index = {'train':0,'val':150,'test':170}
    iternumber = int(opt.iternumber) if opt.iternumber.isdigit() else opt.iternumber
    use_cus = False

    metrics_values = {'original PCC':[],'originial MAE':[],'output PCC':[],'output MAE':[], 'rtm output PCC':[], 'rtm output MAE': []}

    # test_loader = get_customized_dataset(prefix = './data/fwidata')
    datamode = 'test'
    prefix_index = {'train':0,'val':150,'test':170}

    test_loader = get_testloader(opt, prefix = './data/fwidata', mode = datamode, iternumber = iternumber)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create 

    if opt.eval:
        model.eval()
    if opt.savefig:
        os.makedirs(os.path.join('result',opt.save_path,'rtm_another_result_fig'), exist_ok = True)
    if opt.compute_patch:
        os.makedirs(os.path.join('result',opt.save_path,'compare_patch_result_fig'), exist_ok = True)

    headers = ['vec num', 'run iter', *metrics_values.keys()]
    values_list = []
    vec_num = 174
    run_iter = 10

    for i, data in enumerate(test_loader):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
      
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        if use_cus:
            run_iter = 10 + 10 * i
        else:
            vec_num = i // 30 + prefix_index[datamode] # which vec model
            run_iter = (i % 30) * 10 + 10

        # if opt.compute_fft:
        #     single_data_metric = compute_fft_mae_and_pcc(visuals, opt = opt)
        #     values_list.append([vec_num, run_iter, *[single_data_metric[keys] for keys in metrics_values.keys()]])
        #     for key,values in single_data_metric.items():
        #         metrics_values[key].append(values)
                
        if not opt.compute_patch:
            # single_data_metric = plot_image(visuals, opt = opt, savepath = os.path.join('result',opt.save_path,'rtm_another_result_fig','No_'+str(vec_num)+'iter'+str(run_iter)+".png"), vec_num = vec_num, run_iter = run_iter, show_grad = opt.multi_task, show_residual=opt.show_residual)
            single_data_metric = compare_rtm_and_sth_pcc(visuals, opt = opt, savepath = None, vec_num = vec_num, run_iter = run_iter)
            values_list.append([vec_num, run_iter, *[single_data_metric[keys] for keys in metrics_values.keys()]])
            for key,values in single_data_metric.items():
                metrics_values[key].append(values)
        else:
            plot_patch_pcc(visuals, opt = opt, savepath = os.path.join('result',opt.save_path,'compare_patch_result_fig','No_'+str(vec_num)+'iter'+str(run_iter)+".png"), vec_num = vec_num, run_iter = run_iter, compute_type='pcc', window_size = 31)

    if not opt.compute_patch:
        for key,values in metrics_values.items():
            metrics_values[key] = np.mean(metrics_values[key])

        with open(os.path.join('result',opt.save_path, opt.epoch+'_rtm_another_metric_result.txt'),'w') as f:
            f.write(str(metrics_values))

        save_metric_2_csv(headers, values_list, savepath = os.path.join('result',opt.save_path, opt.epoch+f'_rtm_another_metric_result_vec{vec_num}.csv'))