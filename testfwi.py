import os
import io

import matplotlib as mpl
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

def PCC(img1,img2):
    m1 = np.mean(img1)
    m2 = np.mean(img2)
    diffimg1 = img1 - m1
    diffimg2 = img2 - m2
    PCC = np.sum(diffimg1 * diffimg2) / np.sqrt(np.sum(diffimg1**2)*np.sum(diffimg2**2))
    return PCC

def MAE(img1,img2):
    return np.average(np.abs(img1-img2))

def find_min_max(list_data):
    min_val = []
    max_val = []
    for item in list_data:
        min_val.append(np.min(item))
        max_val.append(np.max(item))
    return np.min(min_val),np.max(max_val)

def plot_image(visuals, savepath = None, opt = None, vec_num = None, run_iter = None, show_grad = False, show_residual = False):
    rtm_img = np.squeeze(visuals['real_A'][:,0,:,:].cpu().float().numpy())
    itervec_img = np.squeeze(visuals['real_A'][:,1,:,:].cpu().float().numpy())
    truevec = np.squeeze(visuals['real_B'].cpu().float().numpy())
    output_vec = np.squeeze(visuals['fake_B'].cpu().float().numpy())
    
    if opt.output_nc == 3:
        grad_x = output_vec[1]
        grad_y = output_vec[2]
        output_vec = output_vec[0]
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
    
    original_PCC = PCC(itervec_img,truevec)
    original_MAE = MAE(itervec_img,truevec)
    output_PCC = PCC(output_vec,truevec)
    output_MAE = MAE(output_vec,truevec)

    min_val,max_val = find_min_max([itervec_img, truevec, output_vec])
    vnorm = mpl.colors.Normalize(vmin = min_val, vmax=max_val)

    title_list = {'rtm image':rtm_img,f'iter vec image PCC:{original_PCC:.4f},MAE:{original_MAE:.4f}':itervec_img,'true_vec':truevec,f'output vec PCC:{output_PCC:.4f},MAE:{output_MAE:.4f}':output_vec} if not opt.input_nc == 3 else {'rtm image':rtm_img,f'iter vec image PCC:{original_PCC:.4f},MAE:{original_MAE:.4f}':itervec_img,'true_vec':truevec,f'output vec PCC:{output_PCC:.4f},MAE:{output_MAE:.4f}':output_vec,'born image':born_img}
    
    if show_grad:
        title_list.update({'grad x':grad_x,'grad y':grad_y})
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
    return {'original PCC':original_PCC,'originial MAE':original_MAE,'output PCC':output_PCC,'output MAE':output_MAE}

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
    # figure = plt.figure()
    # loc_index = 1
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
    # opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    # opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    datamode = 'test'
    prefix_index = {'train':0,'val':150,'test':170}
    iternumber = int(opt.iternumber) if opt.iternumber.isdigit() else opt.iternumber

    metrics_values = {'original PCC':[],'originial MAE':[],'output PCC':[],'output MAE':[]}

    test_loader = get_testloader(opt,prefix = './data/fwidata', mode = datamode, iternumber = iternumber)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create 

    save_name_suffix = 'show_gray_ppt_result_fig'

    if opt.eval:
        model.eval()
    if opt.savefig:
        os.makedirs(os.path.join('result',opt.save_path,save_name_suffix), exist_ok = True)
    if opt.savedata:
        os.makedirs('./data/fwidata/outputvec', exist_ok = True)
    
    headers = ['vec num', 'run iter', *metrics_values.keys()]
    values_list = []

    for i, data in enumerate(test_loader):
        # if i >= opt.num_test and not opt.savedata:
        #     break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
      
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        
        # plot_single_image(visuals)
        if not opt.savedata:
            if opt.iternumber == 'all':
                vec_num = i // 30 + prefix_index[datamode] # which vec model
                run_iter = (i % 30) * 10 + 10
                single_data_metric = plot_image(visuals, opt = opt, savepath = os.path.join('result',opt.save_path,save_name_suffix,'No_'+str(vec_num)+'iter'+str(run_iter)+".png"), vec_num = vec_num, run_iter = run_iter, show_grad = False, show_residual = False)
                # print(f'values:{values}')
                values_list.append([vec_num, run_iter, *[single_data_metric[keys] for keys in metrics_values.keys()]])
            else:
                single_data_metric = plot_image(visuals, opt = opt, savepath = os.path.join('result',opt.save_path,'result_fig',str(i)+".png"),show_grad = opt.multi_task, show_residual=opt.show_residual)
                values_list.append([i, 50, *[single_data_metric[keys] for keys in metrics_values.keys()]])

            for key,values in single_data_metric.items():
                metrics_values[key].append(values)
        else:
            savedata(visuals, savepath = os.path.join('./data/fwidata/outputvec',str(i+prefix_index[datamode])+"_iter_50"+'.npy'))
        # figure = plot_image(visuals,opt = opt)
        # savedata(visuals, savepath = './data/fwidata/outputvec')

    for key,values in metrics_values.items():
        metrics_values[key] = np.mean(metrics_values[key])
    with open(os.path.join('result',opt.save_path, opt.epoch+'gray_metric_result.txt'),'w') as f:
        f.write(str(metrics_values))
    save_metric_2_csv(headers, values_list, savepath = os.path.join('result',opt.save_path, opt.epoch+'gray_metric_csv.csv'))