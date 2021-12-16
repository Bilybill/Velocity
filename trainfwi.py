"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
from math import inf
import time
import torch
from torch.utils.data import dataset
from options.train_options import TrainOptions
from data import create_dataset,fwi_dataset
from data.fwi_dataset import gettrain_and_val_loader
from models import create_model
from torch.utils.tensorboard import SummaryWriter
import os
from testfwi import PCC
import numpy as np

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    os.makedirs(os.path.join('result', opt.save_path, 'model_saved'),exist_ok=True)
    if opt.backup:
        os.makedirs(os.path.join('result',opt.save_path,'back_up'),exist_ok=True)
        backup_dir = os.path.join('result',opt.save_path,'back_up')
        os.system('cp *.py %s/' % backup_dir)
        os.system('cp *.sh %s/' % backup_dir)
        os.system('cp models/*.py %s' % backup_dir)
        os.system('cp data/fwi_dataset.py %s' % backup_dir)
        os.system('cp options/*.py %s' % backup_dir)
    
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # dataset_size = len(dataset)    # get the number of images in the dataset.
    # print('The number of training images = %d' % dataset_size)
    iternumber = int(opt.iternumber) if opt.iternumber.isdigit() else opt.iternumber
    train_loader,validation_loader = gettrain_and_val_loader(opt,prefix = './data/fwidata', iternumber = iternumber)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    writer = SummaryWriter(os.path.join('result',opt.save_path,'tb_runs'))
    
    total_iters = 0                # the total number of training iterations
    best_pcc = -inf

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        learning_rate = model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        writer.add_scalar('learning rate', learning_rate, epoch)
        model.train()

        for i, data in enumerate(train_loader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            if opt.model == 'rudsr':
                model.optimize_parameters(writer, total_iters)   # calculate loss functions, get gradients, update network weights
            else:
                model.optimize_parameters()

            # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            #     save_result = total_iters % opt.update_html_freq == 0
            #     model.compute_visuals()
            #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print(f'current loss:{losses} at iters {total_iters} epoch {epoch}')
                for name,loss_value in losses.items():
                    writer.add_scalar(name,loss_value,total_iters)
                

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            # model.save_networks(epoch)
            model.eval()
            test_pcc = []
            for validation_data in validation_loader:
                model.set_input(validation_data)
                model.test()
                visuals = model.get_current_visuals()
                outputtervec_img = np.squeeze(visuals['fake_B'].cpu().float().numpy())
                true_vec_img = np.squeeze(visuals['real_B'].cpu().float().numpy())
                test_pcc.append(PCC(outputtervec_img,true_vec_img))
            test_pcc = np.mean(test_pcc)
            print(f'epoch:{epoch}\tvalidation PCC:{test_pcc}')
            writer.add_scalar('validation PCC', test_pcc, epoch)
            if test_pcc > best_pcc:
                print(f'saving the model at best app {epoch}, pcc {test_pcc} .v.s best_pcc {best_pcc}')
                best_pcc = test_pcc
                model.save_networks('best_pcc')
        print('End of epoch %d / %d \t Time Taken: %d sec best pcc is %.2f' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time, best_pcc))
