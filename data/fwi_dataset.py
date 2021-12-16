import os
from sys import int_info, prefix
from unicodedata import normalize
from matplotlib.pyplot import figure
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import scipy.io as scio
import torch
import torch.nn.functional as F
import math
from scipy import signal
from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1
import csv

def PCC(img1,img2):
    m1 = np.mean(img1)
    m2 = np.mean(img2)
    diffimg1 = img1 - m1
    diffimg2 = img2 - m2
    PCC = np.sum(diffimg1 * diffimg2) / np.sqrt(np.sum(diffimg1**2)*np.sum(diffimg2**2))
    return PCC

def MAE(img1,img2):
    return np.average(np.abs(img1-img2))

def save_metric_2_csv(headers, values, savepath):
    with open(savepath,'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(values)

class fwiDataset(Dataset):
    def __init__(self, transform = None, train_mode = 'train', prefix = None, iternumber = 50, useborn_img = False, normalize_method = 'zscore', multi_task = False):
        
        prefix_index = {'train':0,'val':150,'test':170}
        if iternumber == 'all':
            data_len = {'train':4500,'val':600,'test':630}
        else:
            data_len = {'train':150,'val':20,'test':21}
        self.iternumber = iternumber
        self.data_length = data_len[train_mode]
        self.prefix_index = prefix_index[train_mode]
        self.transform = transform
        self.prefix = prefix
        self.useborn_img = useborn_img
        self.normalize_method = normalize_method
        self.multi_task = multi_task
        assert normalize_method in ['zscore', 'minmax']

    def __getitem__(self, index):
        if_loop = True
        while(if_loop):
            iternumber = self.iternumber if self.iternumber != 'all' else (index % 30) * 10 + 10
            vec_num = index // 30 if self.iternumber == 'all' else index
            # print(f'vec num : {vec_num+self.prefix_index}\t iternumber : {iternumber}')
            rtm_img = scio.loadmat(os.path.join(self.prefix,'migration',f'migration_{vec_num+self.prefix_index:04d}_iter{iternumber:04d}.mat'))['v']
            vec_img = scio.loadmat(os.path.join(self.prefix,'fwivel',f'fwivel_{vec_num+self.prefix_index:04d}_iter{iternumber:04d}.mat'))['v']
            if self.useborn_img:
                born_img = scio.loadmat(os.path.join(self.prefix,'reflmatall',f'No{vec_num+self.prefix_index}-iter{iternumber:03d}.mat'))['refl']
                if self.normalize_method == 'zscore':
                    born_img = (born_img / 0.0714)[np.newaxis,:,:]
                elif self.normalize_method == 'minmax':
                    born_img = born_img[np.newaxis,:,:]
                else:
                    raise ValueError('unkown normalize method')

            truevec = scio.loadmat(os.path.join(self.prefix,'vel',f'vel_{vec_num+self.prefix_index:04d}.mat'))['v']
            if self.normalize_method == 'zscore':
                rtm_img = (rtm_img / 0.0074)[np.newaxis,:]
                vec_img = ((vec_img - 4034) / 514)[np.newaxis,:]
                truevec = ((truevec - 4034) / 514)[np.newaxis,:]
            elif self.normalize_method == 'minmax':
                rtm_img = (rtm_img / 0.7502193450927734)[np.newaxis,:]
                vec_img = (vec_img / 6605.074823646438
)[np.newaxis,:]
                truevec = (truevec / 6605.074823646438
)[np.newaxis,:]

            if self.multi_task:
                grad_x = np.load(os.path.join(self.prefix,'truevec_grad_x',f'vec_grad_x_{vec_num+self.prefix_index:04d}.npy'))[np.newaxis,:]
                grad_y = np.load(os.path.join(self.prefix,'truevec_grad_y',f'vec_grad_y_{vec_num+self.prefix_index:04d}.npy'))[np.newaxis,:]
                truevec = np.concatenate([truevec,grad_x,grad_y],axis=0)

            if self.useborn_img:
                input_data = torch.from_numpy(np.concatenate([rtm_img,vec_img,born_img,truevec],axis=0)).type(torch.FloatTensor)
            else:
                input_data = torch.from_numpy(np.concatenate([rtm_img,vec_img,truevec],axis=0)).type(torch.FloatTensor)

            if torch.isnan(input_data).int().sum() > 0 or torch.isinf(input_data).int().sum() > 0:
                index = np.random.randint(0,self.data_length)
            else:
                if_loop = False
        if self.transform is not None:
            input_data = self.transform(input_data)
        A_to_B_data = {'A':input_data[0:2,:],'B':input_data[2:,:]} if not self.useborn_img else {'A':input_data[0:3,:],'B':input_data[3:,:]}
        return A_to_B_data
    def __len__(self):
        return self.data_length

def gettrain_and_val_loader(opt,prefix = './fwidata', iternumber = 50, train_shuffle = True):
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(45),
        transforms.RandomCrop(256),
    ])
    
    train_dataset = fwiDataset(transform = train_transform, prefix = prefix, train_mode = 'train', iternumber = iternumber, useborn_img = opt.useborn_img, normalize_method = opt.normalize_method, multi_task = opt.multi_task)
    train_loader = DataLoader(train_dataset, batch_size = opt.batch_size, shuffle = train_shuffle, num_workers=int(opt.num_threads))
    
    validation_dataset = fwiDataset(transform = None, prefix = prefix, train_mode = 'val', iternumber = iternumber, useborn_img = opt.useborn_img, normalize_method = opt.normalize_method, multi_task = opt.multi_task)
    validation_loader = DataLoader(validation_dataset, batch_size = 1, shuffle = False, num_workers = int(opt.num_threads) )

    return train_loader,validation_loader

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def get_testloader(opt,prefix = './fwidata', iternumber = 50, mode = 'test'):
    test_dataset = fwiDataset(transform = None, prefix = prefix, train_mode = mode, iternumber = iternumber, useborn_img = opt.useborn_img, normalize_method = opt.normalize_method, multi_task = opt.multi_task)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = int(opt.num_threads))
    return test_loader

def generate_sobel_data(data_path, save_path_x = None, save_path_y = None):
    truevec = scio.loadmat(data_path)['v']
    truevec = ((truevec - 4034) / 514)
    sobel_y = np.array([[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]])
    sobel_x = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])
    grad_x = signal.convolve2d(truevec, sobel_x, boundary = 'symm', mode = 'same') / 4
    grad_y = signal.convolve2d(truevec, sobel_y, boundary = 'symm', mode = 'same') / 4

    if save_path_x is not None and save_path_y is not None:
        np.save(save_path_x, grad_x)
        np.save(save_path_y, grad_y)


def get_customized_dataset(start_vec_index = 170, another_vec_index = 170, prefix = './fwidata'):
    data_list = []
    # width = 788
    # height = 266
    for iternumber in range(10,310,10):
        rtmiternumber = iternumber # - 10 if iternumber > 10 else iternumber
        veciternumber = iternumber
        rtm_index = 170
        vec_index = 174
        start_rtm_img = scio.loadmat(os.path.join(prefix,'migration',f'migration_{rtm_index:04d}_iter{rtmiternumber:04d}.mat'))['v']
        start_vec_img = scio.loadmat(os.path.join(prefix,'fwivel',f'fwivel_{vec_index:04d}_iter{veciternumber:04d}.mat'))['v']
        # another_rtm_img = scio.loadmat(os.path.join(prefix,'migration',f'migration_{another_vec_index:04d}_iter{iternumber:04d}.mat'))['v']
        # another_vec_img = scio.loadmat(os.path.join(prefix,'fwivel',f'fwivel_{another_vec_index:04d}_iter{iternumber:04d}.mat'))['v']
        vec_true_vec_img = scio.loadmat(os.path.join(prefix,'vel',f'vel_{vec_index:04d}.mat'))['v']
        rtm_true_vec_img = scio.loadmat(os.path.join(prefix,'vel',f'vel_{rtm_index:04d}.mat'))['v']
        # another_true_vec_img = scio.loadmat(os.path.join(prefix,'vel',f'vel_{another_vec_index:04d}.mat'))['v']

        # start_true_vec_img[:,393:] = another_true_vec_img[:,393:]
        # start_vec_img[:,393:] = another_vec_img[:,393:]
        # start_rtm_img[:,393:] = another_rtm_img[:,393:]

        vec_true_vec_img = vec_true_vec_img[np.newaxis,:]
        rtm_true_vec_img = rtm_true_vec_img[np.newaxis,:]

        start_vec_img = start_vec_img[np.newaxis,:]
        start_rtm_img = start_rtm_img[np.newaxis,:]

        # vec_true_vec_img = (start_true_vec_img - 4034) / 514
        start_vec_img = (start_vec_img - 4034) / 514
        start_rtm_img = start_rtm_img / 0.0074

        input_data = np.concatenate([start_rtm_img,start_vec_img], axis = 0)[np.newaxis,:]
        # output_data = start_true_vec_img[np.newaxis,:]
        output_data = np.concatenate([vec_true_vec_img,rtm_true_vec_img])
        output_data = (output_data - 4034) / 514

        data_list.append({
            'A':torch.from_numpy(input_data).type(torch.FloatTensor),'B':torch.from_numpy(output_data).type(torch.FloatTensor)
        })
        
    return data_list

def frequency_ana(prefix = './fwidata'):
    from tqdm import tqdm
    shoulian_num = 170
    fasan_num = 177

    shoulianvec = scio.loadmat(os.path.join(prefix,'vel',f'vel_{shoulian_num:04d}.mat'))['v']
    fasanvec = scio.loadmat(os.path.join(prefix,'vel',f'vel_{fasan_num:04d}.mat'))['v']
    
    shoulianvec = np.fft.fftshift(np.fft.fft2(shoulianvec))
    fasanvec = np.fft.fftshift(np.fft.fft2(fasanvec))

    shoulianvec = np.log(np.abs(shoulianvec))
    fasanvec = np.log(np.abs(fasanvec))

    metrics = []
    # fasanimgs = []

    for iternumber in tqdm(range(10,310,10)): 
        
        fasan_img1 = scio.loadmat(os.path.join(prefix,'fwivel',f'fwivel_{fasan_num:04d}_iter{iternumber:04d}.mat'))['v']
        shoulian_img1 = scio.loadmat(os.path.join(prefix,'fwivel',f'fwivel_{shoulian_num:04d}_iter{iternumber:04d}.mat'))['v']

        fasan_img1 = np.fft.fftshift(np.fft.fft2(fasan_img1))
        shoulian_img1 = np.fft.fftshift(np.fft.fft2(shoulian_img1))
        fasan_img1 = np.log(np.abs(fasan_img1))
        shoulian_img1 = np.log(np.abs(shoulian_img1))

        metrics.append({'iter':iternumber, 'fasan PCC':PCC(fasan_img1, fasanvec), 'fasan MAE': MAE(fasan_img1, fasanvec), 'shoulian PCC': PCC(shoulian_img1,shoulianvec), 'shoulian MAE': MAE(shoulian_img1, shoulianvec)})

        plt.subplot(121, title = 'iter vec fft')
        plt.plot(fasan_img1[133,:])
        plt.subplot(122, title = 'true vec fft')
        plt.plot(fasanvec[133, :])
        plt.tight_layout()
        plt.savefig(f'./tempfig/{fasan_num}_iter_{iternumber}.png',bbox_inches='tight',pad_inches = 0)
        plt.close()

    # plt.imshow(fasanvec)
    # plt.savefig(f'./tempfig/{fasan_num}_true.png',bbox_inches='tight',pad_inches = 0)
    # plt.close()

        # plt.subplot(121,title = 'fasan')
        # plt.imshow(np.log(np.abs(fasan_fft)))
        # plt.subplot(122,title = 'shoulian')
        # plt.imshow(np.log(np.abs(shoulian_fft)))
        # plt.show()

    # headers = list(metrics[0].keys())
    # with open(f'{shoulian_num}-vs-{fasan_num}.csv','w') as f:
    #     csvWriter = csv.DictWriter(f,fieldnames=headers)
    #     csvWriter.writeheader()
    #     csvWriter.writerows(metrics)

if __name__ == '__main__':
    # generate_sobel_data(f'./fwidata/vel/vel_{0:04d}.mat')
    # generate_sobel_data(f'./fwidata/vel/vel_{0:04d}.mat')

    # from tqdm import tqdm
    # for index in tqdm(range(191)):
    #     generate_sobel_data(f'./fwidata/vel/vel_{index:04d}.mat',f'./fwidata/truevec_grad_x/vec_grad_x_{index:04d}.npy',f'./fwidata/truevec_grad_y/vec_grad_y_{index:04d}.npy')

    # train_transform = transforms.Compose([
    #     transforms.Resize([256,256])
    # ])
    # train_dataset = fwiDataset(transform = train_transform,prefix = './fwidata')
    # train_loader = DataLoader(train_dataset, batch_size = 4 , shuffle=True)
    # for input in train_loader:
    #     print(input['A'].shape,input['B'].shape)
    # train_dataset = fwiDataset(transform = train_transform,prefix = './fwidata',train_mode='val')
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # for input,label in train_loader:
    #     continue
    # train_dataset = fwiDataset(transform = train_transform,prefix = './fwidata',train_mode='test')
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # from tqdm import tqdm
    # class options:
    #     num_threads = 0
    #     useborn_img = False
    #     batch_size = 1
    #     multi_task = True
    #     normalize_method = 'zscore'

    # traindl,val_dl = gettrain_and_val_loader(options(), iternumber = 'all', train_shuffle = False)
    # test_dl = get_testloader(options(), iternumber='all')
    
    # test_dataset = fwiDataset(transform = None, prefix = './fwidata', train_mode = 'test', iternumber = 50)
    # test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    # for input_data in test_loader:
    #     print(input_data['A'].shape)

    # from matplotlib import pyplot as plt
    # import matplotlib
    # from tqdm import tqdm
    # iternumber_dic = {}

    frequency_ana()
    
    # test_dl = get_customized_dataset()

    # for inputdata in test_dl:
    #     # print(f'input shape:{inputdata["A"].shape} output shape : {inputdata["B"].shape}')
    #     # continue
    #     plt.subplot(1,3,1, title = 'rtm')
    #     plt.imshow(inputdata['A'][0,0,:] * 0.0074 )
    #     plt.subplot(1,3,2, title = 'vec')
    #     plt.imshow(inputdata['A'][0,1,:] * 514 + 4034 ) 
    #     plt.subplot(1,3,3, title = 'true vec')
    #     plt.imshow(inputdata['B'][0,0,:] * 514 + 4034 )
    #     # plt.subplot(2,3,4, title = 'grad x')
    #     # im = plt.imshow(inputdata['B'][0,1,:])
    #     # add_colorbar(im)
    #     # plt.subplot(2,3,5, title = 'grad y')
    #     # im = plt.imshow(inputdata['B'][0,2,:])
    #     # add_colorbar(im)
    #     plt.show()
    

    # print(np.unique(list(iternumber_dic.keys())))
    # for keys,values in iternumber_dic.items():
    #     if iternumber_dic[keys] != iternumber_dic[0]:
    #         print(keys)

        # plt.subplot(2,2,1,title = 'rtm image')
        # plt.imshow(inputdata['A'][0,0,:])
        # plt.subplot(2,2,2,title = 'vec image')
        # plt.imshow(inputdata['A'][0,1,:])
        # plt.subplot(2,2,3,title = 'born image')
        # plt.imshow(inputdata['A'][0,2,:])
        # plt.subplot(2,2,4,title = 'true vec image')
        # plt.imshow(inputdata['B'][0,0,:])
        # plt.tight_layout()
        # plt.show()
        # plt.close()
    
    # train_data1,label = train_dataset.__getitem__(0)
    # print(train_data1.shape,label.shape)
    # plt.subplot(3,1,1)
    # plt.imshow(train_data1[0,:])
    # plt.subplot(3,1,2)
    # plt.imshow(train_data1[1,:])
    # plt.subplot(3,1,3)
    # plt.imshow(label)
    # plt.show()
    # print(type(train_data),type(test_data))
    # data_path = './my_label_crop.json'
    # data_dict = load_json_data(data_path)
    # train_data, test_data = train_test_split(data_dict)
    # print(f'train_num:{len(train_data)}\ttest_num:{len(test_data)}')
    # list_2_txt(test_data,'./traintest_txt/test.txt')
    # list_2_txt(train_data,'./traintest_txt/train.txt')
    # print(f'train data num:{len(train_data)}\ttest data num:{len(test_data)}')
    # print(have_test_data)
    # print(label_list)
    # co_transform = selfTransforms.Compose([
    #     selfTransforms.Resize(224),
    #     selfTransforms.Normalize(rgb_mean=[0.49139968, 0.48215827, 0.44653124], rgb_std=[0.24703233, 0.24348505, 0.26158768], d_max=65535),
    #     selfTransforms.RandomVerticalFlip(),
    #     selfTransforms.RandomHorizontalFlip(),
    #     selfTransforms.RandomRotate(180),
    # ])
    # input_transform = transforms.Compose([
    #     selfTransforms.ArrayToTensor(),
    # ])
    # train_dataset = ostraDataset(train_data, test_data, transform=input_transform, co_transform=co_transform, is_train=True, prefix='../../classification_data')
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # print(input[0].shape)
    # for input, label1,label2,label3 in train_loader:
    #     print(input[0].shape)
        # print(torch.max(input[0]))


