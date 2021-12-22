import os
from numba.core.errors import IRError
from tqdm import tqdm
from scipy import io as scio
import numpy as np
from matplotlib import pyplot as plt
if __name__ == '__main__':
    rows = 266
    cols = 788
    test_vec_num = 170
    test_iter = 10
    cuda_device = 1
    # truevec = np.fromfile(os.path.join('../data/fwidata/truevel',f'{test_vec_num:04d}.bin'), dtype = np.float32).reshape((cols,rows)).T

    # fwi_vec = scio.loadmat(os.path.join('../data/fwidata/fwivel',f'fwivel_{test_vec_num:04d}_iter{test_iter:04d}.mat'))['v'].astype(np.float32).T.reshape(-1)
    # fwi_vec.tofile(os.path.join('../data/fwidata/tempoutputvec',f'vec{test_vec_num:04d}_iter{test_iter:04d}.bin'))
    
    rtm_data = scio.loadmat(os.path.join('../data/fwidata/migration',f'migration_{test_vec_num:04d}_iter{test_iter:04d}.mat'))['v']
    # os.system(f'./tti2d_acousticRTM {test_vec_num} {cuda_device} {test_iter}')
    
    itermig = np.fromfile(os.path.join('../data/fwidata/tempoutputrtm',f'mig_vec{test_vec_num:04d}_iter{test_iter:04d}.bin'),dtype = np.float32).reshape((cols,rows)).T

    plt.subplot(3,1,1,title = f'original rtm, data range:{np.min(rtm_data)}~{np.max(rtm_data)}')
    plt.imshow(rtm_data)
    plt.subplot(3,1,2,title = f'program rtm, data range:{np.min(itermig)}~{np.max(itermig)}')
    plt.imshow(itermig)
    plt.subplot(3,1,3,title = f'residual, data range:{np.min(itermig-rtm_data)}~{np.max(itermig-rtm_data)}')
    plt.imshow(itermig - rtm_data)
    plt.show()