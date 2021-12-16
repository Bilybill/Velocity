from sys import prefix
from numpy.core.numeric import Inf
import scipy.io as scio
import os
import numpy as np
from tqdm import tqdm

iternumber = 50
prefix = './fwidata'
mean = []
std = []

born_max = -Inf
rtm_max = -Inf
vec_max = -Inf
true_max = -Inf

for index in tqdm(range(4500)):

    iternumber = (index % 30) * 10 + 10
    vec_num = index // 30

    born_img = scio.loadmat(os.path.join(prefix,'reflmatall',f'No{vec_num}-iter{iternumber:03d}.mat'))['refl']
    
    rtm_img = scio.loadmat(os.path.join(prefix,'migration',f'migration_{vec_num:04d}_iter{iternumber:04d}.mat'))['v']

    vec_img = scio.loadmat(os.path.join(prefix,'fwivel',f'fwivel_{vec_num:04d}_iter{iternumber:04d}.mat'))['v']

    truevec = scio.loadmat(os.path.join(prefix,'vel',f'vel_{vec_num:04d}.mat'))['v']

    if born_max < np.max(np.abs(born_img)):
        born_max = np.max(np.abs(born_img))
    if rtm_max < np.max(np.abs(rtm_img)):
        rtm_max = np.max(np.abs(rtm_img))
    if vec_max < np.max(np.abs(vec_img)):
        vec_max = np.max(np.abs(vec_img))
    if true_max < np.max(np.abs(truevec)):
        true_max = np.max(np.abs(truevec))

print(f'born max {born_max}\trtm max {rtm_max}\tvec_max {vec_max}\ttrue max {true_max}')

#     mean.append(np.mean(born_img))
#     std.append(np.std(born_img))


# print(f'mean {np.mean(mean)}, std: {np.mean(std)}')