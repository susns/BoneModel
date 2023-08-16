import os
import time

import matplotlib.pyplot as plt
from pycpd import RigidRegistration
import numpy as np
import open3d as o3d
import json
from PCA.tools import distance_RMSE


def file_filter(f):
    if f == '.DS_Store':
        return False
    else:
        return True


def space_align_and_norm(load_path, save_path):
    names = os.listdir(load_path)
    names = list(filter(file_filter, names))
    datas = []
    for name in names:
        data = np.loadtxt(os.path.join(load_path, name), delimiter=' ')
        datas.append(data)

    datas = np.array(datas)
    template = datas[0]
    scales1 = []

    for k in range(1):
        datas_align = []
        for i in range(len(datas)):
            t0 = time.time()
            name = names[i]
            Y = datas[i]
            reg = RigidRegistration(**{'X': template, 'Y': Y})  # 把Y跟X对齐
            Y_hat, (s, r, t) = reg.register()

            datas_align.append(Y_hat)
            scales1.append(s)
            print(f'{i}/{len(datas)} {name} {time.time()-t0}')

        mean = np.mean(datas_align, axis=0)
        d = distance_RMSE(template, mean)
        print('D(template, mean)', d)

        template = mean

    center = np.mean(datas_align, axis=1, keepdims=True)
    datas_align = datas_align - center
    scales2 = np.max(np.sqrt(np.sum(datas_align**2, axis=2, keepdims=True)), axis=1, keepdims=True)
    datas_align = datas_align / scales2

    scales = {}
    scales2 = scales2.reshape(-1)
    for i in range(len(names)):
        scales[names[i]] = scales1[i]*scales2[i]
        np.savetxt(os.path.join(save_path, names[i]), datas_align[i], fmt='%f', delimiter=' ')

    with open(os.path.join(save_path, 'scale'), 'w') as f_obj:
        json.dump(scales, f_obj)


if __name__ == '__main__':
    load_path = './data/pc/atlas/pelvis'
    save_path = 'data/pc/atlas/left_5000'

    # load_path = 'data/pc/ly/left'
    # save_path = 'data/pc/ly/left_1024'
    space_align_and_norm(load_path, save_path)
