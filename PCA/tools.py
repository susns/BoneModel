import os

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance


def distance_RMSE(X, Y, scale=None):
    X = X.reshape(len(X), -1, 3)
    Y = Y.reshape(len(Y), -1, 3)

    d = (X - Y)**2
    d = np.sum(d, axis=2)
    d = np.sqrt(d)
    if scale is not None:
        d = d*scale.reshape(-1, 1)
    d = d.mean()
    return d


def distance_Chamfer(X, Y, scale=None):
    if scale is not None:
        X = X * scale
        Y = Y * scale
    X = torch.Tensor(X.reshape(1, -1, 3))
    Y = torch.Tensor(Y.reshape(1, -1, 3))
    dis, _ = chamfer_distance(X, Y)
    return dis


def get(names, suffix='.mhd', name_only=True):
    choose = []
    for name in names:
        if os.path.splitext(name)[1] == suffix:
            if name_only:
                choose.append(os.path.splitext(name)[0])
            else:
                choose.append(name)
    return choose


def get_all(path='data/ly/40-60 Age/F_23/', names=None):
    if names is None:
        names = get(os.listdir(path), suffix='.txt', name_only=False)

    pcs = []
    for name in names:
        try:
            pc = np.loadtxt(os.path.join(path, name), delimiter=' ')
            pcs.append(pc)
        except ValueError:
            pass

    if len(pcs) == 0:
        return None

    pcs = np.array(pcs)
    return pcs.reshape(len(pcs), -1)




