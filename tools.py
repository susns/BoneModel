import os

import numpy as np

from PCA.tools import get
import open3d as o3d


def obj2txt(from_path, save_path, suffix='.obj'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    names = get(os.listdir(from_path), suffix, name_only=False)
    for name in names:
        if name == 'example'+suffix:
            continue

        mesh = o3d.io.read_triangle_mesh(os.path.join(from_path, name))
        print(name, mesh)

        pcs = np.array(mesh.vertices)
        np.savetxt(os.path.join(save_path, name.replace(suffix[1:], 'txt')), pcs, fmt='%f', delimiter=' ')


def obj2ply(from_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mesh = o3d.io.read_triangle_mesh(os.path.join(from_path, 'example.obj'))
    o3d.io.write_triangle_mesh(os.path.join(save_path, 'example.ply'), mesh, write_ascii=True)


if __name__ == '__main__':
    s = 'new'
    obj2txt(f'data/{s}/align', f'data/{s}/pc', '.obj')
    obj2ply(f'data/{s}/align', f'data/{s}/left')