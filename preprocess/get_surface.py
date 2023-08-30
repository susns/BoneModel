import os
import time

import numpy as np
import open3d as o3d
from medpy.io import load
from skimage import measure

from PCA.tools import get


def get_surface(from_path, save_path, part, vis=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    names = get(os.listdir(from_path), suffix='.mhd', name_only=True)
    for i, name in enumerate(names):
        allparts, header = load(os.path.join(from_path, name+'.mhd'))
        bone = np.where(allparts == part, 2, 0).astype(np.int32)

        if np.sum(bone)==0:  # 没有该骨骼
            continue

        t1 = time.time()
        verts, faces, normals, values = measure.marching_cubes(bone, level=1, spacing=header.spacing)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        # mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.triangles = o3d.utility.Vector3iVector(faces[:, [1, 0, 2]])
        save_name = os.path.join(save_path, name+'.ply')

        if vis:
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh], window_name=name)

        o3d.io.write_triangle_mesh(save_name, mesh, write_ascii=True)

        print(f'{i+1:>2}/{len(names)} {name:>3} {time.time()-t1:>5.2f}s')


if __name__ == '__main__':
    fpath = '../data/new/mask'
    spath = '../data/new/surface'
    p = 1  # left pelvis 23 for old, 1 for new
    get_surface(fpath, spath, p, vis=False)