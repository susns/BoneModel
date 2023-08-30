import os
import time

import open3d as o3d
import numpy as np
import torch
from pytorch3d import ops
from PCA.tools import get
from preprocess.alignmesh.align import align


# 先用fps下采样mesh对应pc，再用ball_pivoting方法从pc重建回mesh
def plan1(from_path, save_path):
    names = get(os.listdir(from_path), '.ply', name_only=False)
    for i, name in enumerate(names):
        mesh = o3d.io.read_triangle_mesh(os.path.join(from_path, name))
        mesh.compute_vertex_normals()

        pts = np.array(mesh.vertices)
        normals = np.array(mesh.vertex_normals)
        pts_ = torch.Tensor(pts).unsqueeze(0)
        _, idx = ops.sample_farthest_points(pts_, None, 2000)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts[idx[0].numpy()])
        pc.normals = o3d.utility.Vector3dVector(normals[idx[0].numpy()])

        distances = pc.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        max_dist = np.max(distances)

        radius = np.array([2 * avg_dist, 3 * avg_dist, 3 * max_dist])
        # radius = np.array([avg_dist, 2 * avg_dist, 2 * max_dist])

        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, o3d.utility.DoubleVector(radius))

        # rec_mesh.compute_vertex_normals()
        # rec_mesh.paint_uniform_color([0.6, 0.6, 0.6])
        # o3d.visualization.draw_geometries([mesh])
        # o3d.visualization.draw_geometries([rec_mesh])

        o3d.io.write_triangle_mesh(os.path.join(save_path, name), rec_mesh, write_ascii=True)

        print(f'{i + 1}/{len(names)} {name}')
        print(avg_dist, max_dist)
        print("Mesh", rec_mesh)
        print()


# 用poisson重建方法得到mesh，既可实现一点下采样的效果，也可以平滑表面
def smooth(from_path, save_path, vis=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    names = get(os.listdir(from_path), '.ply', name_only=False)
    for i, name in enumerate(names):
        mesh = o3d.io.read_triangle_mesh(os.path.join(from_path, name))
        mesh.compute_vertex_normals()

        pc = o3d.geometry.PointCloud()
        pc.points = mesh.vertices
        pc.normals = mesh.vertex_normals  # critical
        # pc.estimate_normals()

        # with o3d.utility.VerbosityContextManager(
        #         o3d.utility.VerbosityLevel.Debug) as cm:
        #     rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=7)
        rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=7)

        if vis:
            rec_mesh.compute_vertex_normals()
            rec_mesh.paint_uniform_color([0.6, 0.6, 0.6])
            o3d.visualization.draw_geometries([pc], window_name=name, point_show_normal=True)
            o3d.visualization.draw_geometries([mesh], window_name='origin')
            o3d.visualization.draw_geometries([rec_mesh], window_name='poisson')

        o3d.io.write_triangle_mesh(os.path.join(save_path, name),
                                   rec_mesh,
                                   write_ascii=True,
                                   write_vertex_normals=False,
                                   write_vertex_colors=False)

        print(f'{i + 1}/{len(names)} {name}')
        print("Mesh", rec_mesh)
        print()


# 利用指定voxel实现mesh下采样
def plan3(from_path, save_path, num=80):
    names = get(os.listdir(from_path), '.ply', name_only=False)
    for i, name in enumerate(names):
        mesh = o3d.io.read_triangle_mesh(os.path.join(from_path, name))

        voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / num
        mesh_voxel = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)

        # mesh_voxel.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh_voxel], window_name='voxel')

        o3d.io.write_triangle_mesh(os.path.join(save_path, name), mesh_voxel, write_ascii=True)

        print(f'{i + 1}/{len(names)} {name} {mesh}')
        print(f'voxel_size = {voxel_size:e} or {num}')
        print('voxel', mesh_voxel)
        print()


# 指定triangles num下采样mesh
def down_sampling(from_path, save_path, tri_num=10000, vis=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    names = get(os.listdir(from_path), '.ply', name_only=False)
    for i, name in enumerate(names):
        mesh = o3d.io.read_triangle_mesh(os.path.join(from_path, name))
        mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=tri_num)

        if vis:
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh], window_name=name)
            mesh_smp.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh_smp], window_name='face')

        o3d.io.write_triangle_mesh(os.path.join(save_path, name), mesh_smp, write_ascii=True)

        print(f'{i + 1}/{len(names)} {name} {mesh}')
        print('face', mesh_smp)
        print()


if __name__ == '__main__':
    # 光滑MC提取的表面
    # fpath = '../data/new/surface'
    # spath = '../data/new/poisson'
    # smooth(fpath, spath)

    # 下采样
    fpath = '../data/old/poisson'
    spath = '../data/old/sample'
    down_sampling(fpath, spath, tri_num=10026)


