import numpy as np;
import torch
from torch.utils.data import Dataset

np.set_printoptions(precision=4)
import argparse
from collections import defaultdict
from PCA.SAP.src import config
from PCA.SAP.src.dpsr import DPSR
from PCA.SAP.src.model import Encode2Points
from PCA.SAP.src.utils import load_config, load_model_manual, scale2onet

from tqdm import tqdm
import open3d as o3d

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 读取为numpy数组
padding = 1


# 读取 .xyz 文件保存到 npz格式
def get_path():
    # current_path = os.getcwd()
    current_path = os.path.abspath(os.path.dirname(__file__))
    print(current_path)
    return current_path


# current_path = os.getcwd()
# path = os.path.join(current_path, 'configs')
# print(path)


def toNpz(vertices):
    current_path = get_path()
    points = vertices
    normals = vertices
    # 处理方式修改 给定数据为0-1 变为 -0.5-0.5
    points = [ x - 0.5 for x in points]
    # points = [x / 2 for x in points]
    if not os.path.exists(os.path.join(current_path, 'data_npz/my_try_one/1/1/')):
        os.makedirs(os.path.join(current_path, 'data_npz/my_try_one/1/1/'))
    np.savez(os.path.join(current_path, 'data_npz/my_try_one/1/1/pointcloud.npz'), points=points, normals=normals)


def main():
    current_path = get_path()
    parser = argparse.ArgumentParser(description='MNIST toy experiment')
    # parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--iter', type=int, metavar='S', help='the training iteration to be evaluated.')

    args = parser.parse_args()
    cfg = load_config(os.path.join(current_path, 'configs/learning_based/noise_small/ours_pretrained.yaml'),
                      os.path.join(current_path, 'configs/default.yaml'))
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda:2" if use_cuda else "cpu")
    device = torch.device("cpu")

    data_type = cfg['data']['data_type']
    input_type = cfg['data']['input_type']
    vis_n_outputs = cfg['generation']['vis_n_outputs']
    if vis_n_outputs is None:
        vis_n_outputs = -1
    # Shorthands
    out_dir = cfg['train']['out_dir']
    if not out_dir:
        os.makedirs(out_dir)
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
    out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
    out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

    # PYTORCH VERSION > 1.0.0
    assert (float(torch.__version__.split('.')[-3]) > 0)

    dataset = config.get_dataset('test', cfg, return_idx=True)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)

    # model配置   cfg是读取的配置文件
    model = Encode2Points(cfg).to(device)
    # 模型权重文件修改 
    state_dict = torch.load(os.path.join(current_path, 'configs/learning_based/model_best.pt'), map_location='cpu')
    # 将权重读取进模型中
    load_model_manual(state_dict['state_dict'], model)
    # Generator
    generator = config.get_generator(model, cfg, device=device)

    # Determine what to generate
    generate_mesh = cfg['generation']['generate_mesh']
    generate_pointcloud = cfg['generation']['generate_pointcloud']

    # Statistics
    time_dicts = []

    # Generate
    model.eval()
    dpsr = DPSR(res=(cfg['generation']['psr_resolution'],
                     cfg['generation']['psr_resolution'],
                     cfg['generation']['psr_resolution']),
                sig=cfg['generation']['psr_sigma']).to(device)

    # Count how many models already created
    model_counter = defaultdict(int)

    print('Generating...')

    for data in test_loader:
        imgs = data
        print(imgs)

    for it, data in enumerate(tqdm(test_loader)):
        out = generator.generate_mesh(data)
        v, f, points, normals, stats_dict = out

        if len(v.shape) > 2:
            v, f = v[0], f[0]
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
            f = f.detach().cpu().numpy()
        mesh = o3d.geometry.TriangleMesh()
        # 反向变换
        mesh.vertices = o3d.utility.Vector3dVector(scale2onet(v))
        mesh.triangles = o3d.utility.Vector3iVector(f)

        return mesh
        # # Output folders
        # mesh_dir = os.path.join(generation_dir, 'meshes')
        # in_dir = os.path.join(generation_dir, 'input')
        # pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
        # generation_vis_dir = os.path.join(generation_dir, 'vis', )

        # # Get index etc.
        # idx = data['idx'].item()

        # try:
        #     model_dict = dataset.get_model_dict(idx)
        # except AttributeError:
        #     model_dict = {'model': str(idx), 'category': 'n/a'}

        # modelname = model_dict['model']
        # category_id = model_dict['category']

        # try:
        #     category_name = dataset.metadata[category_id].get('name', 'n/a')
        # except AttributeError:
        #     category_name = 'n/a'

        # if category_id != 'n/a':
        #     mesh_dir = os.path.join(mesh_dir, str(category_id))
        #     pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
        #     in_dir = os.path.join(in_dir, str(category_id))

        #     folder_name = str(category_id)
        #     if category_name != 'n/a':
        #         folder_name = str(folder_name) + '_' + category_name.split(',')[0]

        #     generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

        # # Create directories if necessary
        # if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        #     os.makedirs(generation_vis_dir)

        # if generate_mesh and not os.path.exists(mesh_dir):
        #     os.makedirs(mesh_dir)

        # if generate_pointcloud and not os.path.exists(pointcloud_dir):
        #     os.makedirs(pointcloud_dir)

        # if not os.path.exists(in_dir):
        #     os.makedirs(in_dir)

        # # Timing dict
        # time_dict = {
        #     'idx': idx,
        #     'class id': category_id,
        #     'class name': category_name,
        #     'modelname':modelname,
        # }
        # time_dicts.append(time_dict)

        # Generate outputs
        # out_file_dict = {}

        # if generate_mesh:
        #     #! deploy the generator to a separate class
        #     out = generator.generate_mesh(data)

        #     v, f, points, normals, stats_dict = out
        #     time_dict.update(stats_dict)

        #     # Write output
        #     mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
        #     export_mesh(mesh_out_file, scale2onet(v), f)
        #     out_file_dict['mesh'] = mesh_out_file

        # if generate_pointcloud:
        #     pointcloud_out_file = os.path.join(
        #         pointcloud_dir, '%s.ply' % modelname)                
        #     export_pointcloud(pointcloud_out_file, scale2onet(points), normals)
        #     out_file_dict['pointcloud'] = pointcloud_out_file

        # if cfg['generation']['copy_input']:
        #     inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
        #     p = data.get('inputs').to(device)
        #     export_pointcloud(inputs_path, scale2onet(p))
        #     out_file_dict['in'] = inputs_path

        # # Copy to visualization directory for first vis_n_output samples
        # c_it = model_counter[category_id]
        # if c_it < vis_n_outputs:
        #     # Save output files
        #     img_name = '%02d.off' % c_it
        #     for k, filepath in out_file_dict.items():
        #         ext = os.path.splitext(filepath)[1]
        #         out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
        #                                 % (c_it, k, ext))
        #         shutil.copyfile(filepath, out_file)

        #     # Also generate oracle meshes
        #     if cfg['generation']['exp_oracle']:
        #         points_gt = data.get('gt_points').to(device)
        #         normals_gt = data.get('gt_points.normals').to(device)
        #         psr_gt = dpsr(points_gt, normals_gt)
        #         v, f, _ = mc_from_psr(psr_gt,
        #                 zero_level=cfg['data']['zero_level'])
        #         out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
        #                                 % (c_it, 'mesh_oracle', '.off'))
        #         export_mesh(out_file, scale2onet(v), f)

        # model_counter[category_id] += 1

    # # Create pandas dataframe and save
    # time_df = pd.DataFrame(time_dicts)
    # time_df.set_index(['idx'], inplace=True)
    # time_df.to_pickle(out_time_file)

    # # Create pickle files  with main statistics
    # time_df_class = time_df.groupby(by=['class name']).mean(numeric_only=True)
    # time_df_class.loc['mean'] = time_df_class.mean()
    # time_df_class.to_pickle(out_time_file_class)

    # # Print results
    # print('Timings [s]:')
    # print(time_df_class)
