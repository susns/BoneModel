import argparse
import os
import time

import numpy as np;
import torch
import trimesh

np.set_printoptions(precision=4)

from PCA.SAP.src.optimization import Trainer
from PCA.SAP.src.utils import load_config, update_config, initialize_logger, \
    get_learning_rate_schedules, adjust_learning_rate, AverageMeter,\
         update_optimizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_target_pc(file_path):
    vertices = np.loadtxt(file_path, delimiter=' ')
    target_pts = torch.tensor(vertices, device=device)[None].float()

    target = {'target_points': target_pts,
                   'target_normals': None,  # normals are never used
                   'gt_mesh': None}
    return target


def get_target(vertices):
    target_pts = torch.tensor(vertices, device=device)[None].float()

    target = {'target_points': target_pts,
                   'target_normals': None,  # normals are never used
                   'gt_mesh': None}
    return target


def get_origin_mesh(sphere_radius, pt_num):
    sphere_mesh = trimesh.creation.uv_sphere(radius=sphere_radius,
                                             count=[256, 256])
    points, idx = sphere_mesh.sample(pt_num,
                                     return_index=True)
    points += 0.5  # make sure the points are within the range of [0, 1)
    normals = sphere_mesh.face_normals[idx]
    points = torch.from_numpy(points).unsqueeze(0).to(device)
    normals = torch.from_numpy(normals).unsqueeze(0).to(device)

    points = torch.log(points / (1 - points))  # inverse sigmoid
    inputs = torch.cat([points, normals], axis=-1).float()
    return inputs


def one_optim(cfg, logger, inputs, targets, log=False):
    inputs.requires_grad = True
    model = None  # no network

    # initialize optimizer
    cfg['train']['schedule']['pcl']['initial'] = cfg['train']['lr_pcl']
    print('Initial learning rate:', cfg['train']['schedule']['pcl']['initial'])
    if 'schedule' in cfg['train']:
        lr_schedules = get_learning_rate_schedules(cfg['train']['schedule'])
    else:
        lr_schedules = None

    optimizer = update_optimizer(inputs, cfg,
                                 epoch=0, model=model, schedule=lr_schedules)

    state_dict = dict()

    start_epoch = state_dict.get('epoch', -1)

    trainer = Trainer(cfg, optimizer, device=device)
    runtime = {}
    runtime['all'] = AverageMeter()

    # training loop
    for epoch in range(start_epoch + 1, cfg['train']['total_epochs'] + 1):

        # schedule the learning rate
        if (epoch > 0) & (lr_schedules is not None):
            if (epoch % lr_schedules[0].interval == 0):
                adjust_learning_rate(lr_schedules, optimizer, epoch)

                if log and len(lr_schedules) > 1:
                    print('[epoch {}] net_lr: {}, pcl_lr: {}'.format(epoch,
                                                                     lr_schedules[0].get_learning_rate(epoch),
                                                                     lr_schedules[1].get_learning_rate(epoch)))
                elif log:
                    print('[epoch {}] adjust pcl_lr to: {}'.format(epoch,
                                                                   lr_schedules[0].get_learning_rate(epoch)))

        start = time.time()
        loss, loss_each = trainer.train_step(targets, inputs, model, epoch)
        runtime['all'].update(time.time() - start)

        if epoch % cfg['train']['print_every'] == 0:
            log_text = ('[Epoch %02d] loss=%.5f') % (epoch, loss)
            if loss_each is not None:
                for k, l in loss_each.items():
                    if l.item() != 0.:
                        log_text += (' loss_%s=%.5f') % (k, l.item())

            log_text += (' time=%.3f / %.3f') % (runtime['all'].val,
                                                 runtime['all'].sum)
            if log:
                logger.info(log_text)
                print(log_text)

        # resample and gradually add new points to the source pcl
        if (epoch > 0) & \
                (cfg['train']['resample_every'] != 0) & \
                (epoch % cfg['train']['resample_every'] == 0) & \
                (epoch < cfg['train']['total_epochs']):
            inputs = trainer.point_resampling(inputs)
            optimizer = update_optimizer(inputs, cfg,
                                         epoch=epoch, model=model, schedule=lr_schedules)
            trainer = Trainer(cfg, optimizer, device=device)

    return inputs, trainer.get_mesh(inputs)


def main(targets, grids=[32, 64, 128, 128]):
    parser = argparse.ArgumentParser(description='MNIST toy experiment')
    parser.add_argument('--config', type=str, default='configs/optim_based/teaser.yaml')
    parser.add_argument('--seed', type=int, default=1457, metavar='S', 
                        help='random seed')
    
    args, unknown = parser.parse_known_args()

    path = os.path.abspath(os.path.dirname(__file__))
    cfg = load_config(os.path.join(path, args.config), os.path.join(path, 'configs/default.yaml'))
    cfg = update_config(cfg, unknown)

    # print(cfg['train']['out_dir'])

    # boiler-plate
    if cfg['train']['timestamp']:
        cfg['train']['out_dir'] += '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    logger = initialize_logger(cfg)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # tensorboardX writer
    tblogdir = os.path.join(cfg['train']['out_dir'], "tensorboard_log")
    if not os.path.exists(tblogdir):
        os.makedirs(tblogdir)

    inputs = get_origin_mesh(cfg['model']['sphere_radius'], cfg['data']['num_points'])
    # targets = get_target_pc(cfg['data']['data_path'])


    # origin
    # grids = [32, 64, 128]
    # lr_pcl = [0.002000, 0.001400, 0.000980]
    # epochs = [1000, 1000, 1000]
    # sigmas = [2, 2, 2]

    # fast
    # grids = [32, 64, 128]
    # lr_pcl = [0.002000, 0.001400, 0.000980]
    # epochs = [500, 200, 50]
    # sigmas = [2, 2, 2]

    # best
    # grids = [32, 64, 128, 128]
    lr_pcl = [0.002, 0.0014, 0.00098, 0.0005]
    epochs = [500, 200, 50, 10]
    sigmas = [2, 2, 2, 3]

    for i in range(len(grids)):
        cfg['model']['grid_res'] = grids[i]
        cfg['train']['lr_pcl'] = lr_pcl[i]
        cfg['train']['total_epochs'] = epochs[i]
        cfg['model']['psr_sigma'] = sigmas[i]
        inputs, mesh = one_optim(cfg, logger, inputs, targets)
        # o3d.io.write_triangle_mesh(os.path.join(cfg['train']['out_dir'], f'{grids[i]}.ply'), mesh)

    return mesh


def get(names, suffix='.mhd', name_only=True):
    choose = []
    for name in names:
        if os.path.splitext(name)[1] == suffix:
            if name_only:
                choose.append(os.path.splitext(name)[0])
            else:
                choose.append(name)
    return choose


if __name__ == '__main__':
    path = '../../data/pc/atlas/left_5000'
    name = 'LeftHip2_cpd'
    targets = get_target_pc(file_path=os.path.join(path, name + '.txt'))
    main(targets)
