# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import numpy as np
import random
from argparse import ArgumentParser
import warnings

from mmseg.ops import resize
from mmseg.apis import init_segmentor
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset, build_dataloader
from mmcv.runner import load_checkpoint
from mmseg.utils import get_root_logger
from mmcv.utils import Config
import mmcv

import mmseg_custom  # noqa: F401,F403
from mmseg_custom.apis.test_multi_steps import np2tmp
import os.path as osp

from debug_q_pred import show_result, plot_img_list


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='ade',
        choices=['ade'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    setup_seed(0)
    # build the model from a config file and a checkpoint file
    cfg = Config.fromfile(args.config)
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    global device
    device = args.device
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.data.test.pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=16, pad_val=0, seg_pad_val=0),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg']),
            ])
    ]
    dataset = build_dataset(cfg.data.test)
    dataset_train = build_dataset(cfg.data.train)
    # build the dataloader
    # data_loader = build_dataloader(dataset, **test_loader_cfg)
    # results = single_gpu_test_multi_steps(model,
    #                                       data_loader,
    #                                       pre_eval=True)
    data = dataset.__getitem__(1643)
    result = []
    with torch.no_grad():
        for i in range(model.decode_head.diffusion_timesteps):
            feat = model.extract_feat(data['img'][0]._data.unsqueeze(0).to(device))
            pred = model.decode_head.forward_one_step(
                feat, data['gt_semantic_seg'][0]._data.unsqueeze(0).to(device), i)
            resize_shape = data['img_metas'][0]._data['img_shape'][:2]
            pred = pred[:, :, :resize_shape[0], :resize_shape[1]]
            pred = resize(
                pred,
                size=data['img_metas'][0]._data['ori_shape'][:2],
                mode='bilinear',
                align_corners=False,
                warning=False)
            pred = F.softmax(pred, dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
            result.append(pred)
    result.reverse()  # [19, ... 0]
    with torch.no_grad():
        result_ = model.simple_test(img=data['img'][0]._data.unsqueeze(0).to(device), img_meta=[data['img_metas'][0]._data])
    result_ = [res[0] for res in result_]
    
    img_list = []
    result = [result[0], result[6], result[13], result[-1]]
    for i, res in enumerate(result):
        img = show_result(data['img_metas'][0]._data['filename'],
                            res,
                            palette=dataset.PALETTE,
                            # out_file='example.png',
                            opacity=1)
        import imageio
        imageio.imsave(f'gt_{i}.png', img)
        img_list.append(img)
    resize_shape = data['img_metas'][0]._data['img_shape'][:2]
    # img = show_result(data['img_metas'][0]._data['filename'],
    #                   data['gt_semantic_seg'][0]._data.squeeze(0).numpy()[:resize_shape[0], :resize_shape[1]],
    #                   palette=dataset.PALETTE,
    #                   # out_file='example.png',
    #                   opacity=1)
    # img_list.append(img)
    # plot_img_list(img_list, 3, out_file='infer_gt.png')

    img_list = []
    result_ = [result_[0], result_[6], result_[13], result_[-2]]
    for i, res in enumerate(result_):
        print(res.max())
        imageio.imsave(f'data_raw_{i}.png', res.astype('uint8'))
        
        img = show_result(data['img_metas'][0]._data['filename'],
                            res,
                            palette=dataset.PALETTE,
                            # out_file='example.png',
                            opacity=1)
        img_list.append(img)
        import imageio
        imageio.imsave(f'data_{i}.png', img)
        
    resize_shape = data['img_metas'][0]._data['img_shape'][:2]
    # img = show_result(data['img_metas'][0]._data['filename'],
    #                   data['gt_semantic_seg'][0]._data.squeeze(0).numpy()[:resize_shape[0], :resize_shape[1]],
    #                   palette=dataset.PALETTE,
    #                   # out_file='example.png',
    #                   opacity=1)
    # img_list.append(img)
    # plot_img_list(img_list, 3, out_file='infer_data.png')

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    main(args)


