# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys
from mmcv import Config

from mmseg.models import build_segmentor

import mmseg_custom


def model_size(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total / 1e6


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--out',
        type=str,
        default=None)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    print('total', model_size(model))
    print('backbone', model_size(model.backbone))
    print('decode_head', model_size(model.decode_head))
    if hasattr(model.decode_head, 'unet'):
        print('decode_head.unet', model_size(model.decode_head.unet))
    
    
    

if __name__ == '__main__':
    main()
