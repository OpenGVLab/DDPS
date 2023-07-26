import os
import torch
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="origin.pth", help='out dir')
    args = parser.parse_args()
    return args


def main(args):
    checkpoint = args.checkpoint
    ema_ckpt = torch.load(checkpoint, map_location='cpu')
    ema_model = ema_ckpt['state_dict']
    origin_model = dict()
    for key, value in ema_model.items():
        if key.startswith('ema'):
            continue
        else:
            ema_key = f"ema_{key.replace('.', '_')}"
            origin_model[key] = ema_model[ema_key]
    origin_ckpt = {'state_dict':origin_model}
    output_dir = os.path.dirname(checkpoint)
    output_file = os.path.join(output_dir, args.out)
    torch.save(origin_ckpt, output_file)
    print('Done!')
    

if __name__ == '__main__':
    args = parse_args()
    main(args)