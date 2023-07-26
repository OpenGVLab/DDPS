# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings

import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

import torch.distributed as dist

def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test_multi_steps(model,
                                data_loader,
                                show=False,
                                out_dir=None,
                                efficient_test=False,
                                opacity=0.5,
                                pre_eval=False,
                                format_only=False,
                                format_args={}):

    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = [[] for _ in model.module.collect_timesteps]
    from torch.utils.data import Subset
    if type(data_loader.dataset) is Subset:
        dataset = data_loader.dataset.dataset
    else:
        dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    if format_only:  
        imgfile_prefix_base = format_args['imgfile_prefix']
        format_args.pop('imgfile_prefix')
        
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
            
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                for idx, time_step in enumerate(model.module.collect_timesteps):
                    file_name = img_meta['ori_filename'][:-4] + f'_{str(time_step)}' + img_meta['ori_filename'][-4:]
                    if out_dir:
                        out_file = osp.join(out_dir, file_name)
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[idx],
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=opacity)
        
        for timestep, result_timestep in enumerate(result):
            if efficient_test:
                result_timestep = [np2tmp(_, tmpdir='.efficient_test') for _ in result_timestep]

            if format_only:
                imgfile_prefix = osp.join(imgfile_prefix_base, str(timestep))
                result_timestep = dataset.format_results(
                    result_timestep, indices=batch_indices, imgfile_prefix=imgfile_prefix, **format_args)
            if pre_eval:
                # TODO: adapt samples_per_gpu > 1.
                # only samples_per_gpu=1 valid now
                result_timestep = dataset.pre_eval(result_timestep, indices=batch_indices)
       
            results[timestep].extend(result_timestep)

        batch_size = len(result[0])
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_test_multi_steps(model,
                               data_loader,
                               tmpdir=None,
                               gpu_collect=False,
                               efficient_test=False,
                               pre_eval=False,
                               format_only=False,
                               format_args={}):
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = [[] for _ in model.module.collect_timesteps]
    from torch.utils.data import Subset
    if type(data_loader.dataset) is Subset:
        dataset = data_loader.dataset.dataset
    else:
        dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader.dataset))

    if format_only:  
        imgfile_prefix_base = format_args['imgfile_prefix']
        format_args.pop('imgfile_prefix')
        
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        for timestep, result_timestep in enumerate(result):
            if efficient_test:
                result_timestep = [np2tmp(_, tmpdir='.efficient_test') for _ in result_timestep]

            if format_only:
                imgfile_prefix = osp.join(imgfile_prefix_base, str(timestep))
                result_timestep = dataset.format_results(
                    result_timestep, indices=batch_indices, imgfile_prefix=imgfile_prefix, **format_args)
            if pre_eval:
                # TODO: adapt samples_per_gpu > 1.
                # only samples_per_gpu=1 valid now
                result_timestep = dataset.pre_eval(result_timestep, indices=batch_indices)

            results[timestep].extend(result_timestep)

        if rank == 0:
            batch_size = len(result[0]) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    for timestep, results_timestep in enumerate(results):
        if gpu_collect:
            results[timestep] = collect_results_gpu(results_timestep, len(dataset))
        else:
            results[timestep] = collect_results_cpu(results_timestep, len(dataset), tmpdir)
        dist.barrier()  # wait for other ranks
    return results