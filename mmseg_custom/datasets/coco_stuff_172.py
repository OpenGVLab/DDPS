import os.path as osp
from prettytable import PrettyTable
from collections import OrderedDict

import mmcv
import numpy as np
from mmseg.datasets.builder import DATASETS, PIPELINES
from mmseg.datasets.custom import CustomDataset
from mmcv.utils import print_log

from mmseg_custom.core.evaluation.metrics import eval_metrics, pre_eval_to_metrics


@PIPELINES.register_module()
class LoadAnnotationsCOCOStuff172(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # calculate background
        if self.reduce_zero_label == False:  # background is 255, make it zero
            background = (gt_semantic_seg == 255)
            gt_semantic_seg[background] = 0
            gt_semantic_seg = gt_semantic_seg + 1
            gt_semantic_seg[background] = 0
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@DATASETS.register_module()
class COCOStuff172Dataset(CustomDataset):
    """COCO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    CLASSES = (
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
        'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
        'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass',
        'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat',
        'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
        'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
        'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
        'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood')

    PALETTE = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
               [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
               [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
               [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
               [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
               [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
               [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
               [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
               [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
               [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
               [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
               [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
               [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
               [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
               [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
               [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
               [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
               [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
               [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
               [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
               [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
               [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
               [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
               [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
               [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
               [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
               [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
               [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
               [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
               [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
               [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
               [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
               [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
               [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
               [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
               [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
               [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
               [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
               [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
               [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
               [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
               [64, 192, 96], [64, 160, 64], [64, 64, 0]]

    def __init__(self, **kwargs):
        super(COCOStuff172Dataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='_labelTrainIds.png', **kwargs)

    def evaluate_diffusion(self,
                           results_timesteps,
                           collect_timesteps,
                           metric='mIoU',
                           logger=None,
                           gt_seg_maps=None,
                           **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        collect_metrics = ['IoU']
        eval_results = {}
        ret_metrics_timesteps = []
        for results in results_timesteps:
            # test a list of files
            if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                    results, str):
                if gt_seg_maps is None:
                    gt_seg_maps = self.get_gt_seg_maps()
                num_classes = len(self.CLASSES)
                ret_metrics = eval_metrics(
                    results,
                    gt_seg_maps,
                    num_classes,
                    self.ignore_index,
                    metric,
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label)
            # test a list of pre_eval_results
            else:
                ret_metrics = pre_eval_to_metrics(results, metric)
            ret_metrics_timesteps.append(ret_metrics)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_summary = OrderedDict()
        ret_metrics_class = OrderedDict()
        for timestep_idx, ret_metrics in enumerate(ret_metrics_timesteps):
            # summary table
            # each class table
            for ret_metric, ret_metric_value in ret_metrics.items():
                if ret_metric in collect_metrics:
                    if ret_metrics_summary.get(ret_metric, False):
                        ret_metrics_summary[ret_metric].append(np.round(np.nanmean(ret_metric_value) * 100, 2))
                        ret_metrics_class[ret_metric].append(np.round(ret_metric_value * 100, 2))
                    else:
                        ret_metrics_summary[ret_metric] = [np.round(np.nanmean(ret_metric_value) * 100, 2)]
                        ret_metrics_class[ret_metric] = [np.round(ret_metric_value * 100, 2)]
        ret_metrics_class.pop('aAcc', None)
        for ret_metric, ret_metric_value in ret_metrics_class.items():
            ret_metric_value = np.stack(ret_metric_value, axis=0).T  # [num_classes, num_steps]
            ret_metrics_class[ret_metric] = ret_metric_value

        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        collect_timesteps_str = ' ' + ','.join(map(str, collect_timesteps))
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            if key != 'Class':
                val_new = []
                for val_data in val:
                    val_new.append(','.join(map(str, val_data)))
                class_table_data.add_column(key + collect_timesteps_str, val_new)
            else:
                class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            val = ','.join(map(str, val))
            if key == 'aAcc':
                summary_table_data.add_column(key + collect_timesteps_str, [val])
            else:
                summary_table_data.add_column('m' + key + collect_timesteps_str, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = [round(step_value / 100.0, 4) for step_value in value]
            else:
                eval_results['m' + key] = [round(step_value / 100.0, 4) for step_value in value]
        eval_results['copy_paste'] = ','.join(map(str, ret_metrics_summary['IoU']))
        # ret_metrics_class.pop('Class', None)
        # for key, value in ret_metrics_class.items():
        #     eval_results.update({
        #         key + '.' + str(name): value[idx] / 100.0
        #         for idx, name in enumerate(class_names)
        #     })

        return eval_results
