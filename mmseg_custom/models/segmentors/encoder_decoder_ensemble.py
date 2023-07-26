import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmcv.utils import Config

from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class EncoderDecoderEnsemble(EncoderDecoder):
    def __init__(self, ensemble_model_cfg, ensemble_model_checkpoint, **kwargs):
        super(EncoderDecoderEnsemble, self).__init__(**kwargs)
        self.ensemble_model_cfg = Config.fromfile(ensemble_model_cfg)
        ensemble_model = self.ensemble_model_cfg.model
        self.ensemble_model = self.init_ensemble_model(ensemble_model, ensemble_model_checkpoint)

    def init_ensemble_model(self, ensemble_model, ensemble_model_checkpoint=None):
        ensemble_model = builder.build_segmentor(ensemble_model)
        if ensemble_model_checkpoint is not None:
            load_checkpoint(ensemble_model, ensemble_model_checkpoint)
        return ensemble_model

    
    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
            seg_logit_ensemble = self.ensemble_model.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
            seg_logit_ensemble = self.ensemble_model.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        output_ensemble = F.softmax(seg_logit_ensemble, dim=1)
        output = 1/2 * (output + output_ensemble)
        
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output