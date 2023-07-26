import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmcv.utils import Config

from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from .encoder_decoder_diffusion import EncoderDecoderDiffusion


@SEGMENTORS.register_module()
class EncoderDecoderDiffusionEnsemble(EncoderDecoderDiffusion):
    def __init__(self,
                 ensemble_model_cfg=None,
                 ensemble_model_checkpoint=None,
                 ensemble_mode='sum',
                 **kwargs):
        super(EncoderDecoderDiffusionEnsemble, self).__init__(**kwargs)
        if ensemble_model_cfg is not None:
            self.ensemble_model_cfg = Config.fromfile(ensemble_model_cfg)
            ensemble_model = self.ensemble_model_cfg.model
            self.ensemble_model = self.init_ensemble_model(ensemble_model, ensemble_model_checkpoint)
        else:
            self.ensemble_model=None
        self.ensemble_mode = ensemble_mode

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
            seg_logits = self.slide_inference(img, img_meta, rescale)
            
        else:
            seg_logits = self.whole_inference(img, img_meta, rescale)
            # seg_logit_ensemble = self.ensemble_model.whole_inference(img, img_meta, rescale)
        if self.ensemble_model is not None:
            if self.test_cfg.mode == 'slide':
                seg_logit_ensemble = self.ensemble_model.slide_inference(img, img_meta, rescale)
            else:
                seg_logit_ensemble = self.ensemble_model.whole_inference(img, img_meta, rescale)
            output_ensemble = F.softmax(seg_logit_ensemble, dim=1)
            output_ensemble_background = torch.zeros([output_ensemble.shape[0], 1,
                                                      output_ensemble.shape[2], output_ensemble.shape[3]],
                                                      device=output_ensemble.device)
            output_ensemble = torch.cat([output_ensemble_background, output_ensemble], dim=1)
        else:
            output_ensemble=None
        if self.ensemble_mode == 'sum':
            return self.sum_ensemble(seg_logits, img_meta, output_base=output_ensemble)
        elif self.ensemble_mode == 'alpha':
            # TODO: finish alpha
            raise NotImplementedError
        else:
            raise NotImplementedError

    
    def sum_ensemble(self, seg_logits, img_meta, output_base=None):
        outputs = []
        for i, seg_logit in enumerate(seg_logits):
            output = F.softmax(seg_logit, dim=1)
            if output_base is not None:
                output = 1/2 * (output + output_base)
            flip = img_meta[0]['flip']
            if flip:
                flip_direction = img_meta[0]['flip_direction']
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    output = output.flip(dims=(3, ))
                elif flip_direction == 'vertical':
                    output = output.flip(dims=(2, ))
            if i > 0:
                output = (output + outputs[-1])
            outputs.append(output)
        return outputs