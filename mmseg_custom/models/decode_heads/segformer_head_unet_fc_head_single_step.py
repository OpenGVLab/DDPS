import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner.checkpoint import load_checkpoint
import numpy as np

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.utils import get_root_logger

from .unet import UnetTimeEmbedding
from .diffusion import q_pred, alpha_schedule_torch


@HEADS.register_module()
class SegformerHeadUnetFCHeadSingleStep(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self,
                 dim=256,
                 out_dim=256,
                 unet_channels=1024,
                 dim_mults=[1,2,4],
                 cat_embedding_dim=16,
                 #  guidance_scale=1.,
                 backbone_drop_out_ratio=0.,
                 interpolate_mode='bilinear',
                 pretrained=None,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.pretrained = pretrained
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.interpolate_mode = interpolate_mode

        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        self.unet = UnetTimeEmbedding(dim=dim,
                                      out_dim=out_dim,
                                      channels=unet_channels,
                                      dim_mults=dim_mults)
        
        self.conv_seg = None
        self.conv_seg_new = nn.Conv2d(out_dim, self.num_classes, kernel_size=1)

        self.embed = nn.Embedding(self.num_classes, cat_embedding_dim)

        # self.guidance_scale = guidance_scale
        self.backbone_drop_out_ratio = backbone_drop_out_ratio

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg_new(feat)
        return output

    def init_weights(self):
        pretrained = self.pretrained
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger,
                            revise_keys=[(r'^module\.', ''), (r'^decode_head\.', '')])

    def _set_trainable_parameters(self):
        logger = get_root_logger()
        trainable_parameters = []
        for name, param in self.named_parameters():
            if name.startswith('unet') or name.startswith('conv_seg_new'):
                param.requires_grad = True
                trainable_parameters.append(name)
            else:
                param.requires_grad = False
        logger.info(f'Trainable parameters in SegformerHeadUnetFCHeadSingleStep: {str(trainable_parameters)}')

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        self.device = inputs[0].device
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        B, C, H, W = out.shape
        t = torch.zeros([B], device=self.device)
        mask = torch.randint(0, self.num_classes, [B, H, W], device=self.device).long()
        mask_embedding = self.embed(mask).permute(0, 3, 1, 2)  # [B, c, H, W]
        out = torch.cat([out, mask_embedding], dim=1)  # [B, C+c, H, W]
        out = self.unet(out, t)
        out = self.cls_seg(out)
        return out
