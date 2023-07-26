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
from .diffusion import q_pred, alpha_schedule_torch, cos_alpha_schedule_torch, q_posterior


NOISE_SCHEDULES = {
    'linear': alpha_schedule_torch,
    'cos': cos_alpha_schedule_torch
}


@HEADS.register_module()
class SegformerHeadUnetFCHeadMultiStep(BaseDecodeHead):
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
                 dim_mults=[1, 2, 4],
                 cat_embedding_dim=16,
                 diffusion_timesteps=20,
                 inference_timesteps=None,
                 collect_timesteps=[19],  # 19 means last step
                 guidance_scale=1.,
                 backbone_drop_out_ratio=0.,
                 alpha_schedule='linear',
                 inference_mode='q_pred',
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
        self.diffusion_timesteps = diffusion_timesteps
        self.inference_timesteps = diffusion_timesteps if inference_timesteps is None else inference_timesteps
        
        collect_timesteps.sort()
        self.collect_timesteps = collect_timesteps

        self.guidance_scale = guidance_scale
        self.backbone_drop_out_ratio = backbone_drop_out_ratio
        at, bt, att, btt = NOISE_SCHEDULES[alpha_schedule](self.diffusion_timesteps, self.num_classes)
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        self.register_buffer('log_cumprod_at', log_cumprod_at)
        self.register_buffer('log_cumprod_bt', log_cumprod_bt)

        log_at = torch.log(at)
        log_bt = torch.log(bt)
        self.register_buffer('log_at', log_at)
        self.register_buffer('log_bt', log_bt)
        
        if inference_mode == 'q_posterior': 
            self.diffusion_loop = self.diffusion_loop_posterior

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
        logger.info(f'Trainable parameters in SegformerHeadUnetFCHeadMultiStep: {str(trainable_parameters)}')

    def sample_t(self, batch_size):
        t = torch.randint(0, self.diffusion_timesteps, [batch_size], device=self.device)
        return t

    def get_timesteps(self):
        num_inference_steps = min(self.diffusion_timesteps, self.inference_timesteps)
        timesteps = list(range(
            0, self.diffusion_timesteps, self.diffusion_timesteps // num_inference_steps
        ))
        if timesteps[-1] != self.diffusion_timesteps - 1:
            timesteps = timesteps + [self.diffusion_timesteps-1]
        return timesteps

    def diffusion_loop(self, out):
        B, C, H, W = out.shape
        condition_embed = out
        mask = torch.randint(0, self.num_classes, [B, H, W], device=self.device).long()
        output_collect = []
        for i in self.get_timesteps():
            # timestep = (self.diffusion_timesteps - i - 1)
            t = (torch.ones([B], device=self.device) * i).long()
            noise_step = (torch.ones([B], device=self.device) * (self.diffusion_timesteps - i - 1)).long()
            if i != 0:
                mask = q_pred(out, noise_step, self.diffusion_timesteps, self.num_classes,
                              self.log_cumprod_at, self.log_cumprod_bt)
            mask_embedding = self.embed(mask).permute(0, 3, 1, 2)  # [B, c, H, W]
            out = torch.cat([condition_embed, mask_embedding], dim=1)  # [B, C+c, H, W]
            out = self.unet(out, t)
            out = self.cls_seg(out)
            
            if self.guidance_scale != 1:
                cfg = self.guidance_scale
                out_uncond = torch.cat([torch.zeros_like(condition_embed), mask_embedding], dim=1)
                out_uncond = self.unet(out_uncond, t)
                out_uncond = self.cls_seg(out_uncond)
                out = out * (1+cfg) - cfg * out_uncond
                # out = out_uncond + (out - out_uncond) * self.guidance_scale
            
            if i in self.collect_timesteps:
                output_collect.append(out)
            if i != (self.diffusion_timesteps - 1):
                out = out.argmax(dim=1)
        return out, output_collect

    def diffusion_loop_posterior(self, out):
        B, C, H, W = out.shape
        condition_embed = out
        mask = torch.randint(0, self.num_classes, [B, H, W], device=self.device).long()
        output_collect = []
        for i in self.get_timesteps():
            # timestep = (self.diffusion_timesteps - i - 1)
            t = (torch.ones([B], device=self.device) * i).long()
            noise_step = (torch.ones([B], device=self.device) * (self.diffusion_timesteps - i - 1)).long()
            if i != 0:
                mask = q_posterior(out, mask, noise_step, self.diffusion_timesteps, self.num_classes,
                                   self.log_cumprod_at, self.log_cumprod_bt, self.log_at, self.log_bt)
            mask_embedding = self.embed(mask).permute(0, 3, 1, 2)  # [B, c, H, W]
            out = torch.cat([condition_embed, mask_embedding], dim=1)  # [B, C+c, H, W]
            out = self.unet(out, t)
            out = self.cls_seg(out)
            if i in self.collect_timesteps:
                output_collect.append(out)
            if i != (self.diffusion_timesteps - 1):
                out = out.argmax(dim=1)
        return out, output_collect

    def forward(self, inputs, gt=None):
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
        
        if self.training and self.backbone_drop_out_ratio > 0:
            is_cond = torch.rand(size=(out.shape[0], *[1] * len(out.shape[1:])),
                                 device=out.device) >= self.backbone_drop_out_ratio
            out = out * is_cond.float()
        
        
        # TODO: fix timesteps to 0
        if self.training:
            B, C, H, W = out.shape
            t = self.sample_t(B)  # [B], t=0 means random mask
            multi_step = (t != 0)  # [B], not predict from random mask
            B_multi_step = None
            if multi_step.sum() > 0:
                with torch.no_grad():  # diffusion predict for t=0
                    out_select = out[multi_step, :, :, :].clone().detach()
                    B_multi_step = out_select.shape[0]
                    t_select = torch.zeros([B_multi_step], device=self.device).long()
                    mask_rand = torch.randint(0, self.num_classes, [B_multi_step, H, W], device=self.device).long()
                    mask_rand_embedding = self.embed(mask_rand).permute(0, 3, 1, 2)  # [B_multi_step, c, H, W]
                    out_select = torch.cat([out_select, mask_rand_embedding], dim=1)  # [B_multi_step, C+c, H, W]
                    out_select = self.unet(out_select, t_select)
                    out_select = self.cls_seg(out_select).softmax(dim=1).argmax(dim=1).long()  # [B_multi_step, H, W]
            content = gt
            x_interpolate = F.interpolate(content.float(), [H, W], mode='nearest').long().squeeze(1)  # [B, H, W]
            if B_multi_step is not None:
                x_interpolate[multi_step, ...] = out_select
            noise_step = self.diffusion_timesteps - t - 1  # t=0 means add 19 steps noise
            mask = q_pred(x_interpolate, noise_step.long(), self.diffusion_timesteps, self.num_classes,
                          self.log_cumprod_at, self.log_cumprod_bt)
            mask_embedding = self.embed(mask).permute(0, 3, 1, 2)  # [B, c, H, W]
            out = torch.cat([out, mask_embedding], dim=1)  # [B, C+c, H, W]
            out = self.unet(out, t)
            out = self.cls_seg(out)
            return out
        else:
            out, output_collect = self.diffusion_loop(out)
            return output_collect

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self(inputs, gt_semantic_seg)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_one_step(self, inputs, content, timestep):
        assert timestep < self.diffusion_timesteps and timestep >= 0
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
        x_interpolate = F.interpolate(content.float(), [H, W], mode='nearest').long().squeeze(1)  # [B, H, W]
        t = (torch.ones([B], device=self.device) * timestep).long()
        noise_step = (self.diffusion_timesteps - t - 1).long()
        mask = q_pred(x_interpolate, noise_step, self.diffusion_timesteps, self.num_classes,
                      self.log_cumprod_at, self.log_cumprod_bt)
        mask_embedding = self.embed(mask).permute(0, 3, 1, 2)  # [B, c, H, W]
        out = torch.cat([out, mask_embedding], dim=1)  # [B, C+c, H, W]
        out = self.unet(out, t)
        out = self.cls_seg(out)
        return out
