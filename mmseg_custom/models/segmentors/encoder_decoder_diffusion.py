import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from .encoder_decoder_freeze import EncoderDecoderFreeze
from mmseg.utils import get_root_logger


@SEGMENTORS.register_module()
class EncoderDecoderDiffusion(EncoderDecoderFreeze):
    def __init__(self, **kwargs):
        super(EncoderDecoderDiffusion, self).__init__(**kwargs)
        self.collect_timesteps = self.decode_head.collect_timesteps

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        outs = self._decode_head_forward_test(x, img_metas)
        outs_new = []
        for out in outs:
            outs_new.append(resize(
                            input=out,
                            size=img.shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners))
        return outs_new

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = [img.new_zeros((batch_size, num_classes, h_img, w_img))
                 for _ in self.collect_timesteps]
        count_mats = [img.new_zeros((batch_size, 1, h_img, w_img))
                     for _ in self.collect_timesteps]
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logits = self.encode_decode(crop_img, img_meta)

                for timestep_idx, crop_seg_logit in enumerate(crop_seg_logits):
                    preds[timestep_idx] += F.pad(crop_seg_logit,
                                                 (int(x1), int(preds[timestep_idx].shape[3] - x2), int(y1),
                                                     int(preds[timestep_idx].shape[2] - y2)))

                    count_mats[timestep_idx][:, :, y1:y2, x1:x2] += 1
        for count_mat in count_mats:
            assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            for i, count_mat in enumerate(count_mats):
                count_mats[i] = torch.from_numpy(
                    count_mat.cpu().detach().numpy()).to(device=img.device)
        for i, pred in enumerate(preds):
            preds[i] = pred / count_mats[i]
        if rescale:
            # remove padding area
            resize_shape = img_meta[0]['img_shape'][:2]
            for i, pred in enumerate(preds):
                pred = pred[:, :, :resize_shape[0], :resize_shape[1]]
                pred = resize(
                    pred,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
                preds[i] = pred
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logits = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                # remove padding area
                for i, seg_logit in enumerate(seg_logits):
                    resize_shape = img_meta[0]['img_shape'][:2]
                    seg_logits[i] = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            for i, seg_logit in enumerate(seg_logits):
                seg_logits[i] = resize(
                    seg_logit,
                    size=size,
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)

        return seg_logits

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logits = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logits = self.whole_inference(img, img_meta, rescale)
        outputs = []
        for i, seg_logit in enumerate(seg_logits):
            if self.num_classes == 1:
                output = F.sigmoid(seg_logit)
            else:
                output = F.softmax(seg_logit, dim=1)
            flip = img_meta[0]['flip']
            if flip:
                flip_direction = img_meta[0]['flip_direction']
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    output = output.flip(dims=(3, ))
                elif flip_direction == 'vertical':
                    output = output.flip(dims=(2, ))
            outputs.append(output)
        return outputs

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logits = self.inference(img, img_meta, rescale)
        seg_preds = []
        for seg_logit in seg_logits:
            if self.num_classes == 1:
                seg_pred = (seg_logit >
                            self.decode_head.threshold).to(seg_logit).squeeze(1)
            else:
                seg_pred = seg_logit.argmax(dim=1)
            if torch.onnx.is_in_onnx_export():
                # our inference backend only support 4D output
                seg_pred = seg_pred.unsqueeze(0)
                return seg_pred
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            seg_preds.append(seg_pred)
        return seg_preds
        
    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logits = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logits = self.inference(imgs[i], img_metas[i], rescale)
            for t, seg_logit in enumerate(seg_logits):
                seg_logits[t] += cur_seg_logits[t]
        seg_preds = []
        for t, seg_logit in enumerate(seg_logits):
            seg_logit /= len(imgs)
            seg_pred = seg_logit.argmax(dim=1)
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            seg_preds.append(seg_pred)
        return seg_preds