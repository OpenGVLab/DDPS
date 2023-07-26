from mmcv.runner.checkpoint import load_checkpoint

from mmseg.models.backbones import MobileNetV2
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger

@BACKBONES.register_module()
class MobileNetV2CustomInitWeights(MobileNetV2):
    def init_weights(self):
        if isinstance(self.init_cfg, list):
            super(MobileNetV2CustomInitWeights, self).init_weights()
        elif self.init_cfg.get('type', None) == 'Pretrained':
            pretrained = self.init_cfg['checkpoint']
            if isinstance(pretrained, str):
                logger = get_root_logger()
                load_checkpoint(self, pretrained, strict=False, logger=logger,
                                revise_keys=[(r'^module\.', ''), (r'^backbone\.', '')])
        else:
            raise NotImplementedError
