import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.utils import get_root_logger


@SEGMENTORS.register_module()
class EncoderDecoderFreeze(EncoderDecoder):
    def __init__(self, freeze_parameters=['backbone'], **kwargs):
        super(EncoderDecoderFreeze, self).__init__(**kwargs)
        self._set_trainable_parameters(freeze_parameters)

    def _set_trainable_parameters(self, freeze_parameters=None):
        _logger = get_root_logger()
        if freeze_parameters:
            for param_name in freeze_parameters:
                model = getattr(self, param_name)
                if hasattr(model, '_set_trainable_parameters'):
                    model._set_trainable_parameters()
                else:
                    for name, param in model.named_parameters():
                        param.requires_grad = False
                _logger.info(f'Parameters in {param_name} freezed!')