import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class MMSEGBackbone(BaseModule):
    def __init__(self, config, init_cfg=None):
        super().__init__()
        if init_cfg is not None:
            config['init_cfg'] = init_cfg
        self.backbone = MODELS.build(config)

    def forward(self, x1, x2):
        feat1 = self.backbone(x1)
        feat2 = self.backbone(x2)

        return feat1, feat2
