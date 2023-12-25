import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import  ConvModule, build_conv_layer
from mmengine.model import BaseModule
from mmengine.model import constant_init, kaiming_init
from mmseg.registry import MODELS


@MODELS.register_module()
class ConcatFusion(nn.Module):
    """
        use concat to fuse features
        "GN": nn.GroupNorm(32, channels)
        'BN': nn.BatchNorm2d
        'LN': LayerNorm
    """
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 ):
        super(ConcatFusion, self).__init__()
        
        self.fusion = nn.ModuleList([])

        for idx, in_channel in enumerate(in_channels):
            conv = nn.Sequential(
                    nn.Conv2d(in_channel*2, in_channel, kernel_size=1, stride=1, padding=0, bias=False),
                    torch.nn.BatchNorm2d(in_channel),
                    torch.nn.ReLU(inplace=True),
                )
            self.fusion.append(conv.cuda())
    
    def forward(self, x1, x2):
        
        y = []
        for idx, [feat1, feat2] in enumerate(zip(x1, x2)):
            feat = torch.cat((feat1, feat2), 1)
            feat = self.fusion[idx](feat)
            y.append(feat)

        out = tuple(y)
        
        return out