import os.path
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from layers.l2norm import L2Norm

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

# (300, 300, 64), (300, 300, 64), (150, 150, 64)
# (150, 150, 128), (150, 150, 128), (75, 75, 128)
# (75, 75, 256), (75, 75, 256), (75, 75, 256), (38, 38, 256)
# (38, 38, 512), (38, 38, 512), (38, 38, 512), (19, 19, 512)
# (19, 19, 512), (19, 19, 512), (19, 19, 512)
base_cfg: List[Union[int, str]] = [64, 64, 'M',
                                   128, 128, 'M',
                                   256, 256, 256, 'C',
                                   512, 512, 512, 'M',
                                   512, 512, 512]
# 'S'标志着下一个卷积stride=2
extras_cfg: List[Union[int, str]] = [256, 'S', 512,  # (19, 19, 1024) -> (19, 19, 256) -> (10, 10, 512)
                                     128, 'S', 256,  # (10, 10, 512) -> (10, 10, 128) -> (5, 5, 256)
                                     128, 256,  # (5, 5, 256) -> (5, 5, 128) -> (3, 3, 256)
                                     128, 256]  # (3, 3, 256) -> (3, 3, 128) -> (1, 1, 256)


def add_vgg(batch_norm: bool = False):
    layers = []
    in_channels = 3
    for v in base_cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # (19, 19, 512) -> (19, 19, 512)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # (19, 19, 512) -> (19, 19, 1024)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)  # (19, 19, 1024) -> (19, 19, 1024)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


def add_extras():
    layers = []

    flag = False
    in_channels = 1024
    for idx, v in enumerate(extras_cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, extras_cfg[idx + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return nn.ModuleList(layers)


class VGG(nn.Module):
    def __init__(self, cfg):
        super(VGG, self).__init__()
        self.cfg = cfg
        self.vgg_base = add_vgg()
        self.extras = add_extras()

        self.l2norm = L2Norm(n_channels=512, scale=20)
        self.reset_parameters()

    def forward(self, x):
        sources = []
        for layer_idx in range(23):
            x = self.vgg_base[layer_idx](x)
        # 至此得到第22层（conv4_3+relu）的输出
        s = self.L2Norm(x)
        sources.append(x)  # conv4_3 feature

        for layer_idx in range(23, len(self.vgg)):
            x = self.vgg[layer_idx](x)

        sources.append(x)  # conv7 feature

        for layer_idx, layer in enumerate(self.extras):
            x = F.relu(layer(x), inplace=True)
            if layer_idx % 2 == 1:
                sources.append(x)

        return tuple(sources)

    def initialize_weights_from_pretrain(self, state_dict):
        self.vgg_base.load_state_dict(state_dict, strict=False)

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg(cfg=None, pretrained=True):
    backbone = VGG(cfg)
    if pretrained:
        state_dict = torch.load('./state_dict/vgg16-397923af.pth') \
            if os.path.exists('./state_dict/vgg16-397923af.pth') \
            else load_state_dict_from_url(url=model_urls['vgg16'], model_dir='./state_dict/', map_location='cpu')
        state_dict = {k.replace('features.', ''): v for k, v in state_dict.items()}
        backbone.initialize_weights_from_pretrain(state_dict)
    return backbone


if __name__ == '__main__':
    model = vgg()
