import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import config


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))  # shape = (20,)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):  # (B, C, H, W)
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps  # (B, 1, H, W)
        x = torch.div(x, norm)  # (B, C, H, W)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x  # (1,20,1,1)
        return out


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the added multibox conv layers.Each multibox layer branchs into:
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        """
        Args:
            phase(string): "test" or "train"
            size(int): input image size
            base: VGG16 layers for input, size of either 300 or 500
            extras: extra layers that feed to multibox loc and conf layers
            head: "multibox head" consists of loc and conf conv layers
        """
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (config.coco, config.voc)[num_classes == 21]

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.location = nn.ModuleList(head[0])
        self.confidence = nn.ModuleList(head[1])
        self.L2Norm = L2Norm(512, 20)

    def forward(self, x):
        sources = []
        for layer_idx in range(23):
            x = self.vgg[layer_idx](x)
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

        loc = []
        conf = []
        for x, l, c in zip(sources, self.location, self.confidence):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())  # (N, 16, 38, 38) -> (N, 38, 38, 16)
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], dim=1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], dim=1)

        output = (loc.view(loc.size(0), -1, 4),
                  conf.view(conf.size(0), -1, self.num_classes))

        return output


def vgg(batch_norm: bool = False):
    # vgg16 config
    layer_cfg = [64, 64, 'M',  # (300, 300, 64), (300, 300, 64), (150, 150, 64)
                 128, 128, 'M',  # (150, 150, 128), (150, 150, 128), (75, 75, 128)
                 256, 256, 256, 'C',  # (75, 75, 256), (75, 75, 256), (75, 75, 256), (38, 38, 256)
                 512, 512, 512, 'M',  # (38, 38, 512), (38, 38, 512), (38, 38, 512), (19, 19, 512)
                 512, 512, 512]  # (19, 19, 512), (19, 19, 512), (19, 19, 512)

    layers = []
    in_channels = 3
    for v in layer_cfg:
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
    # 'S'标志着下一个卷积stride=2
    extra_cfg = [256, 'S', 512,  # (19, 19, 1024) -> (19, 19, 256) -> (10, 10, 512)
                 128, 'S', 256,  # (10, 10, 512) -> (10, 10, 128) -> (5, 5, 256)
                 128, 256,  # (5, 5, 256) -> (5, 5, 128) -> (3, 3, 256)
                 128, 256]  # (3, 3, 256) -> (3, 3, 128) -> (1, 1, 256)

    flag = False
    in_channels = 1024
    for idx, v in enumerate(extra_cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, extra_cfg[idx + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return nn.ModuleList(layers)


def detection_head(vgg, extra_layers):
    detection_cfg = [4, 6, 6, 6, 4, 4]
    num_classes = 21
    location_layer = []
    confidence_layer = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        location_layer += [nn.Conv2d(vgg[v].out_channels, detection_cfg[k] * 4, kernel_size=3, padding=1)]
        confidence_layer += [nn.Conv2d(vgg[v].out_channels, detection_cfg[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        location_layer += [nn.Conv2d(v.out_channels, detection_cfg[k] * 4, kernel_size=3, padding=1)]
        confidence_layer += [nn.Conv2d(v.out_channels, detection_cfg[k] * num_classes, kernel_size=3, padding=1)]

    return location_layer, confidence_layer


def multibox():
    pass


def build_ssd(phase, size=300, num_classes=21):
    '''
    Args:
        phase: 'train' or 'test'
        size: size of input image
        num_classes: catagory of dataset
    '''
    assert phase == 'train' or phase == 'test', f"ERROR: Phase {phase} not recognized"
    assert size == 300, f"You specified size {size}. However currently only SSD300 is supported!"

    base_ = vgg()
    extras_ = add_extras()
    head_ = detection_head(base_, extras_)
    return SSD(phase, size, base_, extras_, head_, num_classes)
