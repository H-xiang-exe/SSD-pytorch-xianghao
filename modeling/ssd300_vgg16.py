import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.l2norm import L2Norm
from utils.prior_anchor import PriorAnchor
from utils.box_utils import decode
from .post_processor import PostProcessor

class SSD300_VGG16(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the added multibox conv layers.Each multibox layer branchs into:
    """

    def __init__(self, phase, size, base, extras, head, num_classes, cfg):
        """
        Args:
            phase(string): "test" or "train"
            size(int): input image size
            base: VGG16 layers for input, size of either 300 or 500
            extras: extra layers that feed to multibox loc and conf layers
            head: "multibox head" consists of loc and conf conv layers
        """
        super(SSD300_VGG16, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.location = nn.ModuleList(head[0])
        self.confidence = nn.ModuleList(head[1])
        self.prior_anchors = PriorAnchor((size, size))()
        self.post_processor= PostProcessor(cfg)

        self.L2Norm = L2Norm(512, 20)

    def forward(self, x):
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_test(x)

    def _forward_train(self, x):
        loc_preds, conf_preds = self._predict(x)
        return loc_preds, conf_preds, self.prior_anchors

    def _forward_test(self, x):
        loc_preds, conf_preds = self._predict(x)
        # ----------------------------------------------------------------------------------- #
        # 以下两部可在模型外部处理，也可在内部处理，这里暂时放在外部，此处仅作注释
        # ----------------------------------------------------------------------------------- #

        # 置信度
        scores = F.softmax(conf_preds, dim=2)
        # ----------------------------------------------------------------------------------- #
        # 解码locations
        # ----------------------------------------------------------------------------------- #
        bboxes = decode(loc_preds, self.prior_anchors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE)

        # 后处理
        detections = (scores, bboxes)
        # detections = self.post_processor(detections)
        return detections

    def _predict(self, x):
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
            # (N, 16, 38, 38) -> (N, 38, 38, 16)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())  # loc: (6, N, h, w, num_anchor*4)
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # (N, h1*w1*num_anchor1*4),(N, h2*w2*num_anchor2*4),(N, h3*w3*num_anchor3*4), ...
        loc = torch.cat([o.view(o.size(0), -1) for o in loc],
                        dim=1)  # loc: (N, h1*w1*num_anchor1*4 + h2*w2*num_anchor2*4 + ...)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], dim=1)

        # (N, h1*w1*num_anchor1 + h2*w2*num_anchor2 + h3*w3*num_anchor3, 4)
        loc.view(loc.shape[0], -1, 4)
        # print(conf.size()[0], self.num_classes)
        conf.view(conf.shape[0], -1, self.num_classes)

        output = (loc.view(loc.size()[0], -1, 4), conf.view(conf.size()[0], -1, self.num_classes))

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


def build_ssd(phase, cfg, size=300, num_classes=21, ):
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
    return SSD300_VGG16(phase, size, base_, extras_, head_, num_classes, cfg)
