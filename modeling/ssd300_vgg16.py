import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.l2norm import L2Norm
from utils.prior_anchor import PriorAnchor
from utils import box_utils
from modeling.boxhead.post_processor import PostProcessor

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
