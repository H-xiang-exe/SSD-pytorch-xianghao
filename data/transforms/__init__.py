from .transforms import *
from .target_transform import SSDTargetTransform

from utils.prior_anchor import PriorAnchor


def build_transform(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ToPersentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubstractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubstractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    center_form_priors = PriorAnchor(cfg)()
    return SSDTargetTransform(center_form_priors, cfg.MODEL.CENTER_VARIANCE, cfg.MODEL.SIZE_VARIANCE,
                              cfg.MODEL.THRESHOLD)
