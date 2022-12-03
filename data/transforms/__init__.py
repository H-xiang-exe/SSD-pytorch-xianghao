from .transforms import *
from .target_transform import SSDTargetTransform

from utils.prior_anchor import PriorAnchor


def build_transform(is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand((104, 117, 123)),
            RandomSampleCrop(),
            RandomMirror(),
            ToPersentCoords(),
            Resize(size=300),
            SubstractMeans((104, 117, 123)),
            ToTensor()
        ]
    else:
        transform = [
            Resize(size=300),
            SubstractMeans((104, 117, 123)),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    center_form_priors = PriorAnchor((300, 300))()
    return SSDTargetTransform(center_form_priors, cfg.MODEL.CENTER_VARIANCE, cfg.MODEL.SIZE_VARIANCE,
                              cfg.MODEL.THRESHOLD)