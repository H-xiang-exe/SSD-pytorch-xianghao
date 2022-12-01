from .transforms import *
from .target_transform import SSDTargetTransform


def build_transform(is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            ToAbsoluteCoords(),
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


def build_target_transform():
    return SSDTargetTransform()
