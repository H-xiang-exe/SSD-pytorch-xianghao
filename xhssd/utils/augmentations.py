import numpy as np

import torch

class Compose(object):
    """Compose serval augmentations together
    Args:
        transforms(List[Transform]): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class ConvertFromInts(object):
    """convert image to float32 from int"""
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:,0] *= width
        boxes[:,2] *= width
        boxes[:,1] *= height
        boxes[:,3] *= height

class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117,123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),

        ])