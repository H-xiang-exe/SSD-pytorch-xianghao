"""Data Augmentation about Object Detection
目标检测数据增强"""
import numpy as np
import cv2

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

class RandomContrast(object):
    """针对像素的数据增强：随机图像对比度"""
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2): # 以0.5的概率进行随机对比度的数据增强
            # 生成随机因子
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels

class ConvertColor(object):
    """图像色彩空间的转换: RGB, HSV"""
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels

class RandomSaturation(object):
    """针对像素的基于HSV空间的数据增强： 随机饱和度"""
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):  # 以0.5的概率进行数据增强
            alpha = np.random.uniform(self.lower, self.upper)  # 饱和度随机银子
            image[:, :, 1] *= alpha
        return image, boxes, labels

class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117,123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),

        ])