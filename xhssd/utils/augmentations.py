"""Data Augmentation about Object Detection
目标检测数据增强"""
import random

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


# ---------------------------------------- 针对像素的数据增强: Begin ----------------------------------------
class ConvertFromInts(object):
    """convert image to float32 from int"""

    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height


class RandomContrast(object):
    """针对像素的数据增强：随机图像对比度"""

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):  # 以0.5的概率进行随机对比度的数据增强
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
            alpha = np.random.uniform(self.lower, self.upper)  # 饱和度随机因子2
            image[:, :, 1] *= alpha
        return image, boxes, labels


class RandomHue(object):
    """针对像素的基于HSV空间的数据增强：随机色调"""

    def __init__(self, delta=18.0):
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            # 规范超过范围的像素值
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomBrightness(object):
    """针对像素的基于RGB空间的数据增强：随机亮度————将像素值加上/减去同一个值"""

    def __init__(self, delta=32):
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class SwapChannels(object):
    """按照给定的交换顺序重新排列图片的通道"""

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class RandomLightingNoise:
    """针对像素的数据增强:在RGB空间内随机交换通道的值"""

    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        return image


class PhotometricDistort(object):
    """基于针对像素的数据增强方法的封装"""

    def __init__(self):
        self.pd = [RandomContrast(),  # 随机对比度
                   ConvertColor(transform='HSV'),  # 色彩空间转换 BGR->HSV
                   RandomSaturation(),  # 随机饱和度
                   RandomHue(),  # 随机色调
                   ConvertColor(current='HSV', transform='BGR'),  # 转换色彩空间 HSV -> BGR
                   RandomContrast()  # 随机对比度
                   ]
        self.rand_brightness = RandomBrightness()  # 随机亮度
        self.rand_light_noise = RandomLightingNoise()  # 随机通道交换

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if np.random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


# ---------------------------------------- 针对像素的数据增强：End ----------------------------------------

# ---------------------------------------- 针对图像的数据增强：Begin ----------------------------------------
# 包括对图像本身的改变，对标注信息的改变

class RandomMirror(object):
    """随机镜像：将图像沿着竖轴中心翻转"""

    def __call__(self, image, boxes, classes=None):
        """
        Args:
            boxes(ndarray): [[xmin, ymin, xmax, ymax], ...]
        """
        _, width, _ = image.shape
        if np.random.randint(2):
            # 图像翻转
            image = image[:, ::-1]
            boxes = boxes.copy()
            # 改变标注框
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels


class Expand(object):
    """保持宽高比不变，随机放大图像，边界框信息也随之改变。原图本身大小不变，只不过在周围加上了黑边，加上黑边后的图像的尺寸和原图是成比例的"""

    def __init__(self, mean):
        """
        Args:
            mean(tuple or list): 扩大图片时在周围填充的颜色.
        """
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if np.random.randint(2):
            # 获取图像的各个维度
            height, width, depth = image.shape
            # 随机缩放尺度
            factor = np.random.uniform(1, 4)  # expand后的图片大小是(height*factor, width*factor, depth)
            # 注意：图像原点在左上角
            left = np.random.uniform(0, width * factor - width)  # 原图最左边的位置在放大后图像中的坐标
            top = np.random.uniform(0, height * factor - height)  # 原图最顶部的位置在放大后图像中的坐标
            # 扩大后的图片维度
            expand_image = np.zeros((int(height * factor), int(width * factor), depth), dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top): int(top + height), int(left):int(left + width)] = image

            # 返回缩放后的图像
            image = expand_image

            # 重新修改边界框位置
            boxes = boxes.copy()
            boxes[:, :2] += (int(left), int(top))  # [xmin, ymin]
            boxes[:, 2:] += (int(left), int(top))  # [xmax, ymax]

        return image, boxes, labels


def intersect(boxes_a, box_b):
    """求解boxes_a中每个box和box_b的交叠的面积
    Args:
        boxes_a: Multiple boxes with size of (nums, 4), 4 means [xmin, ymin, xmax, ymax]
        box_b: a box with size of (4)
    """
    max_xy = np.minimum(boxes_a[:, 2:], box_b[2:])
    min_xy = np.maximum(boxes_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def iou(boxes_a, box_b):
    """求解boxes_a中每个box和box_b的IoU
    Args:
        boxes_a: Multiple boxes with size of (nums, 4), 4 means [xmin, ymin, xmax, ymax]
        box_b: a box with size of (4)
    """
    inter = intersect(boxes_a, box_b)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0])*(boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - inter
    return inter/union

class RandomSampleCrop(object):
    """随机裁剪：随机裁掉原图中的一部分， 然后检查边界框或者目标整体是否被裁掉，如果目标整体被裁掉，则舍弃这次随机过程"""

    def __init__(self):
        self.sample_options = {
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None)
        }

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # 随机选取一种裁剪方式
            mod = random.choice(self.sample_options)
            if mod is None:
                return image, boxes, labels
            # 最小Iou和最大IoU
            min_iou, max_iou = mod
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # 迭代50次
            for _ in range(50):
                current_img = image
                # 宽高随机采样
                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                # 对于宽高比例不当的舍弃采样
                if h / w < 0.5 or h / w > 2:
                    continue

                # 确定宽高之后，对于图像的原点位置进行采样
                left = np.random.uniform(0, width - w)
                top = np.random.uniform(0, height - h)

                # 由此确定图像所在坐标
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                # 求解原始的boxes和新的图像的IoU
                overlap = iou(boxes, rect)

                if overlap.min() < min_iou and overlap.max() > max_iou:
                    continue

                # box中心点坐标
                center = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                m1 = (rect[0] < center[:, 0]) * (rect[1] < center[:, 1])
                m2 = (rect[2] > center[:, 0]) * (rect[3] < center[:, 1])
                # m1 m2均为正时保留
                mask = m1 * m2
                if not mask.any():  # any表示任意一个元素为true，则结果为true
                    continue
                # 将中心点不在裁剪后图像范围的box去掉
                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]
                # 对于中心点在裁剪后图像区域的box的边界重新设定，原边界有可能超出了裁剪后图像区域
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2]) - rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:]) - rect[:2]
                return current_img, current_boxes, current_labels


# ---------------------------------------- 针对图像的数据增强：End ----------------------------------------

class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ConvertColor(),
            RandomSaturation(),
        ])

    def __call__(self, image, boxes=None, labels=None):
        """
        Args:
            boxes(ndarray): [[xmin, ymin, xmax, ymax], ...]
        """
        return self.augment(image)


if __name__ == '__main__':
    image = np.random.randint(0, 255, (300, 300, 3))
    # from matplotlib import pyplot as plt
    # plt.subplot(121)
    # plt.imshow(image)
    augument = SSDAugmentation()
    image, boxes, labels = augument(image)
    # plt.subplot(122)
    # plt.imshow(image)
    # plt.show()
    # print(image)
