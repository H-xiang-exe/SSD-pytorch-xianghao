"""Data Augmentation about Object Detection
目标检测数据增强"""
import random

import numpy as np
import cv2
import random

import torch
import torchvision
from matplotlib import pyplot as plt


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


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, boxes, labels


class ToPersentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        """
        Args:
            boxes(ndarray): (nums, 4), 4 means [xmin, ymin, xmax, ymax]
        """
        boxes = np.array(boxes, dtype='float32')
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 1] /= height
        boxes[:, 2] /= width
        boxes[:, 3] /= height
        return image, boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


def _inter_union(boxes_a, boxes_b):
    """求解boxes_a中每个box和boxes_b的交叠的面积
    Args:
        boxes_a: Multiple boxes with size of (N, 4), 4 means [xmin, ymin, xmax, ymax]
        box_b: Multiple boxes with size of (M, 4), 4 means [xmin, ymin, xmax, ymax]
    Return: intersection and union between boxes_a and boxes_b

    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # (N)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # (M)

    inter_lt = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])  # (N, M, 2)
    inter_rb = np.minimum(boxes_a[:, 2:], boxes_b[:, 2:])  # (N, M, 2)

    inter_wh = np.clip((inter_rb - inter_lt), a_min=0, a_max=np.inf)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # (N, M)

    union_area = area_a[:, None] + area_b - inter_area
    return inter_area, union_area


def iou(boxes_a, boxes_b):
    """
    求解boxes_a中每个box和boxes_b中每个box的IoU

    Args:
        boxes_a(np.ndarray[N, 4]): Multiple boxes with size of (nums, 4), 4 means [xmin, ymin, xmax, ymax]
        boxes_b(np.ndarray[N, 4])

    Returns:
        iou(np.ndarray[N,M])
    """
    inter_area, union_area = _inter_union(boxes_a, boxes_b)
    iou = inter_area / union_area  # (N, M)
    return iou


# ----------------------------------------------------------------------------------
#  针对像素的数据增强
# ----------------------------------------------------------------------------------
class ConvertFromInts(object):
    """convert image to float32 from int"""

    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


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
    """针对像素的基于RGB空间的数据增强: 随机亮度————将像素值加上/减去同一个值"""

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
        return image, boxes, labels


class PhotometricDistort(object):
    """基于针对像素的数据增强方法的封装"""

    def __init__(self):
        self.pd = [RandomContrast(),  # 随机对比度
                   ConvertColor(transform='HSV'),  # 色彩空间转换 BGR->HSV
                   RandomSaturation(),  # 随机饱和度
                   RandomHue(),  # 随机色调
                   # 转换色彩空间 HSV -> BGR
                   ConvertColor(current='HSV', transform='BGR'),
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


class SubstractMeans(object):
    def __init__(self, mean) -> None:
        """

        Args:
            mean (_type_): _description_
        """
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


# ----------------------------------------------------------------------------------------
#  针对图像的数据增强——包括对图像本身的改变，对标注信息的改变
# ----------------------------------------------------------------------------------------
class RandomMirror(object):
    """随机镜像：将图像沿着竖轴中心翻转"""

    def __call__(self, image, boxes, labels=None):
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
            # expand后的图片大小是(height*factor, width*factor, depth)
            factor = np.random.uniform(1, 4)
            # 注意：图像原点在左上角
            left = np.random.uniform(
                0, width * factor - width)  # 原图最左边的位置在放大后图像中的坐标
            top = np.random.uniform(
                0, height * factor - height)  # 原图最顶部的位置在放大后图像中的坐标
            # 扩大后的图片维度
            expand_image = np.zeros(
                (int(height * factor), int(width * factor), depth), dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top): int(top + height), int(left):int(left + width)] = image

            # 返回缩放后的图像
            image = expand_image

            # 重新修改边界框位置
            boxes = boxes.copy()
            boxes[:, :2] += (int(left), int(top))  # [xmin, ymin]
            boxes[:, 2:] += (int(left), int(top))  # [xmax, ymax]

        return image, boxes, labels


class RandomSampleCrop(object):
    """随机裁剪：随机裁掉原图中的一部分， 然后检查边界框或者目标整体是否被裁掉，如果目标整体被裁掉，则舍弃这次随机过程"""

    def __init__(self):
        self.sample_options = (
            # using entire original image
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None)
        )

    def __call__(self, image, boxes=None, labels=None):
        # guard against no boxes
        if boxes is not None and boxes.shape[0] == 0:
            return image, boxes, labels
        height, width, _ = image.shape
        while True:
            # 随机选取一种裁剪方式
            mode = self.sample_options[np.random.randint(0, len(self.sample_options))]
            if mode is None:
                return image, boxes, labels
            # 最小Iou和最大IoU
            min_iou, max_iou = mode
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

                overlap = iou(boxes, np.expand_dims(rect, axis=0))

                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cur the crop from the image
                current_img = current_img[rect[1]:rect[3], rect[0]: rect[2], :]
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
                current_boxes[:, :2] = np.maximum(
                    current_boxes[:, :2], rect[:2]) - rect[:2]
                current_boxes[:, 2:] = np.minimum(
                    current_boxes[:, 2:], rect[2:]) - rect[2:]
                return current_img, current_boxes, current_labels


class Resize(object):
    def __init__(self, size=300) -> None:
        """Resize image shape to (size, size)

        Args:
            size (int): image shape after resizing
        """
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class TestTransform(object):
    def __init__(self, size=(300, 300), mean=(104, 117, 123), letterbox_image=True):
        """用于测试集数据预处理
        @Args:
            size(tuple): 图片要变成的目标尺寸，模型的图片输入尺寸
            letterbox_image(bool): True表示在resize时把长边缩放到size，短边和长边同比例缩放，其余的地方填充灰边，False表示直接resize
            mean(tuple): 使图片三个通道的像素能够有正有负
        """
        self.size = size
        self.mean = mean
        self.letterbox_image = letterbox_image

    def __call__(self, image, boxes=None, labels=None):
        """
        Return:
            new_image(np.ndarray):
            boxes: (num_boxes, 4)
            labels: (num_boxes, 1)
        """
        # 计算原始图片的宽和高
        origin_image_h, origin_image_w, _ = image.shape
        # print(f'origin_h: {origin_image_h}, origin_w: {origin_image_w}')
        # 计算目标图片的宽和高
        target_h, target_w = self.size
        # print(f'target_h: {target_h}, target_w: {target_w}')
        if self.letterbox_image:
            scale = min(target_h / origin_image_h, target_w / origin_image_w)

            # 获得原始图片按长边缩放后的高和宽
            new_image_h = int(origin_image_h * scale)
            new_image_w = int(origin_image_w * scale)
            # print(f'new_h: {new_image_h}, new_w: {new_image_w}')
            # 缩放图片
            new_image = cv2.resize(image, (new_image_w, new_image_h),
                                   interpolation=cv2.INTER_CUBIC)  # 双三次插值，这里要注意，resize时的dsize的顺序是(w,h)
            # 计算在h方向上应该补充的灰边的高度和宽度
            dh = target_h - new_image_h
            dw = target_w - new_image_w
            # 用cv填充灰边
            top = dh // 2
            bottom = dh - top
            left = dw // 2
            right = dw - left
            # 注意，这里的通道顺序是BGR
            GREY = [128, 128, 128]
            # RED = [0, 0, 255]
            new_image = cv2.copyMakeBorder(new_image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                                           value=GREY)
        else:
            new_image = cv2.resize(image, (target_h, target_w), interpolation=cv2.INTER_CUBIC)
        new_image = new_image.astype(np.float32)
        new_image -= self.mean
        return new_image, boxes, labels
