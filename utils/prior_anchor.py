import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import sqrt

import torch


class PriorAnchor(object):
    def __init__(self, input_shape):
        """大小为input_shape的图片，对应于6个feature level(每个level对应着把原图划分为不同数量的网格)，

        Args:
            input_shape(tuple): 输入的图片的大小
            min_size(int): 规定了当前尺度(scale)下的最小框尺寸
            max_size(int): 规定了当前尺度(scale)下的大框尺寸
            aspect_ratios(list): 规定了6个feature level上每一个位置对应的多个框的纵横比。eg.[[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
        """
        super(PriorAnchor, self).__init__()
        # 获得输入图片的大小
        self.img_height, self.img_width = input_shape
        # 先验框的短边
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        # 先验框的长边
        self.max_sizes = [60, 111, 162, 213, 264, 315]

        # 特征层网格size
        self.feature_heights = [38, 19, 10, 5, 3, 1]
        self.feature_widths = [38, 19, 10, 5, 3, 1]

        self.aspect_rations = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]

    def __call__(self):
        """
        获得对应于某个layer的anchor box的集合
        """
        output_boxes = []

        # 求解每个level的feature map对应在原图的先验框的坐标集合
        for idx, (feature_h, feature_w, min_size, max_size) in enumerate(
                zip(self.feature_heights, self.feature_widths, self.min_sizes, self.max_sizes)):
            boxes_height = []
            boxes_width = []
            for ar in self.aspect_rations[idx]:
                if ar == 1:
                    # aspect ratio = 1:1, height=width= min_size
                    boxes_height.append(self.min_sizes[idx])
                    boxes_width.append(self.min_sizes[idx])
                    # aspect ratio = 1:1, height=width=sqrt(min_size*max_size)
                    prime = sqrt(self.min_sizes[idx] * self.max_sizes[idx])
                    boxes_height.append(prime)
                    boxes_width.append(prime)
                else:
                    # aspect ratio = ar, height = min_size/ar, width = min_size*ar
                    boxes_height.append(self.min_sizes[idx] / np.sqrt(ar))
                    boxes_width.append(self.min_sizes[idx] * np.sqrt(ar))
                    # aspect ratio = ar, height = min_size*ar, width = min_size/ar
                    boxes_height.append(self.min_sizes[idx] * np.sqrt(ar))
                    boxes_width.append(self.min_sizes[idx] / np.sqrt(ar))

            boxes_width = np.array(boxes_width)
            boxes_height = np.array(boxes_height)
            # feature的尺寸为h*w，对应的原图被划分为h*w个网格
            # 求解用于detection的feature在原图对应的网格的边长
            step_w = self.img_width / feature_w
            step_h = self.img_height / feature_h

            # 获得网格中心点的横坐标和纵坐标
            linx = np.linspace(0.5 * step_w, self.img_width - 0.5 * step_w, feature_w)
            liny = np.linspace(0.5 * step_h, self.img_height - 0.5 * step_h, feature_h)
            centers_x, centers_y = np.meshgrid(linx, liny)  # (h*w)

            # 每个网格应该有的anchor数量
            num_anchor_ = 2 * len(self.aspect_rations[idx])  # 4 or 6
            centers_x = np.repeat(centers_x, num_anchor_).reshape(-1, 1)  # (h*w*num_anchor_, 1)
            centers_y = np.repeat(centers_y, num_anchor_).reshape(-1, 1)
            boxes_width = np.tile(boxes_width, (feature_h * feature_w, 1)).reshape(-1, 1)
            boxes_height = np.tile(boxes_height, (feature_h * feature_w, 1)).reshape(-1, 1)
            anchor_boxes = np.concatenate([centers_x, centers_y, boxes_width, boxes_height], axis=-1)
            print(f"anchors: {anchor_boxes.shape}")

            # 将先验框变成小数模式
            # 归一化
            anchor_boxes[:, ::2] /= self.img_width
            anchor_boxes[:, 1::2] /= self.img_height
            anchor_boxes = np.clip(anchor_boxes, a_min=0, a_max=1.0)

            # print(anchor_boxes.shape)
            output_boxes.append(anchor_boxes)
        output_boxes = np.concatenate(output_boxes, axis=0)
        output_boxes = torch.tensor(output_boxes, dtype=torch.float32)
        return output_boxes


if __name__ == "__main__":
    # 输入的图片大小为(300, 300)
    input_shape = (300, 300)
    anchor = PriorAnchor(input_shape)
    total_anchor_boxes = anchor()
    print(total_anchor_boxes[5776:5784])
