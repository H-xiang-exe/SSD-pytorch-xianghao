import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import sqrt


class PriorAnchor(object):
    def __init__(self, input_shape):
        """大小为input_shape的图片，对应于6个feature level(每个level对应着把原图划分为不同数量的网格)，

        Args:
            input_shape(list): 输入的图片的大小
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

        boxes_height = []
        boxes_width = []

        # 求解每个level的feature map对应在原图的先验框的坐标集合
        for idx, (feature_h, feature_w, min_size, max_size) in enumerate(
                zip(self.feature_heights, self.feature_widths, self.min_sizes, self.max_sizes)):
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
                    boxes_height.append(self.min_sizes[idx] / ar)
                    boxes_width.append(self.min_sizes[idx] * ar)
                    # aspect ratio = ar, height = min_size*ar, width = min_size/ar
                    boxes_height.append(self.min_sizes[idx] * ar)
                    boxes_width.append(self.min_sizes[idx] / ar)

            # feature的尺寸为h*w，对应的原图被划分为h*w个网格
            # 求解用于detection的feature在原图对应的网格的边长
            step_w = self.img_width / feature_w
            step_h = self.img_height / feature_h
            half_step_w = step_w / 2
            half_step_h = step_h / 2

            # 获得网格中心点的横坐标和纵坐标
            linw = np.linspace(0.5 * step_w, self.img_width - 0.5 * step_w, feature_w)
            linh = np.linspace(0.5 * step_h, self.img_height - 0.5 * step_h, feature_h)
            centers_w, centers_h = np.meshgrid(linw, linh)
            centers_w = centers_w.reshape(-1, 1)  # (h*w, 1)
            centers_h = centers_h.reshape(-1, 1)  # (h*w, 1)

            # if layer_height == 3:
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111)
            #     plt.axis('equal')
            #     plt.ylim(-50, 350)
            #     plt.xlim(-50, 350)
            #     plt.scatter(centers_w, centers_h)

            # 根据网格中心点和每个纵横比获得每个盒子的坐标位置（左上角，右下角）
            num_anchor_ = 2 * len(self.aspect_rations[idx])  # 4 or 6
            anchor_boxes = np.concatenate((centers_w, centers_h), axis=1)  # (center_nums, 2), center_nums= h*w
            # 一个中心点有num_anchor_个先验包围盒，每个包围盒用2个坐标点才能描述，因此对于每个网格，要描述它对应的所有box，需要2*num_anchor个点，因此anchor_boxes要在原来每个点的基础上扩展2*num_anchor_倍，然后分别重新赋值
            anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchor_))
            # 对于每个网格，坐标分别为(box1_left, box1_top, box1_right, box1_down, box2_left, box2_top, box2_right, box2_down, ...)
            # 求解每个网格的所有box的left, top, right, down坐标值
            anchor_boxes[:, ::4] -= half_step_w  # 左上角横坐标
            anchor_boxes[:, 1::4] -= half_step_h  # 左上角纵坐标
            anchor_boxes[:, 2::4] += half_step_w  # 右下角横坐标
            anchor_boxes[:, 3::4] += half_step_h  # 右下角纵坐标

            # if layer_height == 3:
            #     rect1 = plt.Rectangle((anchor_boxes[4, 0], anchor_boxes[4, 1]), box_widths[0], box_heights[0], color='r',
            #                           fill=False)
            #     rect2 = plt.Rectangle((anchor_boxes[4, 4], anchor_boxes[4, 5]), box_widths[1], box_heights[1], color='r',
            #                           fill=False)
            #     rect3 = plt.Rectangle((anchor_boxes[4, 8], anchor_boxes[4, 9]), box_widths[2], box_heights[2], color='r',
            #                           fill=False)
            #     rect4 = plt.Rectangle((anchor_boxes[4, 12], anchor_boxes[4, 13]), box_widths[3], box_heights[3], color='r',
            #                           fill=False)
            #
            #     ax.add_patch(rect1)
            #     ax.add_patch(rect2)
            #     ax.add_patch(rect3)
            #     ax.add_patch(rect4)
            #
            #     plt.show()

            # 将先验框变成小数模式
            # 归一化
            anchor_boxes[:, ::2] /= self.img_width
            anchor_boxes[:, 1::2] /= self.img_height
            anchor_boxes = anchor_boxes.reshape(-1, 4)

            anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)
            output_boxes.append(anchor_boxes)
        return output_boxes


if __name__ == "__main__":
    # 输入的图片大小为(300, 300)
    input_shape = [300, 300]
    anchor = PriorAnchor(input_shape)
    total_anchor_boxes = anchor()
    print(total_anchor_boxes[0][1])
