import numpy as np
import matplotlib.pyplot as plt


class PriorAnchor(object):
    def __init__(self, input_shape, min_size, max_size=None, aspect_ratios=None, flip=True):
        super(PriorAnchor, self).__init__()
        # 获得输入图片的大小
        self.input_shape = input_shape
        # 先验框的短边
        self.min_size = min_size
        # 先验框的长边
        self.max_size = max_size

        # 根据aspect_ratios确定对于某个确定的feature map的每个点对应的anchor box的纵横比
        self.aspect_ratios = []
        for ar in aspect_ratios:
            self.aspect_ratios.append(ar)
            self.aspect_ratios.append(1.0 / ar)

    def call(self, layer_shape, mask=None):
        """
        获得对应于某个layer的anchor box的集合
        """
        # 获得特征层的宽高
        layer_height = layer_shape[0]
        layer_width = layer_shape[1]

        # 原图的宽和高
        img_height = self.input_shape[0]
        img_width = self.input_shape[1]

        # 获得对于每个格子而言，其对应的每个anchor box的宽度和高度
        box_heights = []
        box_widths = []
        for ar in self.aspect_ratios:
            # 首先添加小正方形
            if ar == 1 and len(box_widths) == 0:
                box_heights.append(self.min_size)
                box_widths.append(self.min_size)
            # 其次添加一个大正方形
            elif ar == 1 and len(box_widths) == 1:
                box_heights.append(np.sqrt(self.min_size * self.max_size))
                box_widths.append(np.sqrt(self.min_size * self.max_size))
            else:
                box_heights.append(self.min_size * np.sqrt(ar))  # 这里为什么要开方
                box_widths.append(self.min_size / np.sqrt(ar))

        # 根据特征层的宽高和原图的宽高确定当前特征层在原图对应的网格的中心点
        step_w = img_width / layer_width
        step_h = img_height / layer_height
        linw = np.linspace(0.5 * step_w, img_width - 0.5 * step_w, layer_width)
        linh = np.linspace(0.5 * step_h, img_height - 0.5 * step_h, layer_height)
        centers_w, centers_h = np.meshgrid(linw, linh)
        centers_w = centers_w.reshape(-1, 1)
        centers_h = centers_h.reshape(-1, 1)

        # if layer_height == 3:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     plt.axis('equal')
        #     plt.ylim(-50, 350)
        #     plt.xlim(-50, 350)
        #     plt.scatter(centers_w, centers_h)

        # 根据网格中心点和每个纵横比获得每个盒子的坐标位置（左上角，右下角）
        num_anchor_ = len(self.aspect_ratios)
        anchor_boxes = np.concatenate((centers_w, centers_h), axis=1)  # (center_nums, 2)
        anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchor_))
        # 获得左上角、右下角
        anchor_boxes[:, ::4] -= 0.5 * np.array(box_widths)  # 左上角横坐标
        anchor_boxes[:, 1::4] -= 0.5 * np.array(box_heights)  # 左上角纵坐标
        anchor_boxes[:, 2::4] += 0.5 * np.array(box_widths)  # 右下角横坐标
        anchor_boxes[:, 3::4] += 0.5 * np.array(box_heights)  # 右下角纵坐标

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
        anchor_boxes[:, ::2] /= img_width
        anchor_boxes[:, 1::2] /= img_height
        anchor_boxes = anchor_boxes.reshape(-1, 4)

        anchor_boxes = np.maximum(np.maximum(anchor_boxes, 0.0), 1.0)
        return anchor_boxes


if __name__ == "__main__":
    # 输入的图片大小为(300, 300)
    input_shape = [300, 300]

    # 先验框大小
    anchors_size = [30, 60, 111, 162, 213, 264, 315]

    # 特征层网格size
    feature_heights = [38, 19, 10, 5, 3, 1]
    feature_widths = [38, 19, 10, 5, 3, 1]

    # 不同level先验框的纵横比
    # 每个数字同时表示高宽比和宽高比，所以[1,2]代表着实际的宽高比为[1,1,2, 1/2]，四个宽高比代表四个先验框
    aspect_rations = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]

    anchors = []
    for idx in range(len(feature_heights)):
        anchors.append(PriorAnchor(input_shape, anchors_size[idx], anchors_size[idx + 1], aspect_rations[idx]).call(
            (feature_heights[idx], feature_widths[idx])))
    # print(anchors)
    anchors = np.concatenate(anchors, axis=0)
    print(np.shape(anchors))
