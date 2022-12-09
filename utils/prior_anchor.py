from itertools import product
from math import sqrt

import torch

from config import cfg


class PriorAnchor(object):
    def __init__(self, cfg):
        """大小为input_shape的图片，对应于6个feature level(每个level对应着把原图划分为不同数量的网格)，

        Args:
            input_shape(tuple): 输入的图片的大小
            min_size(int): 规定了当前尺度(scale)下的最小框尺寸
            max_size(int): 规定了当前尺度(scale)下的大框尺寸
            aspect_ratios(list): 规定了6个feature level上每一个位置对应的多个框的纵横比。eg.[[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
        """
        super(PriorAnchor, self).__init__()
        # 获得输入图片的大小
        self.img_size = cfg.INPUT.IMAGE_SIZE
        # 先验框的短边
        self.min_sizes = cfg.MODEL.PRIORS.MIN_SIZES
        # 先验框的长边
        self.max_sizes = cfg.MODEL.PRIORS.MAX_SIZES

        # 特征层网格size
        self.feature_sizes = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.aspect_rations = cfg.MODEL.PRIORS.ASPECT_RATIOS

    def __call__(self):
        """
        获得对应于某个layer的anchor box的集合
        """
        priors = []

        for f_idx, f_size in enumerate(self.feature_sizes):
            for i, j in product(range(f_size), repeat=2):
                # 网格中心点坐标
                cx = (i + 0.5) / f_size
                cy = (j + 0.5) / f_size

                # aspect ratio = 1 prior-small
                w = h = self.min_sizes[f_idx] / self.img_size
                priors.append([cx, cy, w, h])
                # aspect ratio = 1 prior-large
                w = h = self.max_sizes[f_idx] / self.img_size
                priors.append([cx, cy, w, h])

                # other aspect ratio prior-small
                base_w = base_h = self.min_sizes[f_idx] / self.img_size
                for ar in self.aspect_rations[f_idx]:
                    ratio = sqrt(ar)
                    priors.append([cx, cy, base_w / ratio, base_h * ratio])
                    priors.append([cx, cy, base_w * ratio, base_h / ratio])

        priors = torch.tensor(priors, dtype=torch.float32)
        priors = torch.clamp(priors, min=0.0, max=1.0)
        return priors


if __name__ == "__main__":
    # 输入的图片大小为(300, 300)
    input_shape = (300, 300)
    total_anchor_boxes = PriorAnchor(cfg)()
    print(total_anchor_boxes[5776:5784])
