import torch.nn as nn
from modeling.backbone import build_backbone
from modeling.boxhead import build_box_head

class SSD(nn.Module):
    def __init__(self, cfg):
        super(SSD, self).__init__(cfg)
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)

    def forward(self, images):
        features = self.backbone(images)
        detections = self.box_head(features) # 注意：训练时会输出网络最后一层的值，测试时会经过解码、NMS等后处理

        return detections
