import torch
from torchvision.ops import nms

from utils.nms import batched_nms
from structures.container import Container

class PostProcessor(object):
    def __init__(self, cfg):
        """对模型的box和score进行后处理:
        1. convert locations(center form) to boxes(corner form)
        2. NMS

        Args:
            cfg (_type_): _description_
            
        Returns:
            container(structures.container.Container): boxes, labels, scores
        """
        super(PostProcessor, self).__init__()
        self.cfg = cfg
        self.width = cfg.INPUT.IMAGE_SIZE
        self.height = cfg.INPUT.IMAGE_SIZE

    def __call__(self, detections):
        # 模型输出
        batch_scores, batch_boxes = detections
        device = batch_scores.device

        results = []
        for scores, boxes in zip(batch_scores, batch_boxes):
            num_boxes = boxes.shape[0]   # boxes: (num_boxes, 4)
            num_classes = scores.shape[1]  # scores: (num_boxes, cls)

            # (num_boxes, 4) -> (num_boxes, 21, 4) 复制了21份
            boxes = boxes.view(num_boxes, 1, 4).expand(
                num_boxes, num_classes, 4)
            # [0, 1, 2, ..., 21]
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, num_classes).expand_as(scores)  # (num_boxes, cls)

            # ------------------------------------------------------------------ #
            # 移除为背景类的数据
            # ------------------------------------------------------------------ #            
            # 去掉分类为0的那一份，每个盒子的位置还有20份
            boxes = boxes[:, 1:]  # (num_boxes, 20, 4)
            scores = scores[:, 1:] # (num_boxes, 20)
            labels = labels[:, 1:] # (num_boxes, 20)
            
            # 让每个box每一类都成为一条单独的数据
            boxes = boxes.reshape(-1, 4)# (num_boxes*20, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            # print("-----------------------------------")
            # print(boxes.shape, scores.shape)
            # print("-----------------------------------")
            # 获得置信度大于某个阈值的数据实例的索引
            indices = torch.nonzero(scores > self.cfg.TEST.CONFIDENCE_THRESHOLD).squeeze()
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]
            
            boxes[:, ::2] *= self.width
            boxes[:, 1::2] *= self.height
            
            # 通过nms过滤
            keep = batched_nms(boxes, scores, labels, self.cfg.TEST.NMS_THRESHOLD)
            boxes = boxes[keep], scores[keep], labels[keep]
            
            container = Container(boxes=boxes, labels=labels, scores=scores)
            container.img_width = self.width
            container.img_height = self.height
            results.append(container)
        return results
            
            
