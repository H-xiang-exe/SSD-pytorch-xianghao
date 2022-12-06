import torch
import torch.nn as nn

from .post_processor import PostProcessor


class SSDBoxHead(nn.Module):

    def __init__(self, cfg):
        super(SSDBoxHead, self).__init__()
        self.cfg = cfg
        self.predictor = make_box_predictor(cfg)
        self.post_processor = PostProcessor(cfg)

    def forward(self, features):
        """

        Args:
            features: features from backbone

        Returns:
            tuple: loc_preds, conf_preds, priors
            container(test): Container(boxes, scores, labels)
        """
        if self.training:
            return self._forward_train(features)
        else:
            return self._forward_test(features)

    def _forward_train(self, x):
        loc_preds, conf_preds = self._predict(x)
        return loc_preds, conf_preds, self.prior_anchors

    def _forward_test(self, x):
        loc_preds, conf_preds = self._predict(x)
        # ----------------------------------------------------------------------------------- #
        # 以下两部可在模型外部处理，也可在内部处理，这里暂时放在外部，此处仅作注释
        # ----------------------------------------------------------------------------------- #

        # 置信度
        scores = F.softmax(conf_preds, dim=2)
        # ----------------------------------------------------------------------------------- #
        # 解码locations
        # ----------------------------------------------------------------------------------- #
        self.prior_anchors = self.prior_anchors.to(torch.device('cuda'))
        bboxes = box_utils.decode(loc_preds, self.prior_anchors, self.cfg.MODEL.CENTER_VARIANCE,
                                  self.cfg.MODEL.SIZE_VARIANCE)
        # 转换为corner form
        bboxes = box_utils.center_form_to_corner_form(bboxes)

        # 后处理
        detections = (scores, bboxes)
        detections = self.post_processor(detections)  # container(boxes, scores, labels)
        return detections
