import numpy as np
import torch

from utils import box_utils


class SSDTargetTransform(object):

    def __init__(self, center_form_priors, center_variance, size_variance, iou_thread):
        """pass

        Args:
            center_form_priors(torch.Tensor):
            center_variance(float)
            size_variance(float):
            iou_thread(float)

        """
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_thread = iou_thread

    def __call__(self, gt_boxes, gt_labels):
        """
        1. 根据gt boxes和priors的iou将priors和gt boxes匹配，得到所有priors应该回归的目标框及其类别
        2. 对数据集中gt box坐标标签进行variance编码

        Args:
            gt_boxes(torch.Tensor): gt boxes in form of [xmin, ymin, xmax, ymax]. Shape: (num_objs, 4)
            gt_labels(torch.Tensor): labels of gt boxes in form of [class_id]. Shape: (num_objs)

        Returns:
            target_locations(torch.Tensor): target for per prior. Shape: [num_priors, 4]
            target_labels(torch.Tensor): target label for per prior. Shape: [num_priors]
        """
        # l_cx = (b_cx - d_cx)/d_w
        if isinstance(gt_boxes, np.ndarray):
            gt_boxes = torch.from_numpy(gt_boxes)
        if isinstance(gt_labels, np.ndarray):
            gt_labels = torch.from_numpy(gt_labels)
        # match priors with gt boxes 根据gt boxes和prior boxes的iou将两者匹配，选出所有priors需要匹配的gt boxes及其类别标签
        target_boxes, target_labels = box_utils.assign_priors(gt_boxes, gt_labels, self.corner_form_priors,
                                                              self.iou_thread)

        # convert to center form
        target_boxes = box_utils.corner_form_to_center_form(target_boxes)

        target_locations = box_utils.encode(target_boxes, self.center_form_priors, self.center_variance,
                                            self.size_variance)

        return target_locations, target_labels
