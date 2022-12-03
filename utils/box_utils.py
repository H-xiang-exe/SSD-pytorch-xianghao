import torch
import torchvision


def corner_form_to_center_form(boxes):
    """Convert [xmin, ymin, xmax, ymax] to [center_x, center_y, w, h]
    Args:
        boxes(torch.Tensor)
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2
    return torch.cat([center_x, center_y, w, h], dim=-1)


def center_form_to_corner_form(locations):
    """Convert [center_x, center_y, w, h] to [xmin, ymin, xmax, ymax]"""
    xmin = locations[:, 0] - locations[:, 2] / 2
    xmax = locations[:, 0] + locations[:, 2] / 2
    ymin = locations[:, 1] - locations[:, 3] / 2
    ymax = locations[:, 1] + locations[:, 3] / 2
    return torch.cat([xmin, ymin, xmax, ymax], dim=-1)


def assign_priors(gt_boxes, gt_labels, corner_form_priors, iou_threshold):
    """

    Args:
        gt_boxes(torch.Tensor): corner form boxes. Shape:(num_objs, 4)
        gt_labels(torch.Tensor): (num_objs)
        corner_form_priors(torch.Tensor): Shape: (num_priors, 4)
        iou_threshold:

    Returns:
        target_boxes(torch.Tensor): corner form boxes.
        target_labels(torch.Tensor)
    """
    # 计算所有priors和gt_boxes的iou
    ious = torchvision.ops.box_iou(corner_form_priors, gt_boxes)  # (num_priors, num_objs)

    # 计算对于每个prior而言，与所有gt boxes的iou中最大的iou及其对应的gt box 索引
    best_target_iou_per_prior, best_target_index_per_prior = torch.max(ious, dim=1)
    # 计算对于每个gt box而言，与所有priors 的iou中最大的iou及其对应的prior box 索引
    best_prior_iou_per_gt, best_prior_index_per_gt = torch.max(ious, dim=0)

    # 将每个gt对应的best prior的best target变成这个gt
    for gt_index, prior_index in enumerate(best_prior_index_per_gt):
        best_target_index_per_prior[prior_index] = gt_index
    # 将gt对应的prior的iou全变成2
    best_target_iou_per_prior.index_fill_(0, best_prior_index_per_gt, 2)

    target_labels = gt_labels[best_target_index_per_prior]
    target_labels[best_target_iou_per_prior < iou_threshold] = 0  # 背景类
    target_boxes = gt_boxes[best_target_index_per_prior]

    return target_boxes, target_labels


def encode(center_form_target_boxes, center_form_priors, center_variance, size_variance):
    """将prior boxes的variance编码到与prior boxes相匹配（基于Iou）的gt boxes。

    Args:
        center_form_target_boxes(torch.Tensor): coordinates of target boxes matched with each prior box. Shape:[num_priors, 4]
        center_form_priors(torch.Tensor): prior boxes. Shape: [num_priors, 4]
        center_variance(float): variance for cx, cy
        size_variance(float): variance for w, h.

    Returns:
        target_locations(torch.Tensor): 和prior boxes相匹配的经过了variance编码之后的gt boxes. Shape: [num_priors, 4]
    """
    # encode variance
    locations_centers = (center_form_target_boxes[:, :2] - center_form_priors[:, :2]) / (
            center_form_priors[:, 2:] * center_variance)  # (num_priors,2)
    locations_wh = torch.log(center_form_target_boxes[:, 2:] / center_form_priors[:, 2:]) / size_variance

    return torch.cat([locations_centers, locations_wh], dim=-1)
