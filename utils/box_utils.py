import math

import torch
import torchvision
import torch.nn.functional as F


def corner_form_to_center_form(boxes):
    """Convert [xmin, ymin, xmax, ymax] to [center_x, center_y, w, h]
    Args:
        boxes(torch.Tensor)
    """
    wh = boxes[..., 2:] - boxes[..., :2]
    center = (boxes[..., :2] + boxes[..., 2:]) / 2
    return torch.cat([center, wh], dim=-1)


def center_form_to_corner_form(locations):
    """Convert [center_x, center_y, w, h] to [xmin, ymin, xmax, ymax]"""
    xymin = locations[..., :2] - locations[..., 2:] / 2
    xymax = locations[..., :2] + locations[..., 2:] / 2
    return torch.cat([xymin, xymax], dim=-1)


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
    """将prior boxes的variance编码到与prior boxes相匹配（基于Iou）的gt boxes。这里target boxes的数量是一张图片的boxes数量，decode时
    的boxes是一个批次图片，所以shape会不一样.

    Args:
        center_form_target_boxes(torch.Tensor): coordinates of target boxes matched with each prior box.
                                                Shape: [num_gt_boxes, 4].
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


def decode(locations, center_form_priors, center_variance, size_variance):
    """
    这里target boxes的数量是一个batch图片的boxes数量，encode时的boxes是一张图片，所以shape会不一样.
    Args:
        locations: Shape: (N, num_priors, 4)
        center_form_priors: (num_priors, 4) 4 means [cx, cy, w, h]
        center_variance:
        size_variance:

    Returns:
        decoded boxes(torch.Tensor): (N, num_priors, 4)
    """
    if center_form_priors.dim() + 1 == locations.dim():
        center_form_priors = center_form_priors.unsqueeze(0)  # (1, num_priors, 4)
    boxes_xy = center_form_priors[..., 2:] * (center_variance * locations[..., :2]) + center_form_priors[..., :2]
    boxes_wh = center_form_priors[..., 2:] * torch.exp(size_variance * locations[..., 2:])
    boxes = torch.cat([boxes_xy, boxes_wh], dim=-1)
    return boxes


@torch.no_grad()
def hard_negative_mining(pred_confidences, gt_labels, negpos_ratio):
    """

    Args:
        pred_confidences: Shape: (batch_size, num_priors, num_classes)
        gt_locations:
        gt_labels: Shape: (batch_size, num_priors)
        negpos_ratio:

    Returns:

    """
    # ------------------------------------------------------------------------------- #
    # 将所有负例(对应的target_label为0，即背景类)按照loss排序
    # ------------------------------------------------------------------------------- #
    # 计算所有样本在target_label为0(背景类)上的loss
    negative_loss = -F.log_softmax(pred_confidences, dim=-1)[:, :, 0]  # (batch_size, num_priors)
    # 根据target_label找出正例样本的mask -> 其余的即是负例样本的mask
    positive_mask = gt_labels > 0  # (batch_size, num_priors)
    # 为了方便后面对负例排序，这里将正样本在cls=0上的loss直接置为负无穷
    negative_loss[positive_mask] = -math.inf
    # 对负例loss（负例对应的类别是背景类，因此loss即是在target label=0这一维的loss）进行从大到小的排序
    # 根据大小顺序索引得出background loss本身每个loss的排名. eg. 初始loss index: [0, 1, 2, 3, 4, 5], 假设有如下大小顺序:
    # [3, 0, 1, 2, 4, 5], 在6个loss中idx=3对应的loss最大，id=5的对应loss最小，
    _, negative_idx = negative_loss.sort(dim=1, descending=True)  # (batch_size, num_priors)
    # 再进行下一步，对大小顺序本身而不是对idx对应的loss进行排序，得到原来每个loss在整个顺序内的排名.
    # eg.[3, 0, 1, 2, 4, 5] -> [0, 1, 2, 3, 4, 5], 对应的索引是[1, 2, 3, 0, 4, 5] 即表示在原loss中第0个loss在大小顺序中排名第1名，
    # 在原loss中第3个位置的loss在大小顺序中排名第0名
    _, negative_rank = negative_idx.sort(1)  # (batch_size, num_priors)

    # ------------------------------------------------------------------------------- #
    # 按照negpos_ratio计算应该选取用于训练的负例数量
    # ------------------------------------------------------------------------------- #
    # 正例的数量
    num_positive = positive_mask.long().sum(dim=1, keepdims=True)  # (batch_size, 1)
    # 负例的数量
    num_priors = gt_labels.shape[1]
    num_negative = torch.min(num_positive * negpos_ratio, num_priors - num_positive)  # (batch_size, 1)

    # ------------------------------------------------------------------------------- #
    # 在negative loss中选取loss较大的num_negative用于计算损失函数
    # ------------------------------------------------------------------------------- #
    # 每张图片中选取num_negative个负例，选取的负例的位置用如下mask表示
    real_negative_mask = negative_rank < num_negative

    # 返回所有loss中正例和相应倍数的负例对应的mask
    return positive_mask | real_negative_mask
