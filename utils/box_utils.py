import torch
import torchvision


def point_form(boxes):
    """Convert prior_boxes to (xmin, ymin, xmax, yamx) from (center_x, cengter_y, width, height)

    Args:
        boxes: (torch.tensor) boxes location, shape: (num_boxes, 4) 4 means(center_x, center_y, width, height)

    Returns:
        boxes: (torch.tensor) boxes location, shape: (num_boxes, 4) 4 means (xmin, ymin, xmax, ymax)
    """
    return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], dim=-1)


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """将每个prior box和与其具有最高IOU的gt box进行匹配，对bounding boxes进行编码，然后返回与prior boxes匹配的
    gt boxes的索引以及相应的置信度和位置

    Args:
        threshold:
        truths: ground truth boxes, (N, 4)
        priors: prior boxes in corner-point offset (num_priors, 4)
        variances:
        labels: 对于单张图片的所有目标物体的标签 shape: (num_objects)
        loc_t:
        conf_t:
        idx:

    Returns:

    """
    # 获得truth和priors的iou
    overlaps = torchvision.ops.box_iou(truths, point_form(priors))  # (N,M)

    # ------------ 先验框和目标框互相匹配 ---------------
    # 每个目标框优先匹配与其IOU最大的先验框
    best_prior_overlap, best_prior_idx = torch.max(overlaps, dim=1)  # (num_objs)

    # 每个先验框匹配与其IOU最大的目标框
    best_truth_overlap, best_truth_idx = torch.max(overlaps, dim=0)  # (num_priors)

    torch.index_fill(best_truth_overlap, 0, best_prior_idx, 2)
    # TODO refactor: index best_prior with long tensor
    # ensure every gt mathes with its prior of max overlap
    for j in range(best_prior_idx.size()[0]):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]  # 获得所有prior box的目标标签框  shape: (num_priors, 4)
    # 获得所有prior box的目标框的类别(num_priors)
    conf = labels[best_truth_idx] + 1  # 背景类标号为0，其他分类为1-20

    # 对于num_priors个盒子，它们和对应的truth box的iou如果小于某个阈值，我们将其视为负样本，令其分类标签为0
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances) # 和prior boxes相匹配的gt boxes经过variance编码的target boxes
    loc_t[idx] = loc
    conf_t[idx] = conf


def encode(matched, priors, variances):
    """将prior boxes的variance编码到与prior boxes相匹配（基于Iou）的gt boxes。

    Args:
        matched: (tensor) coordinates of gt boxes matched with each prior box. Shape:[num_priors, 4]
        priors: (tensor) prior boxes in corner-point form. Shape: [num_priors, 4]
        variances: (list[float]) Variances of prior boxes

    Returns:
        encoded boxes: (tensor), 和prior boxes相匹配的经过了variance编码之后的gt boxes.
            Shape: [num_boxes, 4]
    """
    # (left or top + right or bottom)/2 - prior_center_x or prior_center_y
    # = gt_center_x or gt_center_y - prior_center_x or prior_center_y
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])  # (num_priors, 2) 2 means encoded cx,cy
    g_wh = (matched[:, 2:] - matched[:, :2]) / 2 / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]  # (num_priors, 2) 2 means encoded w,h
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], dim=1)  # (num_priors, 4)
