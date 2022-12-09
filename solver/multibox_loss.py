import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import box_utils


def log_sum_exp(x):
    """求解log(sum_{i}^C exp(xi))，即crossentropy的log部分"""
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), dim=1, keepdim=True)) + x_max


class MultiBoxLoss(nn.Module):
    """SSD Weighted loss function

    Compute Targets:
        1) Produce Confidence Target Indices by matching ground truth boxes with (default) 'priorboxes'
        that have jaccard index > threshold parameter (default threshold: 0.5)

    Args:
        num_classes: (int) 目标类别数
        overlap_thresh: iou的阈值，高于此阈值的框才会被视作正样本

    """

    def __init__(self, neg_pos):
        super(MultiBoxLoss, self).__init__()
        self.negpos_ratio = neg_pos

    def forward(self, predictions, targets):
        """

        Args:
            predictions: (tuple) A tuple containing location preds, confidence preds and prior boxes from SSD net
                location shape: (batchsize, num_priors, 4)
                confidence shape: (batchsize, num_priors, 21)
                priors shape: prior boxes locations in corner-point offset (num_priors, 4)
            targets: ground truth boxes and labels for a batch. shape: (batch_size, num_objs, 5)

        Returns:

        """
        pred_locations, pred_confidences, priors = predictions  # ()
        num_classes = pred_confidences.shape[2]
        # 获得标签
        target_locations, target_labels = targets['boxes'], targets['labels']  # (16, 8732, 4), (16, 8732)
        # 获得样本中的所有正例的mask
        positive_mask = target_labels > 0  # (16, 8732)

        # ------------------------------------------------------------------------------- #
        # Hard negative mining
        # ------------------------------------------------------------------------------- #
        # 从正例之外再选取一部分负例共同组成训练样本，返回相应的mask
        mask = box_utils.hard_negative_mining(pred_confidences, target_labels, self.negpos_ratio)  # (16, 8732)

        # 计算所有正例的回归loss
        pred_locations_positive = pred_locations[positive_mask, :].view(-1, 4)
        target_locations_positive = target_locations[positive_mask, :].view(-1, 4)
        reg_loss = F.smooth_l1_loss(pred_locations_positive, target_locations_positive, reduction='sum')

        # 计算正例+部分负例的分类loss
        pred_confidences_posneg = pred_confidences[mask].view(-1, num_classes)
        targets_labels_posneg = target_labels[mask]
        cls_loss = F.cross_entropy(pred_confidences_posneg, targets_labels_posneg, reduction='sum')

        # Sum of losses: L(x,c,l,g)= ( L_conf(x,c) + alpha * L_loc(x,l,g) )/N
        num_positive = positive_mask.sum()
        
        loc_loss = reg_loss/ num_positive
        con_loss = cls_loss / num_positive
        
        import math
        if math.isnan(loc_loss) or math.isnan(con_loss):
            print(f"num_positive:{num_positive}")
            print(f"reg_loss:{reg_loss}")
            print(f"cls_loss:{cls_loss}")
            exit(0)

        return loc_loss, con_loss
