import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_utils import match


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

    def __init__(self, num_classes, overlap_thresh, variance, neg_pos, device):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.variance = variance

        self.negpos_ratio = neg_pos

        self.device = device

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
        loc_data, conf_data, priors = predictions
        # 获得输入的数据条数
        batch_size = loc_data.size(0)
        # 获得所有先验框坐标（8732）
        priors = priors[:loc_data.size(1), :]
        # 获得先验框的数量
        num_priors = priors.size(0)

        # ----------------- match priors (default boxes) and groundth boxes ----------------------
        # 为每张图片每个prior box设定一个目标框
        loc_t = torch.Tensor(batch_size, num_priors, 4).to(self.device)
        conf_t = torch.LongTensor(batch_size, num_priors).to(self.device) # 每个prior的类别
        # 通过以下操作获得每张图片的目标框信息（包括位置和分类）
        for idx in range(batch_size):
            # 获得一张图片的若干个目标框的位置
            truths = targets[idx][:, :-1].data
            # 获得一张图片的若干个目标框的类别
            labels = targets[idx][:, -1].data
            # 获得所有先验框的坐标
            defaults = priors.data.to(self.device)
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)

        # 获得所有priors中目标框为非背景类的框的索引
        pos = conf_t > 0  # (batch_size, num_priors)

        # 获得预测结果中非背景类(正例)的框的坐标
        loc_p = loc_data[pos]
        # 获得上面这些框应该要回归的目标框的（正例）坐标encoded form
        loc_t = loc_t[pos]
        # 计算这一个batch的图片中目标框为非背景框的当前坐标预测结果和其对应的要回归的目标框之间的smooth_l1 loss
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Computer max conf across batch for hard negative mining
        # (batch_size, num_priors, 21) -> (batch_size * num_priors, num_classes)
        batch_conf = conf_data.view(-1, self.num_classes)
        # 计算所有框的默认分类损失(cross entropy)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))


        # Hard negative mining
        loss_c = loss_c.view(batch_size, -1)  # (batch_size, num_priors)
        # 因为给负样本排序，所以给正样本置0
        loss_c[pos] = 0  # filter out pos boxes for now
        # 对于每一张图片，都对其内部的所有box的loss进行从大到小的排序
        _, loss_idx = loss_c.sort(1, descending=True)  # 获得按大小顺序排列的索引 [3,2,4,1,0]
        # (batch_size, num_priors) # 获得索引对应的排名 [3,2,4,1,0]->[0,1,2,3,4]其对应的原索引即是[4,3,1,0,2]而这也正是它们的排名rank
        _, idx_rank = loss_idx.sort(1)

        # 每张图片中非背景框的数量
        num_pos = pos.long().sum(1, keepdim=True)  # （batch_size, 1]
        # print(f'num_pos: {num_pos}')
        # 根据negative: postive比值，获得每张图片中用于作为背景框（负例）的数量
        # num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)  # (batch, 1)
        num_neg = torch.min(self.negpos_ratio*num_pos, pos.size(1) - num_pos)
        # print(f'num_neg: {num_neg}')
        neg = idx_rank < num_neg.expand_as(idx_rank)  # 得到负样本索引（batch_size, num_priors)

        # Confidence loss including positive and negative examples
        pos_idx = pos.unsqueeze(dim=2).expand_as(conf_data)  # (batch_size, num_priors, num_classes)
        neg_idx = neg.unsqueeze(dim=2).expand_as(conf_data)  # (batch_size, num_priors, num_classes)

        # 根据pos_idx, neg_idx从output和target中挑选出要进行计算loss的数据
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        # 求解分类loss
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g)= ( L_conf(x,c) + alpha * L_loc(x,l,g) )/N

        N = num_pos.data.sum()  # batch_size
        res_loss_l = loss_l/N
        res_loss_c = loss_c/N
        # import math
        # if(math.isnan(res_loss_c)):
        #     print(loss_c)
        #     print(loss_l)
        #     exit(0)

        return res_loss_l, res_loss_c
