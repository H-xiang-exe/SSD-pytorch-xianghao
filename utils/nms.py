import torch

from torchvision.ops import nms

def batched_nms(boxes, scores, labels, iou_threshold):
    """_summary_

    Args:
        boxes (torch.Tensor): _description_ Shape: (N, 4)
        scores (_type_): _description_
        labels (_type_): _description_
        iou_threshold (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 超过阈值的盒子的数量为0，说明没有合格的盒子
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device = boxes.device)
    
    # 给不同类盒子一个偏移量，使得不同类完全不在同一个区域，这样它们不同类之间的iou保证为0，
    # 在把它们视作同一类用nms处理的时候不会互相干扰。（因为根据nms，同类框会互相计算iou从而
    # 根据大于某个阈值去除掉一些同类框，这里不互相干扰，那么处理任何一类就都不会取出另一类的
    # 盒子）
    max_coordinate = boxes.max()
    offsets = labels.to(boxes.device) * (max_coordinate + 1) # (N,)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep