import os
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision
import torch.nn as nn

from xhssd import ssd300_vgg16
from xhssd import prior_anchor_box
from xhssd.utils import augmentations


def decode_boxes(mbox_loc, anchors, variances):
    """
    Args:
        mbox_loc: 真实框相对于先验框的transform(offset，这里用offset不太恰当)值 [transform_center_x, transform_center_y, transform_width, transform_height]
        anchors: 先验框
        variances(list[float32]): 一个trick操作，用于对预测值进行缩放的参数
    Return:
        decode_bbox: (num_boxes, 4), 4 means (xmin, ymin, xmax, ymax)
    """
    # 获得先验框的宽和高
    anchor_width = anchors[:, 2] - anchors[:, 0]  # [num_anchors]
    anchor_height = anchors[:, 3] - anchors[:, 1]  # [num_anchors]
    # 获得先验框的中心点
    anchor_center_x = 0.5 * (anchors[:, 0] + anchors[:, 2])  # [num_anchors]
    anchor_center_y = 0.5 * (anchors[:, 1] + anchors[:, 3])  # [num_anchors]

    # 根据预测值和先验框位置得出实际的框的位置和宽高(推导公式略。)
    decode_bbox_center_x = anchor_width * variances[0] * mbox_loc[:, 0] + anchor_center_x  # [num_anchors]
    decode_bbox_center_y = anchor_height * variances[0] * mbox_loc[:, 1] + anchor_center_y  # [num_anchors]
    decode_bbox_width = anchor_width * torch.exp(variances[1] * mbox_loc[:, 2])  # [num_anchors]
    decode_bbox_height = anchor_height * torch.exp(variances[1] * mbox_loc[:, 3])  # [num_anchors]

    # 根据获得的真实框中心坐标和宽高获得左上角和右下角
    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width  # [num_anchors]
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width  # [num_anchors]
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height  # [num_anchors]
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height  # [num_anchors]

    # 将每个框的左上右下坐标放在一个list
    decode_bbox = torch.cat((decode_bbox_xmin[:, None],  # None,等价于unsqueeze，用于扩展一个维度
                             decode_bbox_ymin[:, None],
                             decode_bbox_xmax[:, None],
                             decode_bbox_ymax[:, None]), dim=-1)
    # 将值限制在[0,1]范围内
    decode_bbox = torch.min(torch.max(decode_bbox, torch.zeros_like(decode_bbox)), torch.ones_like(decode_bbox))
    return decode_bbox


def decode_box(predictions, anchors, image_shape, input_shape, confidence):
    """对SSD模型输出结果进行解码
    Args:
        predictions: SSD Model output: (location, confidence, num_classes)
            locations -> (Batch_num, h1*w1*num_anchor1 + h2*w2*num_anchor2 + ... + h6*w6*num_anchor6=num_anchors, 4)
            confidence ->(Batch_num, h1*w1*num_anchor1 + h2*w2*num_anchor2 + ... + h6*w6*num_anchor6=num_anchors , 21)
        anchors: all prior boxes locations. (all_boxes_nums, 4)
        confidence: 置信度阈值，超过该阈值的框才会被采用
    """
    # 获得框的位置
    mbox_loc = predictions[0]  # (N, num_boxes)
    # 获得框内各种类的置信度
    mbox_conf = nn.Softmax(dim=-1)(predictions[1])

    results = []
    # 对每一张图片进行处理， 由于在predict.py中只输入一张图片，所以下面的for外层循环只进行一次
    for i in range(len(mbox_loc)):
        results.append([])
        # 利用回归结果对每一张图片的预测结果进行解码：通过预测的transform和先验框的位置还原成真正的预测box值
        decode_bbox = decode_boxes(mbox_loc[i], anchors, variances=[0.1, 0.2])  # (num_boxes, 4)
        for c in range(1, 21):
            # 取出属于该类的所有框的置信度
            # 判断是否大于门限
            c_confs = mbox_conf[i, :, c]  # [num_anchors]
            c_confs_m = c_confs > confidence
            c_confs_m = [not c for c in c_confs_m]  # 暂时写的，之后需要删除
            # if (len(c_confs[c_confs_m])) > 0:
            if (len(c_confs[c_confs_m])) > -1:
                # 取出得分高于confidence的框及其属于c类的概率（置信度）
                boxes2process = decode_bbox[c_confs_m]
                confs2process = c_confs[c_confs_m]
                # 利用NMS算法将该类中的框选出来
                # confs2process = confs2process.to(torch.float64)
                keep = torchvision.ops.nms(boxes2process, confs2process, 0.3)

                # 取出在非极大抑制中效果较好的内容
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                good_boxes = boxes2process[keep]  # (num_boxes_by_nms, 4)
                good_confs = confs2process[keep][:, None]  # (num_boxes_by_nms, 1) 此时num_boxes指通过nms处理过后的盒子
                labels = (c - 1) * torch.ones((len(keep), 1)).to(device)  # (num_boxes_by_nms, 1)
                # 将label，置信度，框的位置进行堆叠
                c_pred = torch.cat((good_boxes, good_confs, labels),
                                   dim=1).cpu().numpy()  # (num_boxes_by_nms, 4+1+1=6), 6 means (xmin, ymin, xmax, ymax, confidence, label)
                # 添加到当前图片的结果里
                results[-1].extend(c_pred)
        # 如果当前图片的结果存在符合要求的框
        if (len(results[-1]) > 0):
            results[-1] = np.array(results[-1])
            # 求每个盒子的中心点坐标和宽高
            box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4]) * 0.5, results[-1][:, 2:4] - results[-1][:,
                                                                                                      0:2]
            results[-1][:,:4] = ssd_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return results

def ssd_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    """
    Args:
        box_xy: 盒子的中心点坐标，size: (num_boxes, 2)
        box_wh: 盒子的宽和高 size: (num_boxes, 2)
        input_shape: 输入网络的图片的维度
    """
    pass

def detect_image(model, image):
    # ---------------------------------
    # 计算输入图片的宽高
    # ---------------------------------
    image_shape = np.array(np.shape(image)[0:2])

    # 将图像转换为RGB图像
    # image = image.convert('RGB')

    # resize图像
    transform = augmentations.BaseTransform()
    image_data, _, _ = transform(image)

    # 图片(h, w,c) -> (1, c, h, w)
    image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), axis=0)

    # 获得所有先验框
    input_shape = [300, 300]
    get_anchor_boxes = prior_anchor_box.PriorAnchor(input_shape)
    prior_anchors = get_anchor_boxes()

    with torch.no_grad():
        # 转换成torch的形式
        images = torch.from_numpy(image_data)
        # 将图像输入网络，进行预测
        # (location, confidence, num_classes) locations: (N, h1*w1*num_anchor1 + h2*w2*num_anchor2 + h3*w3*num_anchor3, 4)
        outputs = model(images)
        # 对预测结果进行解码
        results = decode_box(outputs, prior_anchors, image_shape, input_shape, confidence=0.5)
        # 如果没有检测到物体，则返回原图
        if (len(results) <= 0):
            return image


if __name__ == '__main__':
    model = ssd300_vgg16.build_ssd('test')
    data_root = "./batchdata/VOCdevkit"

    dir_origin_path = os.path.join(data_root, "VOC2007/JPEGImages")
    while True:
        # img = input('Input image filename:')
        img = '000005.jpg'
        img = os.path.join(dir_origin_path, img)
        try:
            image = cv2.imread(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = detect_image(model, image)
            # r_image.show()
            break
