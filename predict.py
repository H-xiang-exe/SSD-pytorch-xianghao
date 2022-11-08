import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn

from xhssd import ssd300_vgg16
from xhssd import prior_anchor_box
from xhssd.utils import augmentations


def detect_image(model, image):
    # ---------------------------------
    # 计算输入图片的宽高
    # ---------------------------------
    image_shape = np.array(np.shape(image)[0:2])

    # 将图像转换为RGB图像
    image = cv2.cvtColor(image)

    # resize图像
    transform = augmentations.BaseTransform()
    image_data = transform(image)

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
        results = decode_box(outputs, prior_anchors, image_shape, input_shape)
        # 如果没有检测到物体，则返回原图
        if (len(results) <= 0):
            return image


def decode_box(predictions, anchors, image_shape, input_shape, confidence):
    """对SSD模型输出结果进行解码
    Args:
        predictions: SSD Model output: (location, confidence, num_classes)
            locations -> (N, h1*w1*num_anchor1 + h2*w2*num_anchor2 + h3*w3*num_anchor3, 4)
            confidence ->(N, h1*w1*num_anchor1 + h2*w2*num_anchor2 + h3*w3*num_anchor3, 21)
        anchors: all prior boxes locations. (all_boxes_nums, 4)
        confidence: 置信度阈值，超过该阈值的框才会被采用
    """
    # 获得框的位置
    mbox_loc = predictions[0]
    # 获得框内各种类的置信度
    mbox_conf = nn.Softmax(-1)(predictions[1])

    results = []
    for i in range(len(mbox_loc)):
        # 利用回归结果对先验框进行解码
        decode_bbox = decode_boxes(mbox_loc[i], anchors)

        for c in range(1, 21):
            # 取出属于该类的所有框的置信度
            # 判断是否大于门限
            c_confs = mbox_conf[i,:,c]
            c_confs_m = c_confs > confidence
            if(len(c_confs[c_confs_m]))>0:
                # 取出得分高于confidence的框
                boxes2process = decode_bbox[c_confs_m]
                confs2process = decode_bbox[c_confs_m]

                # 取出在非极大抑制中效果较好的内容

                # 将label，置信度，框的位置进行堆叠

    return results

def decode_bboxes(mbox_loc, anchors):
    """
    Args:
        mbox_loc: 真实框位置
        anchors: 先验框
    """
    # 获得先验框的宽和高
    anchor_width = anchors[:, 2] - anchors[:, 0]
    anchor_height = anchors[:, 3] - anchors[:, 1]
    # 获得先验框的中心点
    anchor_center_x = 0.5*(anchors[:, 0] + anchors[:, 2])
    anchor_center_y = 0.5*(anchors[:, 1] + anchors[:, 3])

    # 真实框距离先验框中心的xy轴偏移情况
    decode_bbox_center_x = mbox_loc[:, 0] * anchor_width * variances[0] # 这里的variance是用来做什么的？



if __name__ == '__main__':
    model = ssd300_vgg16.build_ssd('test')
    data_root = "./batchdata/VOCdevkit"

    dir_origin_path = os.path.join(data_root, "VOC2007/JPEGImages")

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = detect_image()
            r_image.show()
