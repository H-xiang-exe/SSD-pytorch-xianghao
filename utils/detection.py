import cv2
import colorsys
import torch
import torch.nn as nn
import numpy as np
import torchvision

from data.transforms import transforms


class Detection(object):
    """用于对模型的输出进行解码分析，筛选出预测框
    Args:
        dataset(class object): dataset class
        confidence(float): 置信度阈值，超过该阈值的框才会被采用
        letterbox_image(bool): 用于控制是否对输入图像进行不失真的resize，在多次测试后，发现关闭letterbox_image直接resize更好
    """

    def __init__(self, input_shape, device, dataset, confidence=0.1, letterbox_image=False):
        super(Detection, self).__init__()
        self.input_shape = input_shape
        self.device = device
        self.confidence = confidence
        self.letterbox_image = letterbox_image

        self.dataset = dataset
        self.class_names, self.num_classes = self.dataset.get_class()
        self.num_classes = self.num_classes + 1

        # 为不同类别的物体画框设置不同颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def detect_image(self, model, image):
        # ---------------------------------
        # 计算输入图片的宽高
        # ---------------------------------
        image_shape = np.array(np.shape(image)[0:2])  # (h,w)

        # resize图像
        transform = preprocess.TestTransform()
        image_data, _, _ = transform(image)

        # 图片(h, w,c) -> (1, c, h, w)
        image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), axis=0)

        with torch.no_grad():
            # 转换成torch的形式
            images = torch.from_numpy(image_data)
            # 将图像输入网络，进行预测
            # (location, confidence, num_classes) locations: (N, h1*w1*num_anchor1 + h2*w2*num_anchor2 + h3*w3*num_anchor3, 4)
            outputs = model(images)
            # 对预测结果进行解码 return: (num_image, corresponding_num_boxes, 6)
            # num_images表示图片数量， corresponding_num_boxes表示经过nms之后保留下来的bbox数量，6表示(ymin, xmin, ymax, xmax, confidence, label)
            results = self.decode_box(outputs, image_shape)
            # 如果没有检测到物体，则返回原图
            if len(results) <= 0:
                return image
            # 获得第一张图片（事实上也只有一张图片）所有盒子的标签
            top_label = np.array(results[0][:, 4], dtype='int32')
            top_confidence = results[0][:, 5]
            top_boxes = results[0][:, :4]

        # 设置字体与边框厚度
        from PIL import Image, ImageFont, ImageDraw
        font = ImageFont.truetype(font='simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        # 图像绘制
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # cv转成PIL
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_confidence[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image_shape[0], np.floor(bottom).astype('int32'))
            right = min(image_shape[1], np.floor(right).astype('int32'))

            label = f'{predicted_class} {score:.2f}'
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)  # 返回用指定字体对象显示给定字符串所需要的图像尺寸
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:  # 确定图像上端是否能放下标签图像
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for j in range(thickness):
                draw.rectangle((left + j, top + j, right - j, bottom - j), outline=self.colors[c])

            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
            if i == 3: break

        return image

    def decode_box(self, predictions, image_shape):
        """对SSD模型输出结果进行解码
        Args:
            image_shape: 待检测的原图的shape
            predictions: SSD Model output: (location, confidence, num_classes)
                locations -> (Batch_num, h1*w1*num_anchor1 + h2*w2*num_anchor2 + ... + h6*w6*num_anchor6=num_anchors, 4)
                confidence ->(Batch_num, h1*w1*num_anchor1 + h2*w2*num_anchor2 + ... + h6*w6*num_anchor6=num_anchors , 21)
        Return:
            results(list): (num_image, corresponding_num_boxes, 6) num_images表示图片数量， corresponding_num_boxes表示经过nms之后保留下来的bbox数量，6表示(ymin, xmin, ymax, xmax, confidence, label)
        """
        # 获得框的位置
        mbox_loc = predictions[0]  # (N=1, num_boxes)
        # 获得框内各种类的置信度
        mbox_conf = nn.Softmax(dim=-1)(predictions[1]) # (N=1, num_boxes, 21)

        results = []
        # 对每一张图片进行处理， 由于在predict.py中只输入一张图片，所以下面的for外层循环只进行一次
        for i in range(len(mbox_loc)):
            results.append([])
            # 利用回归结果对每一张图片的预测结果进行解码：通过预测的transform和先验框的位置还原成真正的预测box值
            decode_bbox = self.decode_boxes(mbox_loc[i])  # (num_boxes, 4)
            for c in range(1, 21):
                # 取出属于该类的所有框的置信度
                # 判断是否大于门限
                c_confs = mbox_conf[i, :, c]  # [num_anchors]
                c_confs_m = c_confs > self.confidence
                # c_confs_m = [not x for x in c_confs_m]
                if (len(c_confs[c_confs_m])) > 0:
                    # 取出得分高于confidence的框及其属于c类的概率（置信度）
                    boxes2process = decode_bbox[c_confs_m]
                    confs2process = c_confs[c_confs_m]
                    # 利用NMS算法将该类中的框选出来
                    # confs2process = confs2process.to(torch.float64)
                    keep = torchvision.ops.nms(boxes2process, confs2process, 0.3)

                    # 取出在非极大抑制中效果较好的内容
                    good_boxes = boxes2process[keep]  # (num_boxes_by_nms, 4)
                    good_confs = confs2process[keep][:, None]  # (num_boxes_by_nms, 1) 此时num_boxes指通过nms处理过后的盒子
                    labels = (c - 1) * torch.ones((len(keep), 1))  # (num_boxes_by_nms, 1)
                    # 将label，置信度，框的位置进行堆叠
                    c_pred = torch.cat((good_boxes, good_confs, labels),
                                       dim=1).cpu().numpy()  # (num_boxes_by_nms, 4+1+1=6), 6 means (xmin, ymin, xmax, ymax, confidence, label)
                    # 添加到当前图片的结果里
                    results[-1].extend(c_pred)
            # 如果当前图片的结果存在符合要求的框
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                # 求每个盒子的中心点坐标和宽高
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4]) * 0.5, results[-1][:, 2:4] - results[-1][:,
                                                                                                          :2]
                results[-1][:, :4] = self.ssd_correct_boxes(box_xy, box_wh, image_shape)
        return results

    def decode_boxes(self, mbox_loc, variances=None):
        """
        Args:
            mbox_loc: 真实框相对于先验框的transform(offset，这里用offset不太恰当)值 [transform_center_x, transform_center_y, transform_width, transform_height]
            variances(list[float32]): 一个trick操作，用于对预测值进行缩放的参数。使用一个bool参数来控制两种模式，True时variance其被包含在预测之中，False则需要手动显示设置超参数进行解码：
        Return:
            decode_bbox: (num_boxes, 4), 4 means (xmin, ymin, xmax, ymax)
        """
        # 获得所有先验框 anchors: all prior boxes locations. (all_boxes_nums, 4)
        if variances is None:
            variances = [0.1, 0.2]
        prior_anchors = prior_anchor_box.PriorAnchor(self.input_shape)()
        # 获得先验框的宽和高
        anchor_width = prior_anchors[:, 2] - prior_anchors[:, 0]  # [num_anchors]
        anchor_height = prior_anchors[:, 3] - prior_anchors[:, 1]  # [num_anchors]
        # 获得先验框的中心点
        anchor_center_x = 0.5 * (prior_anchors[:, 0] + prior_anchors[:, 2])  # [num_anchors]
        anchor_center_y = 0.5 * (prior_anchors[:, 1] + prior_anchors[:, 3])  # [num_anchors]

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

    def ssd_correct_boxes(self, box_xy, box_wh, image_shape):
        """将在预处理之后的图片上求出的bbox坐标重新反向运算得到在原图上的坐标。注意这里所有的xywh都是[0,1]的小数形式，实际像素长度需要用图片宽高乘以该百分比值
        Args:
            box_xy(np.ndarray): 盒子的中心点坐标，size: (num_boxes, 2)
            box_wh(np.ndarray): 盒子的宽和高 size: (num_boxes, 2)
            input_shape: 输入网络的图片的维度
            letterbox_image(bool): 该变量用于控制是否对图像进行了不失真的resize；这里还原box坐标时，如果之前进行了不失真的resize那就要考虑这个问题
        Return:
            boxes(np.ndarray): (num_boxes, 4), 4 means (ymin, xmin, ymax, xmax)
        """
        # 表明原图和模型输入图片的尺寸
        origin_h, origin_w = image_shape
        input_h, input_w = self.input_shape

        # 将y轴放前面，和图像的高宽的顺序相匹配
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., :: -1]

        if self.letterbox_image:
            # 求解原图在缩放到input_shape范围后实际有效区域的高宽
            scale = min(input_h / origin_h, input_w / origin_w)
            new_h, new_w = int(origin_h * scale), int(origin_w * scale)

            # 求解填充的灰边的宽度和高度（靠近坐标原点一侧，百分比小数形式）
            offset_h, offset_w = (input_h - new_h) / 2.0 / input_h, (input_w - new_w) / 2 / input_w

            # 求解填充黑边之前的缩放院原图上box坐标点的坐标
            box_yx = box_yx - offset_h
            # 求解缩放前即原图上的box坐标中心点坐标以及box的高宽（小数）
            box_yx *= scale
            box_hw *= scale
        # 对于直接缩放或是缩放后填充黑边的图像，求解出box的左上角右下角坐标（相对于原图的百分比）
        box_mins = np.array(box_yx - (box_hw / 2.0))
        box_maxs = np.array(box_yx + (box_hw / 2.0))
        # 变换格式：shape:(num_boxes, 4), 4-> (ymin, xmin, ymax, xmax)
        boxes = np.concatenate([box_mins[:, 0:1], box_mins[:, 1:2], box_maxs[:, 0:1], box_maxs[:, 1:2]], axis=1)
        # 根据比例获得boxes的实际坐标 (num, 4) * [4]
        boxes *= np.array([input_h, input_w, input_h, input_w])
        return boxes
