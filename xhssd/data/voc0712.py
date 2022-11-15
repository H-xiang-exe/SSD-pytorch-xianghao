import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET  # 一种灵活的容器对象，用于在内存中存储结构化数据

from xhssd.utils import preprocess
class VOCAnnotationTransform(object):
    """用于对Annotation中的box坐标和分类进行归一化并返回[[xmin, ymin, xmax, ymax, cls_id], ...]"""

    def __call__(self, img_annotation, classes, width, height):
        """
        Args:
            img_annotation(ET element): the target annotation
            classes(list): class name of object
            width(int): width
            height(int): height
        """
        boxes = []
        for obj in img_annotation.iter('object'):
            difficult = 0
            if obj.find("difficult") != None:
                difficult = obj.find("difficult").text
            cls = obj.find("name").text
            # print(cls)
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            bbox = obj.find("bndbox")

            bndbox = []
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # 对框的坐标进行归一化: 坐标值/宽（高）
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            bndbox.append(cls_id)  # [xmin, ymin, xmax, ymax, cls_id]
            boxes.append(bndbox)
        return boxes


class VOCDataset(Dataset):
    """输入数据集名字，输出处理好的数据"""

    def __init__(self, root_path, data_root, image_dataset=None, transform=None, target_transform=VOCAnnotationTransform()):
        """Handle the VOC annotation
        Args:
            root_path(str): root path of this project
            data_root(str): file path to VOCdevkit
            image_dataset(tuple[str]): eg. ('2007', 'train'), ('2007', 'test'), ('2007', 'trainval')
        """
        super(VOCDataset, self).__init__()
        self.root_path = root_path
        self.classes, self.num_classes = self.get_class()
        assert image_dataset is not None, f'image dataset is None, needs parameter: image_dataset'
        self.transform = transform
        self.target_transform = target_transform

        self.annotation_path = os.path.join(
            data_root, f"VOC{image_dataset[0]}", "Annotations", "%s.xml")
        self.img_path = os.path.join(
            data_root, f"VOC{image_dataset[0]}", "JPEGImages", "%s.jpg")

        self.img_list = []
        with open(os.path.join(data_root, f"VOC{image_dataset[0]}", "ImageSets", "Main", f"{image_dataset[1]}.txt")) as f:
            lines = f.readlines()
            for line in lines:
                self.img_list.append(line.strip())

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        img: size of (height, width, 3), 3-> BGR
        gt(boxes): size of (num_box, 5), 5-> [xmin, ymin, xmax, ymax, label]
        """
        img, gt, h, w = self.pull_item(idx)

    def pull_item(self, idx):
        img_id = self.img_list[idx]
        target = ET.parse(self.annotation_path % img_id).getroot()
        img = cv2.imread(self.img_path % img_id)  # (height, Width, 'BGR')
        height, width, channels = img.shape
        # print(img_id)

        if self.target_transform is not None:
            target = self.target_transform(target, self.classes, width,
                                           height)  # [[xmin, ymin, xmax, ymax, label], ...]

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])  # 对图片进行数据增强同时改变其对应的box坐标和label值
            # 图片由opencv读取，现在通道顺序是：BGR，现在需要转换成RGB
            img = img[:, :, (2, 1, 0)]
            # boxes: (num_box, 4), labels: (num_box) --> labels: (num_box, 1) --> cat(boxes, labels): (num_box, 5)
            target = np.hstack([boxes, np.expand_dims(labels, axis=1)])
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width  # 输出图片仍然是BGR

    def pull_image(self, idx):
        """根据idx获得数据集中的原图(不经过transform)，形式为PIL
        这里不直接使用pull_item()的原因是不希望经过tranform
        Args:
             idx(int): index of image to show
        Return:
            PIL img
        """
        image_id = self.img_list[idx]
        return cv2.imread(self.img_path % image_id, cv2.IMREAD_COLOR)  # 这里以openCV读取，并非PIL

    def pull_anno(self, idx):
        """根据idx获得图片对应的annotation
        Args:
            idx(int): index of image
        Return:
            list: [img_id, [(label, bbox coords), ...]
            eg. ('001718', [('dog', (96, 13, 438, 332)), ...]
        """
        img_id = self.img_list[idx]
        anno = ET.parse(self.annotation_path % img_id).getroot()
        # 获得box坐标（不做归一化）
        gt = self.target_transform(anno, self.classes, 1, 1)  # ([xmin, ymin, xmax, ymax, cls], ...)
        return img_id, gt

    def get_class(self):
        voc_class = []
        classes_txt = os.path.join(self.root_path, 'xhssd/data/voc_class.txt')
        with open(classes_txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                voc_class.append(line.strip())
        return voc_class, len(voc_class)
