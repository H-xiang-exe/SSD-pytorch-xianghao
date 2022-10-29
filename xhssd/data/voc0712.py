from email.mime import image
from logging import root
from torch.utils.data import Dataset
import os

import cv2
import numpy as np
import xml.etree.ElementTree as ET  # 一种灵活的容器对象，用于在内存中存储结构化数据

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)


class VOCDataset(Dataset):
    """输入数据集名字，输出处理好的数据"""

    def __init__(self, root, image_dataset=None, transform=None, target_transform=None):
        """Handle the VOC annotation
        Args:
            root(str): file path to VOCdevkit
        """
        super(VOCDataset, self).__init__()

        image_dataset = ('2007', 'train')
        self.transform = transform
        self.target_transform = target_transform

        self.annotation_path = os.path.join(
            root, f"VOC{image_dataset[0]}", "Annotations", "%s.xml")
        self.img_path = os.path.join(
            root, f"VOC{image_dataset[0]}", "JPEGImages", "%s.jpg")

        self.img_list = []
        with open(os.path.join(root, f"VOC{image_dataset[0]}", "ImageSets", "Main", f"{image_dataset[1]}.txt")) as f:
            lines = f.readlines()
            for line in lines:
                self.img_list.append(line.strip())

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        pass

    def pull_item(self, idx):
        img_id = self.img_list[idx]
        img_annotation = ET.parse(self.annotation_path % img_id).getroot()
        img = cv2.imread(self.img_path % img_id)
        height, width, channels = img.shape
        # print(img_id)

        if self.transform is not None:
            pass
        
        if self.target_transform is not None:
            pass
        
        boxes = []
        for obj in img_annotation.iter('object'):
            difficult = 0
            if obj.find("difficult") != None:
                difficult = obj.find("difficult").text
            cls = obj.find("name").text
            # print(cls)
            if cls not in VOC_CLASSES or int(difficult) == 1:
                continue
            cls_id = VOC_CLASSES.index(cls)
            bbox = obj.find("bndbox")

            bndbox = []
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # 对框的坐标进行归一化: 坐标值/宽（高）
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            boxes.append(bndbox)


if __name__ == "__main__":
    voc = VOCDataset("/root/autodl-tmp/data/VOCdevkit")
    voc.pull_item(1)
