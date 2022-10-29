from email.mime import image
from logging import root
from torch.utils.data import Dataset
import os

import cv2
import numpy as np
from torch.utils import data
import xml.etree.ElementTree as ET  # 一种灵活的容器对象，用于在内存中存储结构化数据
from torchvision.datasets import FashionMNIST

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)


class VOCDataset(Dataset):
    """输入数据集名字，输出处理好的数据"""
    def __init__(self, root, transform=None, target_transform=None):
        """Handle the VOC annotation
        Args:
            root(str): file path to VOCdevkit
        """
        super(VOCDataset, self).__init__()
        image_datasets = ('2007', 'train')
        dataset_path = os.path.join(root, image_datasets[0])

        self.annotation_path = os.path.join(dataset_path, "Annotations", "%s.xml")
        self.img_path = os.path.join(dataset_path, "JPEGImages", "%s.jpg")

        dataset_name = image_datasets[1]
        self.img_list = []
        with open(os.path.join(dataset_path, "ImageSets", "Main", f"{dataset_name}.txt")) as f:
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




if __name__ == "__main__":
    pass
