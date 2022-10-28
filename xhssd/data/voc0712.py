from email.mime import image
from logging import root
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

class VOCDataset(object):
    def __init__(self):
        """Handle the VOC annotation
        Args:
            root(str): file path to VOCdevkit
        """
        super().__init__(root)
        image_datasets = ('2007', 'train')
        dataset_path = os.path.join(root, image_datasets[0])

        self.annotation_path = os.path.join(dataset_path, "Annotations", "%s.xml")
        self.img_path = os.path.join(dataset_path, "JPEGImages", "%s.jpg")




if __name__ == "__main__":
    pass