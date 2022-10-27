import os

import cv2
import numpy as np
from torch.utils import data
import xml.etree.ElementTree as ET  # 一种灵活的容器对象，用于在内存中存储结构化数据

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

VOC_ROOT = "/home2/xianghao/data/VOCdevkit/"


class VOCTransformation():
    def __init__(self):
        super(VOCTransformation, self).__init__()


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, output is annotation
    Args:
        root(str): file path to VOCdevkit folder
        image_sets(str): imageset to use(eg. 'train', 'val', 'test')
        transform(callable, optional): transformation to perform on the input image
        target_transform(callable, optional): transformation to perform on the target 'annotation'(eg: take in caption string, return tensor of word indices)
        dataset_name(str, optional): which dataset to load (default: 'VOC2007')
    """

    def __init__(self, root, image_sets=None, transform=None,
                 target_transform=None, dataset_name="VOC0712"):
        if image_sets is None:
            image_sets = [("2007", "trainval"), ("2012", "trainval")]
        self.root = root
        self.image_sets = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        self._annopath = os.path.join("%s", "Annotations", "%s.xml")  # used to get annotation
        self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")  # used to get image

        self.ids = []
        for year, name in self.image_sets:
            root_path = os.path.join(self.root, f"VOC{year}")
            with open(os.path.join(root_path, "ImageSets", "Main", f"{name}.txt")) as f:
                lines = f.readlines()
                for line in lines:
                    self.ids.append((root_path, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, 1)))


if __name__ == "__main__":
    voc = VOCDetection("/home2/xianghao/data/VOCdevkit")
    voc.pull_item(0)
