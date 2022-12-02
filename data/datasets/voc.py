from PIL import Image
import torch
import torch.utils.data
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET  # 一种灵活的容器对象，用于在内存中存储结构化数据


class VOCDataset(torch.utils.data.Dataset):
    """输入数据集名字，输出处理好的数据"""
    class_names = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                   'tvmonitor']

    def __init__(self, data_dir, split, transform=None, target_transform=None):
        """Handle the VOC annotation
        Args:
            root_path(str): root path of this project
            data_dir(str): file path to VOC2007 or VOC2012
            image_dataset(str): eg. 'train', 'test', 'trainval'
        """
        super(VOCDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split

        self.transform = transform
        self.target_transform = target_transform

        # -------------------------------------------------------------------------------------------------- #
        # 获得数据集图片名列表
        # -------------------------------------------------------------------------------------------------- #
        self.image_ids = self._get_image_ids()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        img: size of (height, width, 3), 3-> BGR
        gt(boxes): size of (num_box, 5), 5-> [xmin, ymin, xmax, ymax, label]
        """
        image_id = self.image_ids[idx]
        # 读取annotation，得到每张图片的坐标位置，类别，以及检测难易程度
        boxes, labels, is_difficult = self._get_annotation(image_id)

        # 读取图片
        image = self._get_image(image_id)

        # image,annotation的transform
        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform is not None:
            image, boxes, labels = self.target_transform(image, boxes, labels)

        return image, boxes, labels

    def _get_image_ids(self):
        """获取图片训练/测试集图片id的list"""
        image_sets_file_name = os.path.join(self.data_dir, "ImageSets", "Main", f"{self.split}.txt")
        img_ids = []
        with open(image_sets_file_name) as f:
            lines = f.readlines()
            for line in lines:
                img_ids.append(line.strip())
        return img_ids

    def _get_annotation(self, image_id):
        """根据图片id找到对应的annotation file读取并返回目标框坐标位置、分类及检测难易程度"""
        annotation_file = os.path.join(self.data_dir, "Annotations", f"{image_id}.xml")
        objects = ET.parse(annotation_file).findall("object")

        # 初始化boxes,labels, is_difficult，用于存储一张图片中的目标框物体的位置坐标、类别以及检测难易程度
        boxes, labels, is_difficult = [], [], []

        for obj in objects:
            # 获得目标分类
            cls_name = obj.find('name').text.lower().strip()
            cls_id = self.class_names.index(cls_name)
            labels.append(cls_id)
            # 获得目标框坐标
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) - 1
            ymin = float(bbox.find('ymin').text) - 1
            xmax = float(bbox.find('xmax').text) - 1
            ymax = float(bbox.find('ymax').text) - 1
            boxes.append([xmin, ymin, xmax, ymax])
            # 获取目标检测难易程度
            difficult = int(obj.find('difficult').text)
            is_difficult.append(difficult)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64), np.array(is_difficult,
                                                                                           dtype=np.uint8)

    def _get_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", f"{image_id}.jpg")
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def pull_item(self, idx):
        img_id = self.img_list[idx]
        target = ET.parse(self.annotation_path % img_id).getroot()
        img = cv2.imread(self.img_path % img_id)  # (height, Width, 'BGR')
        assert img is not None, 'image is not found.'
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, self.classes_names, width,
                                           height)  # [[xmin, ymin, xmax, ymax, label], ...]

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])  # 对图片进行数据增强同时改变其对应的box坐标和label值
            # 图片由opencv读取，现在通道顺序是：BGR，现在需要转换成RGB
            # boxes: (num_box, 4), labels: (num_box) --> labels: (num_box, 1) --> cat(boxes, labels): (num_box, 5)
            target = np.hstack([boxes, np.expand_dims(labels, axis=1)])
            target = torch.from_numpy(target)
        return img, target, height, width  # 输出图片仍然是BGR

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


def dataset_collate(batch):
    """自定义collate fn用于处理解决批量图片的annotations的stack问题，由于每张图片对应的annotation(bounding boxes
    数量不同，默认的collate fn会报错，因此此处自定义collate
    Args:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Returns:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for img, box in batch:
        imgs.append(img)
        targets.append(box)
    imgs = torch.stack(imgs, dim=0)
    return imgs, targets
