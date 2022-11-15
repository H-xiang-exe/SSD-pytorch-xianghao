import os
import sys

import torch

sys.path.append('../../')

from xhssd.data import voc0712
from xhssd.utils import preprocess
from xhssd.utils import detection
from xhssd import ssd300_vgg16

root = "D:\\Works\\SSD-pytorch-xianghao"
data_root = "D:\\Works\\SSD-pytorch-xianghao\\batchdata\\VOCdevkit"

voc = voc0712.VOCDataset(root, data_root, ('2007', 'train'), transform=preprocess.TrainTransform())
detector = detection.Detection(input_shape=[300, 300], device=torch.device('cpu'), dataset=voc)
print(detector.class_names, detector.num_classes)
img = '000005.jpg'
img = os.path.join(data_root, 'VOC2007', 'JPEGImages', img)

import cv2

image = cv2.imread(img)
model = ssd300_vgg16.build_ssd('test')

detector = detection.Detection(input_shape=[300, 300], device=torch.device('cpu'), dataset=voc)
r_image = detector.detect_image(model, image)
r_image.show()
