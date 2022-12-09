import numpy as np
import torch

from data.datasets import build_dataset
from data.transforms import build_transform, build_target_transform
from config import cfg
cfg.merge_from_file("../configs/voc07.yaml")

# ------------------------------------------------------------------------------------------ #
# 获得数据集列表
# ------------------------------------------------------------------------------------------ #
dataset_list = ("voc_2007_train",)
transform = build_transform()
target_transform = build_target_transform(cfg)
datasets = build_dataset(dataset_list, transform=transform, target_transform=target_transform)
image, target, idx = datasets[0].__getitem__(2)
print(target)
# for i in range(len(datasets[0])):
#     image, target, idx = datasets[0].__getitem__(i)
#     print(f"image:\n{image}")
#     print(f"target:\n{target}")
#     print("=====================================================================")

# import cv2
# print(image.shape)
# image = image.cpu().numpy()
# image = np.transpose(image, (1, 2, 0))
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# print(image.shape)
# cv2.imshow('image.jpg', image)
# cv2.waitKey(0)

# from data.build import make_data_loader
# train_dataloader = make_data_loader(cfg)
# data_iter = iter(train_dataloader)
# images, targets, image_ids = next(data_iter)
# # print(targets)
# locations, labels = targets['boxes'], target['labels']
# print(locations.shape)