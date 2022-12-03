from data.datasets import build_dataset
from data.transforms import build_transform, build_target_transform
# ------------------------------------------------------------------------------------------ #
# 获得数据集列表
# ------------------------------------------------------------------------------------------ #
dataset_list = ("voc_2007_train",)
transform = build_transform()
# target_transform = build_target_transform()
datasets =build_dataset(dataset_list, transform=transform, target_transform=None)
datasets[0].__getitem__(0)

