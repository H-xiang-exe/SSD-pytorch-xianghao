import sys

sys.path.append('../')
import os
import yaml
import data
from data.datasets import voc
from utils import preprocess
from torch.utils.data import DataLoader

BASE_DIR = '/root/autodl-tmp/SSD-pytorch-xianghao/'
datasets_info_path = os.path.join(BASE_DIR, 'configs/datasets.yaml')
with open(datasets_info_path, 'r', encoding='utf-8') as f:
    dataset_info = yaml.safe_load(f)['VOC2007']

training_data = voc.VOCDataset(dataset_info, 'train', preprocess.TrainTransform())
# voc.pull_image(1)
# classes = voc.get_class()
# print(classes)
print('Dataloader Preparing')
dataloader = DataLoader(training_data, batch_size=32, shuffle=True,
                        collate_fn=data.datasets.voc.dataset_collate,
                        )
print('Dataloader Preparing Finished')
print(f'trainset length: {len(dataloader)}')
