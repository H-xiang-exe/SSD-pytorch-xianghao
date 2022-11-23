import sys

sys.path.append('../../')
import os
import yaml
import data
from data import voc
from utils import preprocess
from torch.utils.data import DataLoader

BASE_DIR = 'D:\Works\SSD-pytorch-xianghao'
datasets_info_path = os.path.join(BASE_DIR, 'configs/datasets.yaml')
with open(datasets_info_path, 'r', encoding='utf-8') as f:
    dataset_info = yaml.safe_load(f)['VOC2007']

training_data = voc.VOCDataset(dataset_info, 'train', preprocess.TrainTransform())
# voc.pull_image(1)
# classes = voc.get_class()
# print(classes)
print('Dataloader Preparing')
dataloader = DataLoader(training_data, batch_size=4, shuffle=True,
                        collate_fn=data.voc.dataset_collate,
                        )
print('Dataloader Preparing Finished')
for epoch in range(20):
    print(f'Epoch:{epoch}')
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f'batch_idx:{batch_idx}')
