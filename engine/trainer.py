import torch
import yaml
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.backends import cudnn

import data
from utils import preprocess
from modeling.ssd300_vgg16 import build_ssd
from solver.multibox_loss import MultiBoxLoss


class Trainer(object):
    def __init__(self, args, base_root):
        """

        Args:
            args:
            base_root: root path of project
        """
        super(Trainer, self).__init__()
        self.args = args
        self.base_root = base_root

        # 获得checkpoint存储路径
        self.save_ckp_dir = os.path.join(self.base_root, 'checkpoints')

        # 获得设备(cpu, cuda)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 准备数据集
        self.dataset_info = self._get_dataset_info()  # 当前所选数据集的相关配置信息
        self.training_data, self.test_data = self._get_dataset()

        # 设置dataloader
        self.train_dataloader = DataLoader(self.training_data,
                                           batch_size=args.batch_size,
                                           # num_workers=args.num_workers,
                                           shuffle=True,
                                           collate_fn=data.voc.dataset_collate,
                                           drop_last=True,
                                           # num_workers=1
                                           # pin_memory=True
                                           )
        self.test_dataloader = DataLoader(self.test_data,
                                          # num_workers=args.num_workers,
                                          shuffle=True,
                                          collate_fn=data.voc.dataset_collate,
                                          drop_last=True
                                          # pin_memory=True
                                          )
        # 准备模型
        self.model = self._get_model()

        # 设置损失函数
        self.criterion = MultiBoxLoss(self.dataset_info['num_classes'], 0.5, self.dataset_info['variance'], 3, self.device)

        # 设置优化器
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # 设置学习率
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 80], gamma=0.1)

        # 加载模型权重

    def _get_dataset_info(self):
        # 获得所有数据集的相关信息
        datasets_info_path = os.path.join(self.base_root, 'configs/datasets.yaml')
        with open(datasets_info_path, 'r', encoding='utf-8') as f:
            datasets_info = yaml.safe_load(f)
        # 根据命令行的args获得当前数据集的信息（包括数据根目录、类别数等）
        cur_dataset_info = datasets_info[self.args.dataset]
        return cur_dataset_info

    def _get_dataset(self):
        # 获得数据根目录
        cur_dataset_name = self.dataset_info['name']

        training_data, test_data = None, None
        if cur_dataset_name == 'VOC2007':
            training_data = data.voc.VOCDataset(self.dataset_info, 'train', preprocess.TrainTransform())
            test_data = data.voc.VOCDataset(self.dataset_info, 'val', preprocess.TrainTransform())

        assert training_data is not None and test_data is not None, 'load train/test data failed'
        return training_data, test_data

    def _get_model(self):
        model = build_ssd(self.args.phase, self.dataset_info, size=self.dataset_info['min_dim'],
                          num_classes=self.dataset_info['num_classes'])
        return model

    def train(self):
        self.model.to(self.device)
        if self.device == 'cuda':
                # net = torch.nn.DataParallel(net)  # make parallel
                cudnn.benchmark = True
        for epoch in range(self.args.epoch):
            self.model.train()
            for batch_idx, (images, targets) in tqdm(enumerate(self.train_dataloader)):
                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets]

                outputs = self.model(images)

                loss_dict = self.criterion(outputs, targets)
                loss = sum(loss_dict)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if batch_idx % 2 == 0:
                    loss = loss.item()/len(images)
                    print(f'Epoch: {epoch}, Batch_idx:{batch_idx}, loss: {loss:>7f}')

    def test(self):
        pass
