import logging
import os
import time
import datetime

import torch
from torch.utils.data import DataLoader
import yaml

from data.datasets import voc
from modeling.ssd300_vgg16 import build_ssd
from data.transforms import transforms
from utils.metric_logger import MetricLogger
from utils.checkpoint import CheckPointer
from engine import eval
from solver.multibox_loss import MultiBoxLoss
from data import transforms


def do_train(model, train_data_loader, optimizer, scheduler, checkpointer, device, args):
    """

    Args:
        model:
        train_data_loader:
        optimizer:
        scheduler:
        checkpointer:
        device:
        args:

    Returns:

    """
    pass


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
                                           collate_fn=voc.dataset_collate,
                                           drop_last=True,
                                           # num_workers=1
                                           # pin_memory=True
                                           )
        self.test_dataloader = DataLoader(self.test_data,
                                          # num_workers=args.num_workers,
                                          shuffle=True,
                                          collate_fn=voc.dataset_collate,
                                          drop_last=True
                                          # pin_memory=True
                                          )
        # 准备模型
        self.model = self._get_model()

        # 设置损失函数
        self.criterion = MultiBoxLoss(
            self.dataset_info['num_classes'], 0.5, self.dataset_info['variance'], 3, self.device)

        # 设置优化器
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # 设置学习率
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[40, 50, 60], gamma=0.1)

        # 加载模型权重

    def _get_dataset_info(self):
        # 获得所有数据集的相关信息
        datasets_info_path = os.path.join(
            self.base_root, 'configs/coco.yaml')
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
            training_data = voc.VOCDataset(
                self.dataset_info, 'train', transforms.build_transform(), transforms.build_target_transform())
            test_data = voc.VOCDataset(
                self.dataset_info, 'val', transforms.build_target_transform(), transforms.build_target_transform())

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
            torch.backends.cudnn.benchmark = True

        # 日志
        logger = logging.getLogger('SSD.trainer')
        logger.info("Start training ...")
        # metric记录
        meters = MetricLogger()
        start_training_time = time.time()
        end = time.time()

        # 保存
        checkpointer = CheckPointer(self.model, self.optimizer, self.scheduler, './checkpoints/', logger=logger)

        # dataloader数据的批次
        max_iter = len(self.train_dataloader)
        start_iter = 0
        self.model.train()
        for iteration, (images, targets) in enumerate(self.train_dataloader, start_iter):
            iteration = iteration + 1
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]

            outputs = self.model(images)

            loss_dict = self.criterion(outputs, targets)
            loss = sum(loss_dict)
            meters.update(loss=loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)

            # 输出训练日志
            if iteration % self.args.log_step == 0:
                # 剩余训练时间
                eta_seconds = meters.time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    meters.delimeter.join([
                        f"iter: {iteration:06d}",
                        f"lr: {self.optimizer.param_groups[0]['lr']:.5f}",
                        f"{meters}",
                        f"eta: {eta_string}",
                        f"mem:{torch.cuda.max_memory_allocated() / 1024.0 / 1024.0}M"
                    ])
                )
            # 保存当前模型
            if iteration % self.args.save_step == 0 and iteration != max_iter:
                checkpointer.save(f"model_{iteration:06d}")
            # 评估当前模型
            if iteration % self.args.eval_step == 0:
                eval_results = eval.do_evaluation()
                self.model.train()
        checkpointer.save(f"model_final")

        # 计算训练时间
        total_training_time = int(time.time() - start_training_time)
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(f"Total training time: {total_time_str}({total_training_time / max_iter}s/it)")