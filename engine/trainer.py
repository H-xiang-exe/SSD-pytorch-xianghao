import yaml
import os

from torch.utils.data import DataLoader

import data
from utils import preprocess


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

        # 准备数据集
        self.training_data, self.test_data = self._get_dataset()

        # 设置dataloader
        self.train_dataloader = DataLoader(self.training_data,
                                           batch_size=args.batch_size,
                                           # num_workers=args.num_workers,
                                           shuffle=True,
                                           collate_fn=data.voc.dataset_collate,
                                           # pin_memory=True
                                           )
        self.test_dataloader = DataLoader(self.test_data,
                                          # num_workers=args.num_workers,
                                          shuffle=True,
                                          collate_fn=data.voc.dataset_collate,
                                          # pin_memory=True
                                          )
        # 准备模型
        self.model = self._get_model()

    def _get_dataset(self):
        # 获得所有数据集的相关信息
        datasets_info_path = os.path.join(self.base_root, 'configs/datasets.yaml')
        with open(datasets_info_path, 'r', encoding='utf-8') as f:
            datasets_info = yaml.safe_load(f)
        # 根据命令行的args获得当前数据集的信息（包括数据根目录、类别数等）
        cur_dataset_info = datasets_info[self.args.dataset]

        # 获得数据根目录
        cur_dataset_name = cur_dataset_info['name']
        cur_dataset_root = cur_dataset_info['data_root']

        training_data, test_data = None, None
        if cur_dataset_name == 'VOC2007':
            training_data = data.voc.VOCDataset(cur_dataset_info, 'train', preprocess.TrainTransform())
            test_data = data.voc.VOCDataset(cur_dataset_info, 'val', preprocess.TrainTransform())

        assert training_data is not None and test_data is not None, 'load train/test data failed'
        return training_data, test_data

    def _get_model(self):
        pass

    def train(self):
        pass

    # def train_one_epoch(self):
    #     # ---------------------
    #     # 准备数据集
    #     training_data = voc0712.VOCDataset()
    #     test_data = voc0712.VOCDataset()
    #     # ---------------------
    #
    #     # ---------------------
    #     # Dataloader
    #     train_dataloader = DataLoader(training_data)
    #     test_dataloader = DataLoader(test_data)

    def test(self):
        pass

    def build_model(self):
        pass
        # ssd_net = build_ssd()
        # net = ssd_net

        # if args.cuda:
        #     net = torch.nn.DataParallel(net)
        #     cudnn.benchmark = True
        #
        # if args.cuda:
        #     net.cuda()

        # optimizer = optim.SGD(net.parameters(), lr=args.lr,
        #                       momentum=args.momentum, weight_decay=args.weight_decay)
        # citerion = MultiBoxLoss()

        # prepare dataloader

        # ---------------------------------
        # Start Model Training ...

        # ---------------------------------

        # net.train()
