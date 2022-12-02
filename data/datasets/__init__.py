from torch.utils.data import ConcatDataset

from config.path_datasets import DatasetPath
from data.datasets.voc import VOCDataset
from data.datasets.coco import COCODataset

_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
}


def build_dataset(cfg, is_train=True):
    # ------------------------------------------------------------------------------------------ #
    # 获得数据集列表
    # ------------------------------------------------------------------------------------------ #
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    datasets = []
    # ------------------------------------------------------------------------------------------ #
    # 逐个处理单个数据集
    # ------------------------------------------------------------------------------------------ #
    for dataset_name in dataset_list:
        # 获得数据集信息
        data = DatasetPath.get_name(dataset_name)
        # 获得数据集所在目录及split
        args = data.args
        # 获得数据集对应的类，如 VOCDataset
        factory = _DATASETS[data['factory']]

        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]
