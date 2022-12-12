from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from data.datasets import build_dataset

from data import transforms
from structures.container import Container


class BatchCollator(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        """

        Args:
            batch: Shape: (batch_size, 3) 3 means [image, container(boxes, labels), image_idx]

        Returns:

        """
        images, targets, image_ids = list(zip(*batch))
        images = default_collate(images)
        image_ids = default_collate(image_ids)

        if self.is_train:
            boxes_list = []
            labels_list = []
            for elem in targets:
                boxes_list.append(elem['boxes'])
                labels_list.append(elem['labels'])
            targets = Container(boxes=default_collate(boxes_list), labels=default_collate(labels_list))
        else:
            targets = None
        return images, targets, image_ids


def make_data_loader(cfg, is_train=True):
    # 获得transform/target_transform
    transform = transforms.build_transform(cfg, is_train)
    target_transform = transforms.build_target_transform(cfg) if is_train else None

    # 获得数据集
    dataset_list = cfg.DATASETS.TRAIN
    datasets = build_dataset(dataset_list, transform, target_transform, is_train)

    dataloaders = []
    is_shuffle = is_train
    for dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=is_shuffle,
                                collate_fn=BatchCollator(is_train))
        dataloaders.append(dataloader)

    if is_train:
        # 训练时仅使用单个数据集，测试时测需要分别评估不同测试集上的数据，因此这里如果是训练，需要检查dataloader的数量
        assert len(dataloaders) == 1
        return dataloaders[0]
    return dataloaders


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
