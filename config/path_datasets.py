"""解析数据名获得数据集所在路径"""
import os


class DatasetPath(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "voc_2007_train": {
            "data_dir": "/root/autodl-tmp/data/VOCdevkit/VOC2007",
            "split": "train"
        },
        "voc_2007_trainval": {
            "data_dir": "/root/autodl-tmp/data/VOCdevkit/VOC2007",
            "split": "trainval"
        },
        "voc_2007_test": {
            "data_dir": "/root/autodl-tmp/data/VOCdevkit/VOC2007",
            "split": "test"
        },
        "voc_2012_trainval": {
            "data_dir": "/root/autodl-tmp/data/VOCdevkit/VOC2012",
            "split": "trainval"
        },
    }

    @staticmethod
    def get_name(name):
        if 'voc' in name:
            voc_root = DatasetPath.DATA_DIR
            if 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']
            attrs = DatasetPath.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs['data_dir']),
                split=attrs['split']
            )
            return dict(factory='VOCDataset', args=args)

        elif 'coco' in name:
            pass
        raise RuntimeError(f"Datasets not availabel: {name}")
