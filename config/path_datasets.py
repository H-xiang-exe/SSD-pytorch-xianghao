"""解析数据名获得数据集所在路径"""
import os


class DatasetPath(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "voc_2007_train": {
            "data_dir": "D:\Works\SSD-pytorch-xianghao/batchdata\VOCdevkit\VOC2007",
            "split": "train"
        },
        "voc_2007_test": {
            "data_dir": "D:\Works\SSD-pytorch-xianghao/batchdata\VOCdevkit\VOC2007",
            "split": "test"
        }
    }

    @staticmethod
    def get_name(name):
        if 'voc' in name:
            voc_root = DatasetPath.DATA_DIR
            if 'VOC_ROOT' in os.environ['VOC_ROOT']:
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
