from torch.utils import data


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ""
)

VOC_ROOT = "/home2/xianghao/data/VOCdevkit/"


class VOCTransformation():
    def __init__(self):
        super(VOCTransformation, self).__init__()


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, output is annotation

    Args:
    

    """

    def __init__(self, root, image_sets=None, transform=None, target_transform=None, dataset_name="VOC0712"):
        pass
