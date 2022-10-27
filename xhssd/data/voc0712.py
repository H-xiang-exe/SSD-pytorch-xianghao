import os

from torch.utils import data

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

VOC_ROOT = "/home2/xianghao/data/VOCdevkit/"


class VOCTransformation():
    def __init__(self):
        super(VOCTransformation, self).__init__()


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, output is annotation
    Args:
        root(str): file path to VOCdevkit folder
        image_sets(str): imageset to use(eg. 'train', 'val', 'test')
        transform(callable, optional): transformation to perform on the input image
        target_transform(callable, optional): transformation to perform on the target 'annotation'(eg: take in caption string, return tensor of word indices)
        dataset_name(str, optional): which dataset to load (default: 'VOC2007')
    """

    def __init__(self, root, image_sets=None, transform=None,
                 target_transform=None, dataset_name="VOC0712"):
        if image_sets is None:
            image_sets = [("2007", "trainval"), ("2012", "trainval")]
        self.root = root
        self.image_sets = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        self._annopath = os.path.join("%s", "Annotations", "%s.xml")  # used to get annotation
        self._imgpath = os.path.join("%s", "JPEGImages", ".jpg")  # used to get image

        self.ids = []
        for year, name in self.image_sets:
            root_path = os.path.join(self.root, f"VOC{year}")
            with open(os.path.join(root_path, "ImageSets", "Main", f"{name}.txt")) as f:
                lines = f.readlines()
                for line in lines:
                    self.ids.append((root_path, line.strip()))

if __name__ == "__main__":
    voc = VOCDetection("/home2/xianghao/data/VOCdevkit")
    print(voc.ids[0])