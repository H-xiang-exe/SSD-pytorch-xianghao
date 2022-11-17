import sys
sys.path.append('../../')
print(sys.path)
from xhssd.data import voc0712
from xhssd.utils import preprocess

voc = voc0712.VOCDataset("D:\\Works\\SSD-pytorch-xianghao", "D:\\Works\\SSD-pytorch-xianghao\\batchdata\\VOCdevkit", ('2007', 'train'), transform=preprocess.TrainTransform())
# voc.pull_image(1)
classes = voc.get_class()
print(classes)
