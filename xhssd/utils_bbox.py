import torch

class BBoxUtility(object):
    def __init__(self, num_classes):
        super(BBoxUtility, self).__init__()
        self.num_classes = num_classes

