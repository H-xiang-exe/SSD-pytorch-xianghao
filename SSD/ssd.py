import torch.nn as nn


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the added multibox conv layers.Each multibox layer branchs into:
    """
    def __init__(self, phase, size, base, extras, head, num_classes):
        """
        Args:
            phase(string): "test" or "train"
            size(int): input image size
            base: VGG16 layers for input, size of either 300 or 500
            extras: extra layers that feed to multibox loc and conf layers
            head: "multibox head" consists of loc and conf conv layers
        """
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg =


def multibox():


def build_ssd(phase, size=300, num_classes=21):
    '''
    Args:
        phase: 'train' or 'test'
        size: size of input image
        num_classes: catagory of dataset
    '''
    assert phase == 'train' or phase == 'test', f"ERROR: Phase {phase} not recognized"
    assert size == 300, f"You specified size {size}. However currently only SSD300 is supported!"
    base_, extra_, head_ = multibox()

    return SSD(phase, size, base_, head_, num_classes)
