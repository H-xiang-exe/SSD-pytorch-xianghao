from . import ssd300_vgg16


def build_model(cfg):
    return ssd300_vgg16.build_ssd('train')
