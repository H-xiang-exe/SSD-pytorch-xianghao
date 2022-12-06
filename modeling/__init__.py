from . import ssd_model


def build_model(cfg):
    return ssd300_vgg16.build_ssd(cfg)
