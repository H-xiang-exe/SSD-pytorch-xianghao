from .vgg import vgg


def build_backbone(cfg):
    return vgg(cfg)
