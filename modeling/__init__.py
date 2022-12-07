from . import ssd_model


def build_model(cfg):
    return ssd_model.SSD(cfg)
