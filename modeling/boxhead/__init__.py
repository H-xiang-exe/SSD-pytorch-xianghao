from .ssdboxhead import SSDBoxHead

def build_box_head(cfg):
    return SSDBoxHead(cfg)