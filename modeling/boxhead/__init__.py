from .ssd_box_head import SSDBoxHead

def build_box_head(cfg):
    return SSDBoxHead(cfg)