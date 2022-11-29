import logging

def setup_logger(name, distributed_rank, save_dir=None):
    # 创建logger记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    