import logging

def setup_logger(name, distributed_rank, save_dir=None):
    # 创建logger记录器
    logger = logging.getLogger()
    # 设置日志记录级别
    logger.setLevel(logging.DEBUG)
    
