import logging
import os.path
import sys


def setup_logger(name, save_dir=None):
    # 创建logger记录器
    logger = logging.getLogger()
    # 设置日志记录级别
    logger.setLevel(logging.DEBUG)

    # 日志信息处理器，分发器
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] - [%(filename)s line:%(lineno)d] %(levelname)s => %(message)s", datefmt="%m-%d %H:%M")

    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    if save_dir is not None:
        log_path = os.path.join(save_dir, 'ssd.log')
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
