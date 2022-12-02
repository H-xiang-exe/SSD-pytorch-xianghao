import logging

import torch.backends.cudnn
from torch.utils.data import DataLoader

from config import parse, cfg
from modeling import build_model
from engine.trainer import Trainer
from utils.logger import setup_logger
from utils.checkpoint import CheckPointer


def train(cfg, args):
    # -------------------------------------------------------------------------------------- #
    # 建立logger
    # -------------------------------------------------------------------------------------- #
    logger = logging.getLogger("SSD.trainer")

    # -------------------------------------------------------------------------------------- #
    # 获得device
    # -------------------------------------------------------------------------------------- #
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # -------------------------------------------------------------------------------------- #
    # 建立模型
    # -------------------------------------------------------------------------------------- #
    model = build_model(cfg)
    model.to(device)

    # -------------------------------------------------------------------------------------- #
    # 建立优化器
    # -------------------------------------------------------------------------------------- #
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM,
                                weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    # -------------------------------------------------------------------------------------- #
    # 建立scheduler
    # -------------------------------------------------------------------------------------- #
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.SOLVER.LR_STEPS, cfg.SOLVER.GAMMA)

    # -------------------------------------------------------------------------------------- #
    # 建立checkpointer,用于保存、加载模型参数
    # -------------------------------------------------------------------------------------- #
    checkpointer = CheckPointer(model, optimizer, scheduler, './checkpoints/', logger=logger)

    # -------------------------------------------------------------------------------------- #
    # 构建数据集、
    # -------------------------------------------------------------------------------------- #
    training_data = cfg

if __name__ == '__main__':
    # -------------------------------------------------------------------------------------- #
    # 获得模型、数据集、训练、测试等相关配置参数
    # -------------------------------------------------------------------------------------- #
    args = parse.get_config()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()  # 该对象的任何属性将不能被修改

    # -------------------------------------------------------------------------------------- #
    # 建立一个logger
    # -------------------------------------------------------------------------------------- #
    logger = setup_logger("SSD", './')
    logger.info(f"Loaded configuration file {args.config_file}")
    logger.info(f"Runing with config:\n{cfg}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = train(cfg, args)
