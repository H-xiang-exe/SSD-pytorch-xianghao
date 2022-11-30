import os

from config import parse
from engine.trainer import Trainer
from utils.logger import setup_logger

if __name__ == '__main__':
    BASE_DIR = os.getcwd()

    # get command line args
    args = parse.get_config()

    # get logger
    logger = setup_logger("SSD", './')
    logger.info(f"Using 1 GPUs")
    logger.info(args)

    trainer = Trainer(args, BASE_DIR)
    trainer.train()
