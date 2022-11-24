import os

from config import parse
from engine.trainer import Trainer

if __name__ == '__main__':
    BASE_DIR = os.getcwd()
    args = parse.get_config()

    trainer = Trainer(args, BASE_DIR)
    trainer.train()

