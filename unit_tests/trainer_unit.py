import sys
sys.path.append('../')
from engine.trainer import Trainer
from config import parse

BASE_DIR = '/root/autodl-tmp/SSD-pytorch-xianghao'
args = parse.get_config()


trainer = Trainer(args, BASE_DIR)
trainer.train()
