from config import parse
from engine import trainer

if __name__ == '__main__':
    args = parse.get_config()
    trainer = trainer.Trainer(args)

