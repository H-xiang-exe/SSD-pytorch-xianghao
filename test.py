import logging
from engine.eval import do_evaluation

import torch.backends.cudnn
from torch.utils.data import DataLoader

from config import parse, cfg
from modeling import build_model
from utils.logger import setup_logger
from utils.checkpoint import CheckPointer
from data.build import make_data_loader
from solver.multibox_loss import MultiBoxLoss
def test(cfg, args):
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

    print(model)

    # ckp = torch.load('./checkpoints/model_000003.pth', map_location=torch.device('cpu'))
    # model.load_state_dict(ckp['model'])
    # -------------------------------------------------------------------------------------- #
    # 构建Dataloader
    # -------------------------------------------------------------------------------------- #
    # test_dataloader = make_data_loader(cfg, is_train=False)[0]

    # with torch.no_grad():
    #     model.eval()
    #     # dataloader数据的批次
    #     #     for iteration, (images, targets, _) in enumerate(train_dataloader, start_iter):
    #     for batch_idx, (images, targets, image_ids) in enumerate(test_dataloader):
    #         images = images.to(device)
    #         targets = targets.to(device)

    #         outputs = model(images)

            
    #         exit()
    do_evaluation(cfg, model, iteration=0)

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

    test(cfg, args)
