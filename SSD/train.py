import os

import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import visdom

from SSD.data import config, coco, voc0712
from ssd import build_ssd


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description="Single Shot MultiBox Detector Training With Pytorch")
parser.add_argument("--dataset", default='VOC',
                    choices=["VOC", "COCO"], type=str, help="VOC or COCO")
parser.add_argument("--dataset_root", default=voc0712.VOC_ROOT,
                    help="Dataset root directory path")

parser.add_argument("--lr", default=1e-3, type=float,
                    help="initial learning rate")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum value for optim")
parser.add_argument("--weight_decay", default=5e-4,
                    type=float, help="weight decay for SGD")

parser.add_argument("--cuda", default=True, type=str2bool,
                    help="Use CUDA to train model")
parser.add_argument("--visdom", default=False, type=str2bool,
                    help="Use Visdom for loss visualization")

args = parser.parse_args()


def train():
    if args.dataset == "COCO":
        if args.dataset_root == voc0712.VOC_ROOT:
            if not os.path.exists(coco.COCO_ROOT):
                parser.error("Must specify dataset_root if specifying dataset")
            print(
                "WARGING: Using default COCO dataset_root because --dataset_root was not specified")
            args.dataset_root = coco.COCO_ROOT
        cfg = config.coco
        dataset = coco.COCODetection()
    elif args.dataset == "VOC":
        if args.dataset_root == coco.COCO_ROOT:
            parser.error(
                "Must specify dataset_root if specifying dataset_root")
        cfg = config.voc
        dataset = voc0712.VOCDetection()

    if args.visdom:
        viz = visdom.Visdom()

    ssd_net = build_ssd()
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.cuda:
        net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    citerion = MultiBoxLoss()

    # prepare dataloader


    # ---------------------------------
    # Start Model Training ...

    # ---------------------------------


    net.train()



if __name__ == "__main__":
    train()
