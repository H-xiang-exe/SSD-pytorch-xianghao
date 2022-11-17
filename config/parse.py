import argparse

from data import voc


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description="Single Shot MultiBox Detector Training With Pytorch")

parser.add_argument("--dataset", default='VOC2007',
                    choices=["VOC2007", "VOC2012", "COCO"], type=str, help="VOC or COCO")

parser.add_argument('--batch-size', default=8, type=int, help='batch size of train/test')

parser.add_argument("--lr", default=1e-3, type=float,
                    help="initial learning rate")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum value for optim")
parser.add_argument("--weight_decay", default=5e-4,
                    type=float, help="weight decay for SGD")

parser.add_argument('--num_workers', default=1, type=int)

parser.add_argument("--cuda", default=True, type=str2bool,
                    help="Use CUDA to train modeling")
parser.add_argument("--visdom", default=False, type=str2bool,
                    help="Use Visdom for loss visualization")


def get_config():
    args, unparsed_args = parser.parse_known_args()

    return args
