import argparse

from data import voc


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description="Single Shot MultiBox Detector Training With Pytorch")

# Data
data_arg = parser.add_argument_group('Data')
data_arg.add_argument("--dataset", default='VOC2007', choices=["VOC2007", "VOC2012", "COCO"], type=str,
                      help="VOC or COCO")
data_arg.add_argument('--epoch', default=150, type=int, help='train phase epochs')
data_arg.add_argument('--batch_size', default=8, type=int, help='batch size of train/test')
data_arg.add_argument('--test_batch_size', default=8, type=int, help='batch size of train/test')
data_arg.add_argument('--num_workers', default=1, type=int)
data_arg.add_argument('--pin_memory', default=False, type=bool)

# solver
solver_arg = parser.add_argument_group('Solver')
solver_arg.add_argument("--phase", default='train', type=str, help="train or test")
solver_arg.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
solver_arg.add_argument("--momentum", default=0.9, type=float, help="Momentum value for optim")
solver_arg.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay for SGD")

# Misc
misc_arg = parser.add_argument_group('Misc')
misc_arg.add_argument("--cuda", default=True, type=str2bool, help="Use CUDA to train modeling")
misc_arg.add_argument("--visdom", default=False, type=str2bool, help="Use Visdom for loss visualization")


def get_config():
    args, unparsed_args = parser.parse_known_args()

    return args
