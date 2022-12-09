import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description="Single Shot MultiBox Detector Training With Pytorch")

# Data
data_arg = parser.add_argument_group('Data')
data_arg.add_argument('--config_file', default="configs/voc07.yaml", metavar="FILE", help="path to config file",
                      type=str)

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
misc_arg.add_argument("--log_step", default=10, type=int, help="The interval for logs recorded")
misc_arg.add_argument("--save_step", default=100, type=int, help="The interval for logs recorded")
misc_arg.add_argument("--eval_step", default=3, type=int, help="The interval for logs recorded")


def get_config():
    args, unparsed_args = parser.parse_known_args()

    return args
