import torch


@torch.no_grad()
def do_evaluation(model, iteration):
    """
    """
    if isinstance(model,(torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    