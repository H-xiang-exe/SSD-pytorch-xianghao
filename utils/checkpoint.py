import logging
import os.path

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel


class CheckPointer(object):
    _last_checkpoint_name = 'last_checkpoint.txt'

    def __init__(self, model, optimizer=None, scheduler=None, save_dir="",
                 # save_to_disk=None,
                 logger=None):
        """For saving checkpoint.

        Args:
            model:
            optimizer:
            scheduler:
            save_dir:
            # save_to_disk:
            logger:
        """
        super(CheckPointer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        # self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return
        # if self.save_to_disk is None:
        #     return

        data = {}
        # 保存model
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            data['model'] = self.model.module.state_dict()
        else:
            data['model'] = self.model.state_dict()

        # 保存optimizer
        if self.optimizer is not None:
            data['optimizer'] = self.optimizer.state_dict()
        # 保存scheduler
        if self.scheduler is not None:
            data['scheduler'] = self.scheduler.state_dict()

        data.update(kwargs)

        save_file = os.path.join(self.save_dir, f"{name}.pth")
        self.logger.info(f"Saving checkpoint to {save_file}")
        torch.save(data, save_file)

    def load(self, ckp_file=None):
        if ckp_file is None:
            self.logger.info("No checkpoint found.")
            raise FileNotFoundError
        self.logger.info(f"Loading checkpoint from {ckp_file}")
        checkpoint = self._load_file(ckp_file)
        model = self.model
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = self.model.module

        model.load_state_dict(checkpoint.pop("model"))
        if "optimizer" in checkpoint and self.optimizer is not None:
            self.logger.info(f"Loading optimizer from {ckp_file}")
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler is not None:
            self.logger.info(f"Loading scheduler from {ckp_file}")
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        return checkpoint

    def _load_file(self, file):
        if file.startswith("http"):
            # if the file is a url path, download it and cache it
            raise NotImplementedError("Loading parameters via URL is currently not supported.")
        return torch.load(file, map_location=torch.device("cpu"))
