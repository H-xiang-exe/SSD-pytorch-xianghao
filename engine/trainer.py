import logging
import os
import time
import datetime

import torch

from config import cfg
from utils.metric_logger import MetricLogger
from engine import eval

def do_train(model, train_dataloader, loss_fn,optimizer, scheduler, checkpointer, device, args):
    """

    Args:
        model:
        train_data_loader:
        optimizer:
        scheduler:
        checkpointer:
        device:
        args:

    Returns:

    """
    # 日志
    logger = logging.getLogger('SSD.trainer')
    logger.info("Start training ...")
    # metric记录
    meters = MetricLogger()
    start_training_time = time.time()
    end = time.time()

    # dataloader数据的批次
    data_iter = iter(train_dataloader)
    start_iter = 0
    model.train()
#     for iteration, (images, targets, _) in enumerate(train_dataloader, start_iter):
    for iteration in range(cfg.SOLVER.MAX_ITER):
        iteration = iteration + 1
        try:
            images, targets, _ = next(data_iter)
        except StopIteration as e:
            data_iter = iter(train_dataloader)
            images, targets, _ = next(data_iter)
            
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        reg_loss, cls_loss = loss_fn(outputs, targets)
        loss = reg_loss + cls_loss
        meters.update(loss=loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)

        # 输出训练日志
        if iteration % args.log_step == 0:
            # 剩余训练时间
            eta_seconds = meters.time.global_avg * (cfg.SOLVER.MAX_ITER - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimeter.join([
                    f"iter: {iteration:06d}",
                    f"lr: {optimizer.param_groups[0]['lr']:.5f}",
                    f"{meters}",
                    f"eta: {eta_string}",
                    f"mem:{torch.cuda.max_memory_allocated() / 1024.0 / 1024.0}M"
                ])
            )
        # 保存当前模型
        if iteration % args.save_step == 0 and iteration != cfg.SOLVER.MAX_ITER:
            checkpointer.save(f"model_{iteration:06d}")
        # 评估当前模型
        if iteration % args.eval_step == 0:
            eval_results = eval.do_evaluation(cfg,model, iteration=iteration)
            model.train()
    checkpointer.save(f"model_final")

    # 计算训练时间
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(f"Total training time: {total_time_str}({total_training_time / cfg.SOLVER.MAX_ITER}s/it)")
