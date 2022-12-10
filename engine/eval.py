import torch
from data.build import make_data_loader
from data.evaluation import evaluate


@torch.no_grad()
def inference(model, data_loader, device):
    results = {}
    for batch_idx, (images, targets, image_ids) in enumerate(data_loader):
        images = images.to(device)
        outputs = model(images)  # Container(boxes, scores, labels) = model(images)

        results.update({int(image_id): output for image_id, output in zip(image_ids, outputs)})
    return results


@torch.no_grad()
def do_evaluation(cfg, model, **kwargs):
    if isinstance(model, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = model.module
    # 进入评估模式
    model.eval()
    # 加载测试数据集
    print("Loading dataloaders ...")
    test_dataloaders = make_data_loader(cfg, is_train=False)
    print("Loading Finished.")
    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 在测试集上测试（有可能有多个测试集，dataloader是个list)
    eval_results = []
    for test_dataset_name, test_dataloader in zip(cfg.TEST, test_dataloaders):
        print("Start Inference")
        predictions = inference(model, test_dataloader, device)
        eval_result = evaluate(test_dataloader.dataset, predictions)
        eval_results.append(eval_result)
    return eval_results
