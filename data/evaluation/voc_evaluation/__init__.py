import logging
import numpy as np

from .eval_detection_voc import eval_detection_voc


def voc_evaluation(dataset, predictions, ):
    """evaluate dataset using different model based on dataset type.

    Args:
        dataset (Dataset): Dataset object
        predictions (list[image_id: container(boxes, labels, scores)]): Each item in the list represents the
            prediction results for one image.
    """
    # 数据集类别
    class_names = dataset.class_names

    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []

    gt_boxes_list = []
    gt_labels_list = []
    gt_difficults_list = []

    # 逐一考虑数据集中的每张图片
    for i in range(len(dataset)):
        # -------------------------------------------------------------------------------------- #
        # 获得gt boxes, gt labels, is difficult
        # -------------------------------------------------------------------------------------- #
        # 获得图片的id和标签
        image_id, annotation = dataset.get_anno(i)
        gt_boxes, gt_labels, is_difficult = annotation

        gt_boxes_list.append(gt_boxes)
        gt_labels_list.append(gt_labels)
        gt_difficults_list.append(is_difficult.astype(np.bool))

        # -------------------------------------------------------------------------------------- #
        # 获得pred boxes, pred labels, pred scores
        # -------------------------------------------------------------------------------------- #
        img_height, img_width = dataset.get_image_info(i)  # 获得原图片的宽和高
        prediction = predictions[i]
        prediction = prediction.resize(size=(img_width, img_height)).numpy()  # 将box还原成原图像尺度下的box
        boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']

        pred_boxes_list.append(boxes)
        pred_labels_list.append(labels)
        pred_scores_list.append(scores)

    result = eval_detection_voc(pred_bboxes=pred_boxes_list,
                                pred_labels=pred_labels_list,
                                pred_scores=pred_scores_list,
                                gt_bboxes=gt_boxes_list,
                                gt_labels=gt_labels_list,
                                gt_difficults=gt_difficults_list,
                                iou_thresh=0.5,
                                use_07_metric=True)
    logger = logging.getLogger("SSD.inference")
    result_str = "mAP: {:.4f}\n".format(result["map"])
    metrics = {'mAP': result["map"]}
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        metrics[class_names[i]] = ap
        result_str += "{:<16}: {:.4f}\n".format(class_names[i], ap)
    logger.info(result_str)

    return dict(metrics=metrics)
