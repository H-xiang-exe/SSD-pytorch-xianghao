def voc_evaluation(dataset, predictions,):
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
    gt_scores_list = []

    # 逐一考虑数据集中的每张图片
    for i in range(len(dataset)):
        pass