def voc_evaluation(dataset, predictions,):
    """evaluate dataset using different model based on dataset type.

    Args:
        dataset (Dataset): Dataset object
        predictions (list[image_id: container(boxes, labels, scores)]): Each item in the list represents the
            prediction results for one image.
    """
