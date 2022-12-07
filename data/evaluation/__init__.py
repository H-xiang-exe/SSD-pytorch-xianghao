from data.datasets import VOCDataset, COCODataset
from data.evaluation.voc_evaluation import voc_evaluation
from data.evaluation.coco_evaluation import coco_evaluation


def evaluate(dataset, predictions, *args, **kwargs):
    """evaluate dataset using different model based on dataset type.

    Args:
        dataset (Dataset): Dataset object
        predictions (list[image_id: container(boxes, labels, scores)]): Each item in the list represents the
            prediction results for one image.
    """
    if isinstance(dataset, VOCDataset):
        return voc_evaluation()
    elif isinstance(dataset, COCODataset):
        return coco_evaluation()
    else:
        raise NotImplementedError
