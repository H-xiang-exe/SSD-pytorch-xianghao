class PostProcessor(object):
    def __init__(self, cfg):
        super(PostProcessor, self).__init__()
        self.cfg = cfg

    def __call__(self, detections):
        # 模型输出
        batch_scores, batch_boxes = detections
        device = batch_scores.device
        batch_size = batch_scores.shape[0]

        results = []
        for scores, boxes in zip(batch_scores, batch_boxes):
            num_boxes = scores.shape[0]  # scores: (num_boxes, 4)
            num_classes = boxes.shape[0]  # boxes: (num_boxes, cls)

            boxes = boxes.view(num_boxes, 1, 4).expand(num_boxes, num_classes, 4)
            print(boxes)
