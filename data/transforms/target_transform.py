class SSDTargetTransform(object):
    """用于对Annotation中的box坐标和分类进行归一化并返回[[xmin, ymin, xmax, ymax, cls_id], ...]"""

    def __call__(self, img_annotation, classes, width, height):
        """
        Args:
            img_annotation(ET element): the target annotation
            classes(list): class name of object
            width(int): width
            height(int): height
        """
        boxes = []
        for obj in img_annotation.iter('object'):
            difficult = 0
            if obj.find("difficult") != None:
                difficult = obj.find("difficult").text
            cls = obj.find("name").text
            # print(cls)
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            bbox = obj.find("bndbox")

            bndbox = []
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # 对框的坐标进行归一化: 坐标值/宽（高）
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            bndbox.append(cls_id)  # [xmin, ymin, xmax, ymax, cls_id]
            boxes.append(bndbox)
        return boxes