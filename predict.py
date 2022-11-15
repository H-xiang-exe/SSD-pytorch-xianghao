import os
import cv2
import torch

from xhssd import ssd300_vgg16
from xhssd.utils import detection

if __name__ == '__main__':
    model = ssd300_vgg16.build_ssd('test')
    data_root = "./batchdata/VOCdevkit"

    dir_origin_path = os.path.join(data_root, "VOC2007/JPEGImages")
    while True:
        # img = input('Input image filename:')
        img = '000005.jpg'
        img = os.path.join(dir_origin_path, img)

        # cpu or cuda
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)
        try:
            image = cv2.imread(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            detector = detection.Detection(input_shape=[300,300], device=device)
            r_image = detector.detect_image(model, image)
            # r_image.show()
            break

