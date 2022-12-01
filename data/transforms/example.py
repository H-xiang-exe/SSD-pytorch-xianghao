import matplotlib.pyplot as plt
import cv2
import numpy as np

from data import transforms

if __name__ == '__main__':
    img_path = "../../batchdata/VOCdevkit/VOC2007/JPEGImages/000012.jpg"
    image = cv2.imread(img_path)
    # print(image)
    boxes = np.array([[156, 97, 351, 270]], dtype=np.float32)
    height, width, channels = image.shape
    # print(height, width)
    boxes[:, ::2] /= width
    boxes[:, 1::2] /= height
    # print(boxes)
    labels = np.array([3])

    # 测试SSDAugmentation
    plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    augument = transforms.build_transform()
    image, boxes, labels = augument(image, boxes, labels)
    cv2.imshow('image', image)  # BGR
    cv2.waitKey(0)
    tmp = []
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            b, g, r = image[h][w]
            if b > 0 or g > 0 or r > 0:
                tmp.append([h, w, b, g, r])
    print(tmp)
    plt.subplot(122)  # RGB
    plt.imshow(image)
    plt.show()

    # # 测试TestTransform
    # # cv2.imshow('image', image)
    # # cv2.waitKey(0)
    # plt.figure()
    # plt.subplot(121)
    # pre_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(pre_img)
    # test_transform = TestTransform()
    # image, boxes, labels = test_transform(image)
    # print(image.shape)
    # # cv2.imshow('image after test_transform', image)
    # # cv2.waitKey(0)
    # plt.subplot(122)
    # suf_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(suf_img)
    #
    # plt.colorbar()
    # plt.show()