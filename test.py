import torch

from xhssd.data import voc0712
from xhssd.utils import augmentations
from xhssd import ssd
from torch.autograd import Variable


def test_model(model, testset, transform, threshold):
    num_images = len(testset)
    for i in range(num_images):
        print(f"Testing image {i + 1:d}/{num_images:d}")
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)  # (h , w, c) -> (c, h, w)
        x = Variable(x.unsqueeze(0))  # (c, h, w) -> (1, c, h, w)

        filename = 'test1.txt'
        with open(filename, mode='a') as f:
            f.write(f'\nGround Truth For {img_id}\n')
            for box in annotation:
                f.write('label:' + ' || '.join(str(b) for b in box) + '\n')

        y = model(x)
        detections = y.data
        print(detections)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            pass


def test_voc(root):
    """
    Args:
        root(str): file path to VOCdevkit
    """
    # Data
    testset = voc0712.VOCDataset(root, ('2007', 'test'))  # 测试时图片不进行数据增强
    num_classes = len(testset.classes)+1
    print(num_classes)
    # model
    model = ssd.build_ssd('test', 300, num_classes)
    model.eval()
    test_model(model, testset, transform=augmentations.BaseTransform(), threshold=0.5)


if __name__ == '__main__':
    test_voc('.\\batchdata\\VOCdevkit')
