from xhssd.data import voc0712
from xhssd.utils import augmentations
from xhssd import ssd

def test_model(model, testset, transform, threshold):
    num_images = len(testset)
    for i in range(num_images):
        print(f"Testing image {i+1:d}/{num_images:d}")
        img = testset.pull_image(i)

def test_voc(root):
    """
    Args:
        root(str): file path to VOCdevkit
    """
    # Data
    testset = voc0712.VOCDataset(root, ['2007', 'test'], transform=augmentations.SSDAugmentation())
    num_classes = testset.classes
    # model
    model = ssd.build_ssd('test', 300, num_classes)
    model.eval()
    test_model(model, testset, transform=augmentations.BaseTransform(), threshold=0.5)


if __name__ == '__main__':
    test_voc('./batchdata/VOCdevkit')
