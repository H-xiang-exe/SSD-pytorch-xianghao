from xhssd.data import voc0712


def test_voc(root):
    """
    Args:
        root(str): file path to VOCdevkit
    """
    # Data
    testset = voc0712.VOCDataset()

if __name__ == '__main__':
    test_voc()