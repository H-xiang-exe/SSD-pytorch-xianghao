import torch

from data.voc0712 import VOCDataset
from ssd import SSD

# -----------------------
# 准备数据
training_data = VOCDataset()
test_data = VOCDataset()
# -----------------------

# -----------------------
# 准备模型
model = SSD()