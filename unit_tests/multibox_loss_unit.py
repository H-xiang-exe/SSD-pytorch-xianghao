import torch

from solver import multibox_loss
# batch_size = 8, num_priors = 12
location = torch.rand(8, 12, 4)
confidence = torch.rand(8, 12, 21)
priors = torch.rand(12, 4)

predictions = (location, confidence, priors)
num_classes = 21
num_objs = 3

targets = torch.rand(num_classes, num_objs, 5)

loss = multibox_loss.MultiBoxLoss(num_classes=21, overlap_thresh=0.5, variance=[0.1, 0.2], neg_pos=3)
loss_dict = loss(predictions, targets)
print(loss_dict)