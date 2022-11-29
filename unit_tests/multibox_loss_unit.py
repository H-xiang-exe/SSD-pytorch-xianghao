import numpy as np
import torch

from solver import multibox_loss

# --------------------------------------------------------
# default config
# --------------------------------------------------------
batch_size = 8
input_shape = (300, 300)
feature_shape = (3, 3)
num_priors = feature_shape[0] * feature_shape[1]
num_anchors = 4
# multiple box widths/heights for each meshgrid
box_widths = torch.tensor([213., 237., 150.61, 301.22])
box_heights = torch.tensor([213., 237., 301.22, 150.61])

# --------------------------------------------------------
# Get default prior boxes
# --------------------------------------------------------
grid_width, grid_height = input_shape[0] / feature_shape[0], input_shape[1] / feature_shape[1]
lin_x = torch.linspace(grid_width * 0.5, input_shape[0] - grid_width * 0.5, feature_shape[0])
lin_y = torch.linspace(grid_height * 0.5, input_shape[1] - grid_height * 0.5, feature_shape[0])
centers_y, centers_x = torch.meshgrid(lin_x, lin_y)
centers_x = centers_x.reshape(-1, 1)
centers_y = centers_y.reshape(-1, 1)

priors = torch.cat((centers_x, centers_y), dim=-1)  # (num_centers, 2)
priors = torch.tile(priors, (1, num_anchors * 2))  # (num_centers, 16)
priors[:, ::4] -= 0.5 * box_widths
priors[:, 1::4] -= 0.5 * box_heights
priors[:, 2::4] += 0.5 * box_widths
priors[:, 3::4] += 0.5 * box_heights
priors[:, ::2] /= input_shape[0]
priors[:, 1::2] /= input_shape[1]
priors = torch.clamp(priors, 0.0, 1.0).reshape(-1, 4)
# print(priors.shape)

# --------------------------------------------------------
# Get target boxes locations and confidences
# --------------------------------------------------------
num_objs = 3
target_location_left = torch.rand(batch_size, num_objs, 1) * 0.3
target_location_top = torch.rand(batch_size, num_objs, 1) * 0.3
target_location_right = 0.5 + 0.5 * torch.rand(batch_size, num_objs, 1)
target_location_bottom = 0.5 + 0.5 * torch.rand(batch_size, num_objs, 1)
target_cls = torch.randint(0, 20, (batch_size, num_objs,1))
targets = torch.cat(
    (target_location_left, target_location_top, target_location_right, target_location_bottom, target_cls),
    dim=-1)

# ----------------------------------------------------------
# random get predictions
# ----------------------------------------------------------
location_left = torch.rand(batch_size, num_priors, 1) * 0.3
location_top = torch.rand(batch_size, num_priors, 1) * 0.3
location_right = 0.5 + 0.5 * torch.rand(batch_size, num_priors, 1)
location_bottom = 0.5 + 0.5 * torch.rand(batch_size, num_priors, 1)
predict_locations = torch.cat([location_left, location_top, location_right, location_bottom], dim=-1)
# print(predict_locations.shape)
predict_confidences = torch.rand(batch_size, num_priors, 21)
predictions = (predict_locations, predict_confidences, priors)

loss = multibox_loss.MultiBoxLoss(num_classes=21, overlap_thresh=0.5, variance=[0.1, 0.2], neg_pos=3,
                                  device=torch.device('cpu'))
loss_dict = loss(predictions, targets)
print(loss_dict)
