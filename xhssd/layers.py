import torch
import torch.nn as nn

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))  # shape = (20,)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):  # (B, C, H, W)
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps  # (B, 1, H, W)
        x = torch.div(x, norm)  # (B, C, H, W)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x  # (1,20,1,1)
        return out

