import torch
from torchsummary import summary
import xhssd.ssd as ssd
from thop import profile

net = ssd.build_ssd('train')
inputs = torch.ones(8, 3, 300, 300)
for name, param in net.named_parameters():
    print(name, param.shape)

summary(net, (3, 300, 300), device='cpu')
net_ = torch.jit.trace(net, inputs)
net_.save('ssd.pth')