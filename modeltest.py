import torch.cuda
from torchsummary import summary

from FollowbotNets_PPO import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = FollowbotPolicyNet().to(device)

summary(net, (3, 224, 224))
