import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg11
from RLFramework.net import *


def addnoise(input, level=0.1):
    return input + torch.randn(input.shape).to(input) * level

class FollowbotPolicyNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.pos_encoding = torch.stack([
        #     (torch.arange(224) * torch.ones((224, 1))) / 112 - 1,
        #     (torch.arange(224).reshape(-1, 1) * torch.ones((1, 224))) / 112 - 1
        # ])
        self.head = nn.Sequential(
            # nn.LayerNorm([3, 224, 224]),
            # added LayerNorm
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(64, 16, 5, 1, 2),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
        )
        # self.head = vgg11(pretrained=True).features
        self.body = nn.Sequential(
            nn.Conv2d(16, 4, 5, 1, 2),
            nn.AvgPool2d(4, 4)
        )
        # self.body = nn.Sequential(
        #     nn.Linear(25088, 1024),
        #     nn.ReLU()
        # )
        # self.tails = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(512, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 3)
        #     ),
        #     nn.Sequential(
        #         nn.Linear(512, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 3)
        #     )
        # ])
        self.tail = nn.Sequential(
            nn.Linear(196, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        # self.tail = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 4)
        # )

    def forward(self, x):
        # self.head.eval()
        # with torch.no_grad():
        x = addnoise(x)
        x = self.head(x)
        x = self.body(x)
        x = x.reshape(x.shape[0], -1)
        x = self.tail(x) #.reshape(x.shape[0], -1)
        # print(x)
        return x
        # results = []
        # for tail in self.tails:
        #     results.append(tail(x))

        # return results


class FollowbotValueNet(ValueNet):
    def __init__(self, policy_net: FollowbotPolicyNet, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = policy_net.head
        # self.head = vgg11(pretrained=True).features
        self.body = nn.Sequential(
            nn.Conv2d(16, 1, 5, 1, 2),
            nn.AvgPool2d(4,4)
        )
        self.tail = nn.Sequential(
            nn.Linear(49, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # self.tail = nn.Sequential(
        #     nn.Linear(25088, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1)
        # )

    def forward(self, state):
        # self.head.eval()
        # with torch.no_grad():
        x = addnoise(state)
        x = self.head(x)
        x = self.body(x)
        x = x.reshape(x.shape[0], -1)
        x = self.tail(x)

        return x
