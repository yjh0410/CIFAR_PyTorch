import torch.nn as nn
from utils import Conv2d


class cifar_net(nn.Module):
    def __init__(self):
        super(cifar_net, self).__init__()
        self.conv_1 = nn.Sequential(
            Conv2d(3, 32, 3, padding=1, leaky=True),
            Conv2d(32, 64, 3, padding=1, stride=2, leaky=True)
        )
        self.conv_2 = nn.Sequential(
            Conv2d(64, 64, 3, padding=1, leaky=True),
            Conv2d(64, 128, 3, padding=1, stride=2, leaky=True)
        )
        self.conv_3 = nn.Sequential(
            Conv2d(128, 128, 3, padding=1, leaky=True),
            Conv2d(128, 256, 3, padding=1, stride=2, leaky=True)
        )
        self.conv_4 = nn.Sequential(
            Conv2d(256, 256, 3, padding=1, leaky=True),
            Conv2d(256, 512, 3, padding=1, stride=2, leaky=True)
        )
        self.conv_5 = nn.Sequential(
            Conv2d(512, 512, 3, padding=1, leaky=True),
            Conv2d(512, 1024, 3, padding=1, stride=2, leaky=True)
        )
        self.conv_6 = nn.Sequential(
            Conv2d(1024, 512, 1, leaky=True),
            Conv2d(512, 1024, 3, padding=1, leaky=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(1024, 10, kernel_size=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x.view(x.size(0), -1)
