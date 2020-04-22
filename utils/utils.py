import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, padding=0, stride=1, dilation=1, leaky=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True) if leaky else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
