import torch
import torch.nn as nn

class SPDConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # Space-to-Depth 操作：将空间维度上的信息转移到通道维度
        self.spd = self.space_to_depth
        self.conv = nn.Conv2d(c1 * 4, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    @staticmethod
    def space_to_depth(x):
        # 将输入切分为4份并沿通道拼接，实现 2 倍下采样且保留信息
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                          x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)

    def forward(self, x):
        return self.act(self.bn(self.conv(self.spd(x))))