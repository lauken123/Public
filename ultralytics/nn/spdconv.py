import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

def autopad(k, p=None, d=1):
    """自动计算 padding (same)"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SPDConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # 显式设置 padding=0，避免 autopad 问题
        self.conv = Conv(c1 * 4, c2, k, s, p=0, g=g, act=act)

    def forward(self, x):
        if x.shape[-1] % 2 != 0 or x.shape[-2] % 2 != 0:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        return self.conv(x)

class SimAM(nn.Module):
    """SimAM 注意力模块（无参数）"""
    def __init__(self, c1=256, c2=256, e_lambda=1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

'''class HGConv(nn.Module):
    def __init__(self, c1, c2, k_neighbors=9, stride=1):
        super().__init__()
        self.stride = stride
        self.out_channels = c2

        if stride == 2:
            self.downsample = nn.Conv2d(c1, c1, 3, stride=2, padding=1, groups=c1)
        else:
            self.downsample = nn.Identity()

        self.f_embed = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.downsample(x)
        B, C, H, W = x.shape
        feat = self.f_embed(x)

        c2 = self.out_channels   # 或 feat.size(1)
        x_flat = feat.view(B, c2, -1).permute(0, 2, 1)
        attn = torch.matmul(x_flat, x_flat.transpose(-1, -2))
        attn = F.softmax(attn / (c2 ** 0.5), dim=-1)
        out = torch.matmul(attn, x_flat).permute(0, 2, 1).view(B, c2, H, W)

        return self.act(self.bn(out + feat))'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class HGConv(nn.Module):
    """
    餘弦提純版 HGConv (高階通道自注意力)
    使用 L2 Normalization 將點積約束為餘弦相似度，在數學結構上徹底杜絕 NaN/Inf 溢出。
    """
    def __init__(self, c1, c2, k_neighbors=9, stride=1):
        super().__init__()
        self.stride = stride
        self.out_channels = c2

        if stride == 2:
            self.downsample = nn.Conv2d(c1, c1, 3, stride=2, padding=1, groups=c1)
        else:
            self.downsample = nn.Identity()

        self.f_embed = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        
        # 引入一個可學習的自動縮放系數（類似 Transformer 中的溫度調節）
        self.scale = nn.Parameter(torch.ones(1, c2, 1) * 10.0)

    def forward(self, x):
        x = self.downsample(x)
        B, C, H, W = x.shape
        feat = self.f_embed(x)
        
        c2 = self.out_channels
        x_flat = feat.view(B, c2, -1) # (B, c2, H*W)
        
        # ================================================================
        # 【核心抗溢出機制】對空間維度進行 L2 歸一化
        # 使得每個通道的向量模長為 1，相乘後的矩陣元素絕對在 [-1, 1] 內，全面根除 NaN
        # ================================================================
        x_norm = F.normalize(x_flat, p=2, dim=-1) 
        
        # 計算通道自注意力矩陣 (B, c2, c2)
        attn = torch.matmul(x_norm, x_norm.transpose(-1, -2))
        temperature = math.sqrt(c2)#最新的yolo26m修改的唯一一行代码
        # 透過可學習的 scale 放大特徵對比度，並進行 Softmax
        attn = F.softmax(attn * self.scale, dim=-1)
        
        # 特徵重構（使用原本未歸一化的 x_flat，確保不丟失縮放语义）
        out = torch.matmul(attn, x_flat).view(B, c2, H, W)
        # ================================================================

        return self.act(self.bn(out + feat))