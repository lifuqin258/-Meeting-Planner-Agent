from math import sqrt
import einops
import torch
import torch.nn.functional as F
from torch import nn


class CAM(nn.Module):
    """Cross-Attention Module (保持不变)"""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # visual qkv
        self.v_q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v_k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # tactile qkv
        self.t_q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.t_k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.t_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self._norm_fact = 1 / sqrt(in_channels)

    def forward(self, visual, tactile):
        # visual qkv
        v_q = einops.rearrange(self.v_q(visual), 'b c h w -> b (h w) c')
        v_k_trans = einops.rearrange(self.v_k(visual), 'b c h w -> b c (h w)')
        v_v = einops.rearrange(self.v_v(visual), 'b c h w -> b (h w) c')
        # tactile qkv
        t_q = einops.rearrange(self.t_q(tactile), 'b c h w -> b (h w) c')
        t_k_trans = einops.rearrange(self.t_k(tactile), 'b c h w -> b c (h w)')
        t_v = einops.rearrange(self.t_v(tactile), 'b c h w -> b (h w) c')

        attention_vt = F.softmax(torch.bmm(v_q, t_k_trans), dim=-1) * self._norm_fact
        out_vt = torch.bmm(attention_vt, t_v)
        out_vt = einops.rearrange(out_vt, 'b (h w) c -> b c h w', h=visual.shape[2], w=visual.shape[3])

        attention_tv = F.softmax(torch.bmm(t_q, v_k_trans), dim=-1) * self._norm_fact
        out_tv = torch.bmm(attention_tv, v_v)
        out_tv = einops.rearrange(out_tv, 'b (h w) c -> b c h w', h=tactile.shape[2], w=tactile.shape[3])

        return out_vt, out_tv


class VTFNet(nn.Module):
    """视觉-触觉融合网络 (使用SimpleCNN替代DeepLabV3+)"""

    def __init__(self, visual_channels=256, tactile_channels=256, feature_dim=256):
        super().__init__()
        self.visual_channels = visual_channels
        self.tactile_channels = tactile_channels


        self.fusion_module = CAM(in_channels=feature_dim)
        self.final_conv = nn.Conv2d(1024, 256, kernel_size=1)



    def forward(self, visual, tactile):

        # 跨模态注意力融合
        out_vt, out_tv = self.fusion_module(visual, tactile)

        # 特征拼接
        combined = torch.cat([visual, tactile, out_vt, out_tv], dim=1)
        fusion = self.final_conv(combined)

        return fusion