import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY


class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1, stride=s, bias=bias)
        self.relu = relu

    def forward(self, x):
        out = self.conv(x)
        return torch.nn.functional.leaky_relu(out, negative_slope=0.05) if self.relu else out


class SPABV2_Residual(nn.Module):
    """SPANV2 block with attention moved to after the first 3x3 conv."""

    def __init__(self, in_channels, mid_channels=None, out_channels=None,
                 bias=False, use_span_attn=False):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_span_attn = use_span_attn
        self.c1 = Conv3XC(in_channels, mid_channels, gain1=2, bias=bias)
        self.c2 = Conv3XC(mid_channels, mid_channels, gain1=2, bias=bias)
        self.c3 = Conv3XC(mid_channels, out_channels, gain1=2, bias=bias)
        self.act = nn.ReLU(inplace=True)
        if in_channels == out_channels:
            self.guidance_map_conv = nn.Conv2d(out_channels, out_channels, 1, bias=True)

    def forward(self, x):
        f1 = self.c1(x)

        if self.in_channels == self.out_channels:
            if self.use_span_attn:
                import warnings
                warnings.warn(
                    "use_span_attn=True is not supported in SPABV2_Residual "
                    "(attention moved to c1); using PyTorch fallback."
                )
            guidance_map = self.guidance_map_conv(f1)
            guidance_map = (
                torch.sign(guidance_map)
                * torch.abs(guidance_map).clamp_min(1e-6).pow(1.4)
            )
            f1 = (x + f1) * guidance_map

        f1 = self.act(f1)

        if self.in_channels != self.out_channels:
            f2 = self.act(self.c2(f1))
            f3 = self.c3(f2)
            return self.act(f3)

        f2 = self.act(self.c2(f1))
        f3 = self.c3(f2)
        return f3


@ARCH_REGISTRY.register()
class SPANV2_ESR_Residual(nn.Module):
    """SPANV2 variant with attention applied after the first 3x3 conv."""

    def __init__(self, num_in_ch=1, num_out_ch=1, feature_channels=32, upscale=4,
                 bias=False, img_range=1.0, rgb_mean=None, use_span_attn=False):
        super().__init__()
        self.block_1 = SPABV2_Residual(num_in_ch, feature_channels, feature_channels,
                                       bias, use_span_attn)
        self.block_2 = SPABV2_Residual(feature_channels, bias=bias, use_span_attn=use_span_attn)
        self.block_3 = SPABV2_Residual(feature_channels, bias=bias, use_span_attn=use_span_attn)
        self.block_4 = SPABV2_Residual(feature_channels, bias=bias, use_span_attn=use_span_attn)
        self.block_5 = SPABV2_Residual(feature_channels, bias=bias, use_span_attn=use_span_attn)
        near_channels = num_in_ch * upscale ** 2
        self.conv_near = nn.Conv2d(num_in_ch, near_channels, 3, padding=1,
                                   groups=num_in_ch, bias=False)
        with torch.no_grad():
            self.conv_near.weight.zero_()
            for channel in range(num_in_ch):
                start = channel * upscale ** 2
                self.conv_near.weight[start:start + upscale ** 2, 0, 1, 1] = 1.0
        cat_channels = near_channels + feature_channels
        self.depthwise_conv = nn.Conv2d(cat_channels, cat_channels, 3, padding=1,
                                        groups=cat_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(cat_channels, num_out_ch * upscale ** 2, 1, bias=bias)
        self.depth_to_space = nn.PixelShuffle(upscale)

    def forward(self, x):
        out_near = self.conv_near(x)
        out = self.block_1(x)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        out = torch.cat([out_near, out], dim=1)
        out = self.depthwise_conv(out)
        return self.depth_to_space(self.pointwise_conv(out))
