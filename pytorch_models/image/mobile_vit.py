# https://arxiv.org/abs/2110.02178
# https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py

import torch
from torch import Tensor, nn

from ..transformer import Encoder


# from MobileNetv2
class MBConv(nn.Sequential):
    def __init__(self, in_dim: int, expansion: int, out_dim: int, stride: int = 1) -> None:
        hidden_dim = in_dim * expansion
        self.residual = (in_dim == out_dim) and (stride == 1)
        super().__init__(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x) if self.residual else super().forward(x)


def unfold(x: Tensor, patch_size: int) -> tuple[Tensor, tuple[int, int]]:
    N, C, H, W = x.shape
    nH = H // patch_size
    nW = W // patch_size
    return (
        x.view(N, C, nH, patch_size, nW, patch_size)
        .permute(0, 3, 5, 2, 4, 1)
        .reshape(N, patch_size * patch_size, nH * nW, C)
    ), (nH, nW)


def fold(x: Tensor, patch_size: int, n_patches: tuple[int, int]) -> Tensor:
    nH, nW = n_patches
    N = x.shape[0]
    C = x.shape[-1]
    return (
        x.view(N, patch_size, patch_size, nH, nW, C)
        .permute(0, 5, 3, 1, 4, 2)
        .reshape(N, C, nH * patch_size, nW * patch_size)
    )


class MobileViTBlock(nn.Module):
    patch_size = 2

    def __init__(self, in_dim: int, d_model: int, n_layers: int) -> None:
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.SiLU(),
            nn.Conv2d(in_dim, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
        )
        self.transformer = Encoder(n_layers, d_model, mlp_ratio=2.0)
        self.out_proj = nn.Sequential(nn.Conv2d(d_model, in_dim, 1, bias=False), nn.BatchNorm2d(in_dim), nn.SiLU())
        self.out_fusion = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, 3, 1, 1, bias=False), nn.BatchNorm2d(in_dim), nn.SiLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        out, n_patches = unfold(self.in_conv(x), self.patch_size)
        out = fold(self.transformer(out), self.patch_size, n_patches)
        return self.out_fusion(torch.cat([x, self.out_proj(out)], 1))


class MobileViT(nn.Sequential):
    def __init__(self, channels: list[int], d_models: list[int], out_dim: int, expansion: int) -> None:
        super().__init__(
            nn.Sequential(
                nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.SiLU()),
                MBConv(16, expansion, channels[0]),
            ),
            nn.Sequential(
                MBConv(channels[0], expansion, channels[1], 2),
                MBConv(channels[1], expansion, channels[1]),
                MBConv(channels[1], expansion, channels[1]),
            ),
            nn.Sequential(
                MBConv(channels[1], expansion, channels[2], 2),
                MobileViTBlock(channels[2], d_models[0], 2),
            ),
            nn.Sequential(
                MBConv(channels[2], expansion, channels[3], 2),
                MobileViTBlock(channels[3], d_models[1], 4),
            ),
            nn.Sequential(
                MBConv(channels[3], expansion, channels[4], 2),
                MobileViTBlock(channels[4], d_models[2], 3),
                nn.Sequential(
                    nn.Conv2d(channels[4], out_dim, 1),
                    nn.BatchNorm2d(out_dim),
                    nn.SiLU(),
                ),
            ),
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1)),
        )

    @staticmethod
    def from_apple(variant: str, *, pretrained: bool = False) -> "MobileViT":
        channels, d_models, out_dim, expansion = dict(
            xxs=([16, 24, 48, 64, 80], [64, 80, 96], 320, 2),
            xs=([32, 48, 64, 80, 96], [96, 120, 144], 384, 4),
            s=([32, 64, 96, 128, 160], [144, 192, 240], 640, 4),
        )[variant]

        m = MobileViT(channels, d_models, out_dim, expansion)

        if pretrained:
            pass

        return m
