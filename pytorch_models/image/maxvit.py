# https://arxiv.org/pdf/2204.01697.pdf
# https://github.com/google-research/maxvit

import torch
from torch import Tensor, nn

from ..transformer import MHA, MLP


def conv_norm_act(in_dim: int, out_dim: int, kernel_size: int, stride: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_dim, eps=1e-3, momentum=0.01),
        nn.GELU(),
    )


class SqueezeExcitation(nn.Sequential):
    def __init__(self, dim: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.SiLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * super().forward(x)


# pre-norm MBConv
# NOTE: we don't include stochastic depth
class MBConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        hidden_dim = in_dim * 4
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            conv_norm_act(in_dim, hidden_dim, 1),
            conv_norm_act(hidden_dim, hidden_dim, 3, stride, hidden_dim),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, out_dim, 1),
        )

        if stride > 1:
            self.skip = nn.Sequential(nn.AvgPool2d(stride), nn.Conv2d(in_dim, out_dim, 1))
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.skip(x) + self.residual(x)


def block(x: Tensor, size: int) -> Tensor:
    N, H, W, C = x.shape
    nH = H // size
    nW = W // size
    x = x.view(N, nH, size, nW, size, C).transpose(2, 3).reshape(N, nH * nW, size * size, C)
    return x, nH, nW


def unblock(x: Tensor, nH: int, nW: int, size: int) -> Tensor:
    N, _, _, C = x.shape
    return x.view(N, nH, nW, size, size, C).transpose(2, 3).reshape(N, nH * size, nW * size, C)


# similar to MobileViT's unfold
def grid(x: Tensor, size: int) -> Tensor:
    N, H, W, C = x.shape
    nH = H // size
    nW = W // size
    x = x.view(N, size, nH, size, nW, C).permute(0, 2, 4, 1, 3, 5).reshape(N, nH * nW, size * size, C)
    return x, nH, nW


def ungrid(x: Tensor, nH: int, nW: int, size: int) -> Tensor:
    N, _, _, C = x.shape
    return x.view(N, nH, nW, size, size, C).permute(0, 3, 1, 4, 2, 5).reshape(N, size * nH, size * nW, C)


class RelativeMHA(MHA):
    def __init__(self, input_size: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__(d_model, head_dim=32, dropout=dropout)
        relative_size = 2 * input_size - 1  # [-(input_size - 1), input_size - 1]
        self.attn_bias = nn.Parameter(torch.zeros(self.n_heads, relative_size, relative_size))

        index = torch.empty(input_size, input_size, dtype=torch.long)
        for i in range(input_size):
            for j in range(input_size):
                index[i][j] = j - i + input_size - 1
        self.register_buffer("bias_index", index.view(-1), persistent=False)
        self.bias_index: Tensor

    def forward(self, x: Tensor) -> Tensor:
        bias = self.attn_bias[:, self.bias_index]
        bias = bias[:, :, self.bias_index]
        return super().forward(x, attn_bias=bias)


class MaxViTBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1, window_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__()
        self.mbconv = MBConv(in_dim, out_dim, stride)
        self.block_sa = RelativeMHA(window_size, out_dim, dropout)
        self.block_mlp = MLP(out_dim, out_dim * 4, dropout)
        self.grid_sa = RelativeMHA(window_size, out_dim, dropout)
        self.grid_mlp = MLP(out_dim, out_dim * 4, dropout)
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.mbconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x, nH, nW = block(x, self.window_size)
        x = x + self.block_sa(x)
        x = unblock(x, nH, nW, self.window_size)
        x = x + self.block_mlp(x)

        x, nH, nW = grid(x, self.window_size)
        x = x + self.grid_sa(x)
        x = ungrid(x, nH, nW, self.window_size)
        x = x + self.grid_mlp(x)

        return x


class MaxViT(nn.Module):
    def __init__(self, stem_dim: int, n_blocks: list[int], dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            conv_norm_act(3, stem_dim, 3, 2),
            nn.Conv2d(stem_dim, stem_dim, 3, 1, 1),
        )
        in_dim = stem_dim

        self.blocks = nn.Sequential()
        for n_block, dim in zip(n_blocks, dims):
            block = nn.Sequential()
            for i in range(n_block):
                block.append(MaxViTBlock(in_dim, dim, stride=2 if i == 0 else 1, dropout=dropout))
                in_dim = dim
            self.blocks.append(block)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x).permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x)
        return x

    @staticmethod
    def from_google(variant: str, *, pretrained: bool = False, **kwargs) -> "MaxViT":
        # table 1
        stem_dim, n_blocks, dims = dict(
            tiny=(64, [2, 2, 5, 2], [64, 128, 256, 512]),
            small=(64, [2, 2, 5, 2], [96, 192, 384, 768]),
            base=(64, [2, 6, 14, 2], [96, 192, 384, 768]),
            large=(128, [2, 6, 14, 2], [128, 256, 512, 1024]),
            xlarge=(192, [2, 6, 14, 2], [192, 384, 768, 1536]),
        )[variant]

        m = MaxViT(stem_dim, n_blocks, dims, **kwargs)

        if pretrained:
            pass

        return m
