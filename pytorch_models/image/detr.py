# https://arxiv.org/abs/2005.12872
# https://github.com/facebookresearch/detr

from torch import Tensor, nn


class Bottleneck(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        bottleneck = out_dim // 4
        self.residual = nn.Sequential(
            nn.Conv2d(in_dim, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, bottleneck, 3, stride, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_dim, out_dim, bias=False), nn.BatchNorm2d(out_dim))
            if stride > 1 or out_dim != in_dim
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.shortcut(x) + self.residual(x))


class ResNetLayer(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, stride: int) -> None:
        super().__init__(
            Bottleneck(in_dim, out_dim, stride=stride), *[Bottleneck(out_dim, out_dim) for _ in range(n_layers - 1)]
        )


class ResNet(nn.Module):
    def __init__(self, n_layers: list[int]) -> None:
        super().__init__()
        in_dim = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_dim, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )

        self.layers = nn.Sequential()
        for i, n_layer in enumerate(n_layers):
            self.layers.append(ResNetLayer(in_dim, 256 * 2**i, n_layer, 1 if i == 0 else 2))
            in_dim = 256 * 2**i

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layers(x)
        return x
