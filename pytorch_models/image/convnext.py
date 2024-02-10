# https://arxiv.org/abs/2201.03545
# https://github.com/facebookresearch/ConvNeXt

import torch
from torch import Tensor, nn


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)


class ConvNeXtBlock(nn.Sequential):
    expansion = 4

    def __init__(self, d_model: int, norm_eps: float = 1e-6, v2: bool = False) -> None:
        hidden_dim = d_model * self.expansion
        super().__init__(
            Permute(0, 3, 1, 2),
            nn.Conv2d(d_model, d_model, 7, padding=3, groups=d_model),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(d_model, norm_eps),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.gamma = nn.Parameter(torch.full((d_model,), 1e-6))

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x) * self.gamma


class ConvNeXt(nn.Sequential):
    def __init__(self, d_model: int, depths: tuple[int, ...], norm_eps: float = 1e-6, v2: bool = False) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, d_model, 4, 4), Permute(0, 2, 3, 1), nn.LayerNorm(d_model, norm_eps))

        self.stages = nn.Sequential()

        for stage_idx, depth in enumerate(depths):
            stage = nn.Sequential()
            if stage_idx > 0:
                # equivalent to PatchMerging in SwinTransformer
                downsample = nn.Sequential(
                    nn.LayerNorm(d_model, norm_eps),
                    Permute(0, 3, 1, 2),
                    nn.Conv2d(d_model, d_model * 2, 2, 2),
                    Permute(0, 2, 3, 1),
                )
                d_model *= 2
            else:
                downsample = nn.Identity()
            stage.append(downsample)

            for block_idx in range(depth):
                block = ConvNeXtBlock(d_model, norm_eps, v2)
                stage.append(block)

            self.stages.append(stage)

        self.pool = nn.Sequential(Permute(0, 3, 1, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.norm = nn.LayerNorm(d_model, norm_eps)

    @staticmethod
    def from_facebook(variant: str, *, pretrained: bool = False) -> "ConvNeXt":
        d_model, depths = dict(
            atto=(40, (2, 2, 6, 2)),
            femto=(48, (2, 2, 6, 2)),
            pico=(64, (2, 2, 6, 2)),
            nano=(80, (2, 2, 8, 2)),
            tiny=(96, (3, 3, 9, 3)),
            small=(96, (3, 3, 27, 3)),
            base=(128, (3, 3, 27, 3)),
            large=(192, (3, 3, 27, 3)),
            xlarge=(256, (3, 3, 27, 3)),
            huge=(352, (3, 3, 27, 3)),
        )[variant]
        m = ConvNeXt(d_model, depths)

        if pretrained:
            url = f"https://dl.fbaipublicfiles.com/convnext/convnext_{variant}_22k_224.pth"
            state_dict = torch.hub.load_state_dict_from_url(url)["model"]
            m.load_facebook_state_dict(state_dict)

        return m

    @torch.no_grad()
    def load_facebook_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        state_dict = dict(state_dict)

        def copy_(m: nn.Conv2d | nn.Linear | nn.LayerNorm, prefix: str):
            m.weight.copy_(state_dict.pop(f"{prefix}.weight"))
            m.bias.copy_(state_dict.pop(f"{prefix}.bias"))

        copy_(self.stem[0], "downsample_layers.0.0")
        copy_(self.stem[2], "downsample_layers.0.1")

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx > 0:
                copy_(stage[0][0], f"downsample_layers.{stage_idx}.0")
                copy_(stage[0][2], f"downsample_layers.{stage_idx}.1")

            for block_idx in range(1, len(stage)):
                block: ConvNeXtBlock = stage[block_idx]
                prefix = f"stages.{stage_idx}.{block_idx - 1}"

                copy_(block[1], f"{prefix}.dwconv")
                copy_(block[3], f"{prefix }.norm")
                copy_(block[4], f"{prefix}.pwconv1")
                copy_(block[6], f"{prefix}.pwconv2")
                block.gamma.copy_(state_dict.pop(f"{prefix}.gamma"))

        copy_(self.norm, "norm")
