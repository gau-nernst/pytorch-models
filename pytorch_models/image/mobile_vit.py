# https://arxiv.org/abs/2110.02178
# https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py

import torch
from torch import Tensor, nn

from ..transformer import Encoder


def conv_norm_act(in_dim: int, out_dim: int, kernel_size: int, stride: int = 1, groups: int = 1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.SiLU(),
    )


# from MobileNetv2
class MBConv(nn.Sequential):
    def __init__(self, in_dim: int, expansion: int, out_dim: int, stride: int = 1) -> None:
        hidden_dim = in_dim * expansion
        self.residual = (in_dim == out_dim) and (stride == 1)
        super().__init__()
        self.pw1 = conv_norm_act(in_dim, hidden_dim, 1)
        self.dw = conv_norm_act(hidden_dim, hidden_dim, 3, stride, groups=hidden_dim)
        self.pw2 = nn.Sequential(nn.Conv2d(hidden_dim, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim))

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
        # NOTE: d_model might not be divisible by 32
        self.in_conv = nn.Sequential(conv_norm_act(in_dim, in_dim, 3), nn.Conv2d(in_dim, d_model, 1, bias=False))
        self.transformer = Encoder(n_layers, d_model, n_heads=d_model // 32, mlp_ratio=2.0, act="silu")
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = conv_norm_act(d_model, in_dim, 1)
        self.out_fusion = conv_norm_act(in_dim * 2, in_dim, 3)

    def forward(self, x: Tensor) -> Tensor:
        out, n_patches = unfold(self.in_conv(x), self.patch_size)
        out = fold(self.norm(self.transformer(out)), self.patch_size, n_patches)
        return self.out_fusion(torch.cat([x, self.out_proj(out)], 1))


class MobileViT(nn.Sequential):
    def __init__(self, channels: list[int], d_models: list[int], out_dim: int, expansion: int) -> None:
        super().__init__(
            nn.Sequential(
                conv_norm_act(3, 16, 3, 2),
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
                conv_norm_act(channels[4], out_dim, 1),
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
            base_url = "https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification"
            url = f"{base_url}/mobilevit_{variant}.pt"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            m.load_apple_state_dict(state_dict)

        return m

    @torch.no_grad()
    def load_apple_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        state_dict = dict(state_dict)

        def load_weight(layer: nn.Linear | nn.Conv2d | nn.BatchNorm2d | nn.LayerNorm, prefix: str):
            layer.weight.copy_(state_dict.pop(f"{prefix}.weight"))
            if layer.bias is not None:
                layer.bias.copy_(state_dict.pop(f"{prefix}.bias"))

            if isinstance(layer, nn.BatchNorm2d):
                layer.running_mean.copy_(state_dict.pop(f"{prefix}.running_mean"))
                layer.running_var.copy_(state_dict.pop(f"{prefix}.running_var"))
                layer.num_batches_tracked.copy_(state_dict.pop(f"{prefix}.num_batches_tracked"))

        def load_conv_norm(layer, prefix: str):
            load_weight(layer[0], f"{prefix}.block.conv")
            load_weight(layer[1], f"{prefix}.block.norm")

        def load_mbconv(layer: MBConv, prefix: str):
            load_conv_norm(layer.pw1, f"{prefix}.exp_1x1")
            load_conv_norm(layer.dw, f"{prefix}.conv_3x3")
            load_conv_norm(layer.pw2, f"{prefix}.red_1x1")

        def load_transformer(layers: Encoder, prefix: str):
            for i, layer in enumerate(layers):
                load_weight(layer.sa_norm, f"{prefix}.{i}.pre_norm_mha.0")
                qw, kw, vw = state_dict.pop(f"{prefix}.{i}.pre_norm_mha.1.qkv_proj.weight").chunk(3)
                layer.sa.q_proj.weight.copy_(qw)
                layer.sa.k_proj.weight.copy_(kw)
                layer.sa.v_proj.weight.copy_(vw)
                qb, kb, vb = state_dict.pop(f"{prefix}.{i}.pre_norm_mha.1.qkv_proj.bias").chunk(3)
                layer.sa.q_proj.bias.copy_(qb)
                layer.sa.k_proj.bias.copy_(kb)
                layer.sa.v_proj.bias.copy_(vb)
                load_weight(layer.sa.out_proj, f"{prefix}.{i}.pre_norm_mha.1.out_proj")

                load_weight(layer.mlp_norm, f"{prefix}.{i}.pre_norm_ffn.0")
                load_weight(layer.mlp[0], f"{prefix}.{i}.pre_norm_ffn.1")
                load_weight(layer.mlp[2], f"{prefix}.{i}.pre_norm_ffn.4")

        def load_mobilevit_block(layer: MobileViTBlock, prefix: str):
            load_conv_norm(layer.in_conv[0], f"{prefix}.local_rep.conv_3x3")
            load_weight(layer.in_conv[1], f"{prefix}.local_rep.conv_1x1.block.conv")
            load_transformer(layer.transformer, f"{prefix}.global_rep")
            load_weight(layer.norm, f"{prefix}.global_rep.{len(layer.transformer)}")
            load_conv_norm(layer.out_proj, f"{prefix}.conv_proj")
            load_conv_norm(layer.out_fusion, f"{prefix}.fusion")

        load_conv_norm(self[0][0], "conv_1")
        load_mbconv(self[0][1], "layer_1.0.block")

        load_mbconv(self[1][0], "layer_2.0.block")
        load_mbconv(self[1][1], "layer_2.1.block")
        load_mbconv(self[1][2], "layer_2.2.block")

        load_mbconv(self[2][0], "layer_3.0.block")
        load_mobilevit_block(self[2][1], "layer_3.1")

        load_mbconv(self[3][0], "layer_4.0.block")
        load_mobilevit_block(self[3][1], "layer_4.1")

        load_mbconv(self[4][0], "layer_5.0.block")
        load_mobilevit_block(self[4][1], "layer_5.1")
        load_conv_norm(self[4][2], "conv_1x1_exp")

        state_dict.pop("classifier.fc.weight")
        state_dict.pop("classifier.fc.bias")
        assert len(state_dict) == 0
