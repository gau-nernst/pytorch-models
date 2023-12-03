# ViT: https://arxiv.org/abs/2010.11929
# AugReg: https://arxiv.org/abs/2106.10270
# https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..transformer import MHA, MLP, Encoder
from ..utils import torch_hub_download


class ClassTokenPooling(nn.Module):
    def forward(self, x: Tensor):
        return x[:, 0]


class GlobalAveragePooling(nn.Module):
    def forward(self, x: Tensor):
        return x.mean(1)


class MHAPooling(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = True, mlp_ratio: float = 4.0, layernorm_eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mha = MHA(d_model, n_heads=n_heads, bias=bias)
        self.norm = nn.LayerNorm(d_model, layernorm_eps)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        x = self.mha(self.probe, x).squeeze(1)
        x = x + self.mlp(self.norm(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        patch_size: int,
        img_size: int,
        cls_token: bool = True,
        pool_type: str = "cls_token",
        bias: bool = True,
        dropout: float = 0.0,
        layernorm_eps: float = 1e-6,
    ) -> None:
        assert img_size % patch_size == 0
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if cls_token else None
        self.pe = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, d_model))

        self.encoder = Encoder(
            n_layers, d_model, n_heads=n_heads, bias=bias, dropout=dropout, layernorm_eps=layernorm_eps
        )

        self.pooler = dict(
            cls_token=ClassTokenPooling,
            gap=GlobalAveragePooling,
            mha=partial(MHAPooling, d_model, n_heads, bias, layernorm_eps=layernorm_eps),
        )[pool_type]()

    def forward(self, imgs: Tensor) -> Tensor:
        out = self.patch_embed(imgs).flatten(2).transpose(1, 2)  # (N, C, H, W) -> (N, H*W, C)
        out = out + self.pe
        if self.cls_token is not None:
            out = torch.cat([self.cls_token, out], 1)
        out = self.encoder(out)
        out = self.pooler(out)
        return out

    @torch.no_grad()
    def resize_pe(self, size: int, interpolation_mode: str = "bicubic") -> None:
        old_size = int(self.pe.shape[1] ** 0.5)
        new_size = size // self.patch_embed.weight.shape[2]
        pe = self.pe.unflatten(1, (old_size, old_size)).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, (new_size, new_size), mode=interpolation_mode)
        pe = pe.permute(0, 2, 3, 1).flatten(1, 2)
        self.pe = nn.Parameter(pe)

    @staticmethod
    def from_google(variant: str, img_size: int, *, weights: str | None = None) -> "ViT":
        variant, patch_size = variant.split("/")

        n_layers, d_model, n_heads = dict(
            Ti=(12, 192, 3),
            S=(12, 384, 6),
            M=(12, 512, 8),
            B=(12, 768, 12),
            L=(24, 1024, 16),
            H=(32, 1280, 16),
        )[variant]
        patch_size = int(patch_size)
        kwargs = dict()
        if weights == "siglip":
            kwargs.update(cls_token=False, pool_type="mha")

        m = ViT(n_layers, d_model, n_heads, patch_size, img_size, **kwargs)

        if weights == "augreg":
            assert img_size == 224
            ckpt = {
                ("Ti", 16): "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
                ("S", 32): "S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz",
                ("S", 16): "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
                ("B", 32): "B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
                ("B", 16): "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
                ("L", 16): "L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz",
            }[(variant, patch_size)]
            m.load_flax_ckpt(f"augreg/{ckpt}")

        elif weights == "siglip":
            ckpt = {
                ("B", 16, 224): "webli_en_b16_224_63724782.npz",
                ("B", 16, 256): "webli_en_b16_256_60500360.npz",
                ("B", 16, 384): "webli_en_b16_384_68578854.npz",
                ("B", 16, 512): "webli_en_b16_512_68580893.npz",
                ("L", 16, 256): "webli_en_l16_256_60552751.npz",
                ("L", 16, 384): "webli_en_l16_384_63634585.npz",
            }[(variant, patch_size, img_size)]
            m.load_flax_ckpt(f"siglip/{ckpt}", big_vision=True, prefix="params/img/")

        elif not weights is None:
            raise ValueError(f"Unsupported weights={weights}")

        return m

    @torch.no_grad()
    def load_flax_ckpt(self, ckpt: str, *, big_vision: bool = False, prefix: str = "") -> None:
        if big_vision:
            # https://github.com/google-research/big_vision
            gcs_bucket = "big_vision"
            mha_norm = "LayerNorm_0"
            mha = "MultiHeadDotProductAttention_0"
            mlp_norm = "LayerNorm_1"
            mlp = "MlpBlock_0"

        else:
            # https://github.com/google-research/vision_transformer
            gcs_bucket = "vit_models"
            mha_norm = "LayerNorm_0"
            mha = "MultiHeadDotProductAttention_1"
            mlp_norm = "LayerNorm_2"
            mlp = "MlpBlock_3"

        path = torch_hub_download(f"https://storage.googleapis.com/{gcs_bucket}/{ckpt}")
        jax_weights = {k[len(prefix) :]: torch.from_numpy(v) for k, v in np.load(path).items() if k.startswith(prefix)}

        if self.cls_token is not None:
            self.cls_token.copy_(jax_weights.pop("cls"))
        if big_vision:
            self.pe.copy_(jax_weights.pop("pos_embedding"))
        else:
            pe = jax_weights.pop("Transformer/posembed_input/pos_embedding")
            self.cls_token.add_(pe[:, 0])
            self.pe.copy_(pe[:, 1:])
        load_flax_conv2d(self.patch_embed, jax_weights, "embedding")
        load_flax_ln(self.encoder.norm, jax_weights, "Transformer/encoder_norm")

        for i, layer in enumerate(self.encoder.layers):
            load_flax_ln(layer.norm1, jax_weights, f"Transformer/encoderblock_{i}/{mha_norm}")
            load_flax_mha(layer.mha, jax_weights, f"Transformer/encoderblock_{i}/{mha}")

            load_flax_ln(layer.norm2, jax_weights, f"Transformer/encoderblock_{i}/{mlp_norm}")
            load_flax_linear(layer.mlp.linear1, jax_weights, f"Transformer/encoderblock_{i}/{mlp}/Dense_0")
            load_flax_linear(layer.mlp.linear2, jax_weights, f"Transformer/encoderblock_{i}/{mlp}/Dense_1")

        # big_vision only
        if isinstance(self.pooler, MHAPooling):
            self.pooler.probe.copy_(jax_weights.pop("MAPHead_0/probe"))
            load_flax_mha(self.pooler.mha, jax_weights, "MAPHead_0/MultiHeadDotProductAttention_0")
            load_flax_ln(self.pooler.norm, jax_weights, "MAPHead_0/LayerNorm_0")
            load_flax_linear(self.pooler.mlp.linear1, jax_weights, "MAPHead_0/MlpBlock_0/Dense_0")
            load_flax_linear(self.pooler.mlp.linear2, jax_weights, "MAPHead_0/MlpBlock_0/Dense_1")

        if len(jax_weights) > 0:
            print(jax_weights.keys())


def load_flax_ln(norm: nn.LayerNorm, weights: dict[str, Tensor], prefix: str) -> None:
    norm.weight.copy_(weights.pop(f"{prefix}/scale"))
    norm.bias.copy_(weights.pop(f"{prefix}/bias"))


def load_flax_linear(linear: nn.Linear, weights: dict[str, Tensor], prefix: str) -> None:
    d0, d1 = linear.weight.shape
    linear.weight.copy_(weights.pop(f"{prefix}/kernel").view(d1, d0).T)
    if linear.bias is not None:
        linear.bias.copy_(weights.pop(f"{prefix}/bias").flatten())


def load_flax_conv2d(conv2d: nn.Conv2d, weights: dict[str, Tensor], prefix: str) -> None:
    conv2d.weight.copy_(weights.pop(f"{prefix}/kernel").permute(3, 2, 0, 1))
    if conv2d.bias is not None:
        conv2d.bias.copy_(weights.pop(f"{prefix}/bias"))


def load_flax_mha_in_proj(linear: nn.Linear, weights: dict[str, Tensor], prefix: str) -> None:
    linear.weight.copy_(weights.pop(f"{prefix}/kernel"))


def load_flax_mha(mha: MHA, weights: dict[str, Tensor], prefix: str) -> None:
    load_flax_linear(mha.q_proj, weights, f"{prefix}/query")
    load_flax_linear(mha.k_proj, weights, f"{prefix}/key")
    load_flax_linear(mha.v_proj, weights, f"{prefix}/value")
    load_flax_linear(mha.out_proj, weights, f"{prefix}/out")
