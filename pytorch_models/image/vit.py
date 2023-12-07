# ViT: https://arxiv.org/abs/2010.11929
# AugReg: https://arxiv.org/abs/2106.10270
# DeiT-3: https://arxiv.org/abs/2204.07118
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
        self, d_model: int, n_heads: int, bias: bool = True, mlp_ratio: float = 4.0, norm_eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = MHA(d_model, n_heads=n_heads, bias=bias)
        self.norm = nn.LayerNorm(d_model, norm_eps)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(self.probe, x).squeeze(1)
        x = x + self.mlp(self.norm(x))
        return x


# NOTE: layer scale and stochastic depth are not supported
# TODO: support non-square input
class ViT(nn.Module):
    norm_eps = 1e-6

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        patch_size: int,
        img_size: int = 224,
        cls_token: bool = True,
        pool_type: str = "cls_token",
        dropout: float = 0.0,
    ) -> None:
        assert img_size % patch_size == 0
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if cls_token else None
        self.pe = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, d_model))

        self.layers = Encoder(n_layers, d_model, n_heads=n_heads, dropout=dropout, norm_eps=self.norm_eps)
        self.norm = nn.LayerNorm(d_model)

        self.pooler = dict(
            cls_token=ClassTokenPooling,
            gap=GlobalAveragePooling,
            mha=partial(MHAPooling, d_model, n_heads, norm_eps=self.norm_eps),
        )[pool_type]()

    def forward(self, imgs: Tensor) -> Tensor:
        out = self.patch_embed(imgs).flatten(2).transpose(1, 2)  # (N, C, H, W) -> (N, H*W, C)
        out = out + self.pe
        if self.cls_token is not None:
            out = torch.cat([self.cls_token, out], 1)
        out = self.layers(out)
        out = self.norm(out)
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
    def from_google(model_tag: str, *, pretrained: bool = False, **kwargs) -> "ViT":
        if "_" in model_tag:
            model_tag, weights = model_tag.split("_")
        else:
            weights = "augreg"

        size, patch_size = model_tag.split("/")
        patch_size = int(patch_size)

        n_layers, d_model, n_heads = dict(
            Ti=(12, 192, 3),
            S=(12, 384, 6),
            M=(12, 512, 8),
            B=(12, 768, 12),
            L=(24, 1024, 16),
            H=(32, 1280, 16),
        )[size]

        _kwargs = dict()
        if weights == "siglip":
            _kwargs.update(cls_token=False, pool_type="mha")

        m = ViT(n_layers, d_model, n_heads, patch_size, **_kwargs, **kwargs)

        # TODO: support pe resizing
        if pretrained:
            if weights == "augreg":
                ckpt = {
                    "Ti/16": "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
                    "S/32": "S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz",
                    "S/16": "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
                    "B/32": "B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
                    "B/16": "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
                    "L/16": "L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz",
                }[model_tag]
                m.load_flax_ckpt(f"augreg/{ckpt}")

            elif weights == "siglip":
                img_size = kwargs.get("img_size", 224)
                ckpt = {
                    ("B/16", 224): "webli_en_b16_224_63724782.npz",
                    ("B/16", 256): "webli_en_b16_256_60500360.npz",
                    ("B/16", 384): "webli_en_b16_384_68578854.npz",
                    ("B/16", 512): "webli_en_b16_512_68580893.npz",
                    ("L/16", 256): "webli_en_l16_256_60552751.npz",
                    ("L/16", 384): "webli_en_l16_384_63634585.npz",
                }[(model_tag, img_size)]
                m.load_flax_ckpt(f"siglip/{ckpt}", big_vision=True, prefix="params/img/")

            else:
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
        load_flax_ln(self.norm, jax_weights, "Transformer/encoder_norm")

        for i, layer in enumerate(self.layers):
            load_flax_ln(layer.sa_norm, jax_weights, f"Transformer/encoderblock_{i}/{mha_norm}")
            load_flax_mha(layer.sa, jax_weights, f"Transformer/encoderblock_{i}/{mha}")

            load_flax_ln(layer.mlp_norm, jax_weights, f"Transformer/encoderblock_{i}/{mlp_norm}")
            load_flax_linear(layer.mlp.linear1, jax_weights, f"Transformer/encoderblock_{i}/{mlp}/Dense_0")
            load_flax_linear(layer.mlp.linear2, jax_weights, f"Transformer/encoderblock_{i}/{mlp}/Dense_1")

        # big_vision only
        if isinstance(self.pooler, MHAPooling):
            self.pooler.probe.copy_(jax_weights.pop("MAPHead_0/probe"))
            load_flax_mha(self.pooler.attn, jax_weights, "MAPHead_0/MultiHeadDotProductAttention_0")
            load_flax_ln(self.pooler.norm, jax_weights, "MAPHead_0/LayerNorm_0")
            load_flax_linear(self.pooler.mlp.linear1, jax_weights, "MAPHead_0/MlpBlock_0/Dense_0")
            load_flax_linear(self.pooler.mlp.linear2, jax_weights, "MAPHead_0/MlpBlock_0/Dense_1")

        if len(jax_weights) > 0:
            print(jax_weights.keys())

    @staticmethod
    def from_deit3(model_tag: str, *, pretrained: bool = False, **kwargs) -> "ViT":
        size, patch_size = model_tag.split("/")
        patch_size = int(patch_size)

        n_layers, d_model, n_heads = dict(
            Ti=(12, 192, 3),
            S=(12, 384, 6),
            M=(12, 512, 8),
            B=(12, 768, 12),
            L=(24, 1024, 16),
            H=(32, 1280, 16),
        )[size]
        m = ViT(n_layers, d_model, n_heads, patch_size, **kwargs)

        # TODO: support patch_size resizing
        if pretrained:
            assert patch_size == 16
            img_size = kwargs.get("img_size", 224)
            _size = dict(S="small", M="medium", B="base", L="large", H="huge")[size]

            url = f"https://dl.fbaipublicfiles.com/deit/deit_3_{_size}_{img_size}_21k.pth"
            state_dict = torch.hub.load_state_dict_from_url(url)["model"]
            m.load_deit3_state_dict(state_dict)

        return m

    @torch.no_grad()
    def load_deit3_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        state_dict = state_dict.copy()  # shallow clone

        def copy_(m: nn.Linear | nn.LayerNorm, prefix: str):
            m.weight.copy_(state_dict.pop(prefix + ".weight").view(m.weight.shape))
            m.bias.copy_(state_dict.pop(prefix + ".bias"))

        copy_(self.patch_embed, "patch_embed.proj")
        pe = state_dict.pop("pos_embed")
        self.pe.copy_(pe[:, -self.pe.shape[1] :])

        self.cls_token.copy_(state_dict.pop("cls_token"))
        if pe.shape[1] > self.pe.shape[1]:
            self.cls_token.add_(pe[:, 0])

        copy_(self.norm, "norm")
        for i, layer in enumerate(self.layers):
            prefix = f"blocks.{i}"
            copy_(layer.sa_norm, f"{prefix}.norm1")
            copy_(layer.mlp_norm, f"{prefix}.norm2")

            q_w, k_w, v_w = state_dict.pop(f"{prefix}.attn.qkv.weight").chunk(3, 0)
            layer.sa.q_proj.weight.copy_(q_w)
            layer.sa.k_proj.weight.copy_(k_w)
            layer.sa.v_proj.weight.copy_(v_w)

            q_b, k_b, v_b = state_dict.pop(f"{prefix}.attn.qkv.bias").chunk(3, 0)
            layer.sa.q_proj.bias.copy_(q_b)
            layer.sa.k_proj.bias.copy_(k_b)
            layer.sa.v_proj.bias.copy_(v_b)

            copy_(layer.sa.out_proj, f"{prefix}.attn.proj")
            scale = state_dict.pop(f"{prefix}.gamma_1")
            layer.sa.out_proj.weight.mul_(scale.view(-1, 1))
            layer.sa.out_proj.bias.mul_(scale)

            copy_(layer.mlp.linear1, f"{prefix}.mlp.fc1")
            copy_(layer.mlp.linear2, f"{prefix}.mlp.fc2")
            scale = state_dict.pop(f"{prefix}.gamma_2")
            layer.mlp.linear2.weight.mul_(scale.view(-1, 1))
            layer.mlp.linear2.bias.mul_(scale)

        print(state_dict.keys())


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
