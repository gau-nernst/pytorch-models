# https://arxiv.org/abs/2105.01601
# https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py

import numpy as np
import torch
from torch import Tensor, nn

from ..utils import torch_hub_download
from .vit import MLP, load_flax_conv2d, load_flax_linear, load_flax_ln


class MixerBlock(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        d_model: int,
        mlp_ratio: tuple[int, int] = (0.5, 4.0),
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        tokens_mlp_dim, channels_mlp_dim = [int(d_model * ratio) for ratio in mlp_ratio]
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, norm_eps)
        self.token_mixing = MLP(n_tokens, tokens_mlp_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model, norm_eps)
        self.channel_mixing = MLP(d_model, channels_mlp_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x -> (B, n_tokens, d_model)
        x = x + self.token_mixing(self.norm1(x).transpose(-1, -2)).transpose(-1, -2)
        x = x + self.channel_mixing(self.norm2(x))
        return x


class MLPMixer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        patch_size: int,
        img_size: int = 224,
        mlp_ratio: tuple[float, float] = (0.5, 4.0),
        dropout: float = 0.0,
        layernorm_eps: float = 1e-6,
    ) -> None:
        assert img_size % patch_size == 0
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        n_tokens = (img_size // patch_size) ** 2
        self.layers = nn.Sequential(
            *[MixerBlock(n_tokens, d_model, mlp_ratio, dropout, layernorm_eps) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model, layernorm_eps)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (N, C, H, W) -> (N, H*W, C)
        x = self.layers(x)
        x = self.norm(x)
        x = x.mean(1)
        return x

    @staticmethod
    def from_google(model_tag: str, *, pretrained: bool = False, **kwargs) -> "MLPMixer":
        if "_" in model_tag:
            model_tag, weights = model_tag.split("_")
        else:
            weights = "gsam"

        size, patch_size = model_tag.split("/")
        patch_size = int(patch_size)

        # Table 1 in https://arxiv.org/pdf/2105.01601.pdf
        n_layers, d_model = dict(
            S=(8, 512),
            B=(12, 768),
            L=(24, 1024),
            H=(32, 1280),
        )[size]
        m = MLPMixer(n_layers, d_model, patch_size, **kwargs)

        if pretrained:
            url = f"https://storage.googleapis.com/mixer_models/{weights}/Mixer-{size}_{patch_size}.npz"
            m.load_jax_weights(torch_hub_download(url))

        return m

    @torch.no_grad()
    def load_jax_weights(self, path: str) -> None:
        jax_weights = {k: torch.from_numpy(v) for k, v in np.load(path).items()}

        load_flax_conv2d(self.patch_embed, jax_weights, "stem")
        load_flax_ln(self.norm, jax_weights, "pre_head_layer_norm")

        for i, layer in enumerate(self.layers):
            load_flax_ln(layer.norm1, jax_weights, f"MixerBlock_{i}/LayerNorm_0")
            load_flax_linear(layer.token_mixing.linear1, jax_weights, f"MixerBlock_{i}/token_mixing/Dense_0")
            load_flax_linear(layer.token_mixing.linear2, jax_weights, f"MixerBlock_{i}/token_mixing/Dense_1")

            load_flax_ln(layer.norm2, jax_weights, f"MixerBlock_{i}/LayerNorm_1")
            load_flax_linear(layer.channel_mixing.linear1, jax_weights, f"MixerBlock_{i}/channel_mixing/Dense_0")
            load_flax_linear(layer.channel_mixing.linear2, jax_weights, f"MixerBlock_{i}/channel_mixing/Dense_1")
