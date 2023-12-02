# https://arxiv.org/abs/2006.11477
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py

import json

import requests
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..transformer import Encoder


class LayerNorm1d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class FeatureEncoder(nn.Sequential):
    def __init__(
        self,
        dims: tuple[int, ...],
        kernels: tuple[int, ...],
        strides: tuple[int, ...],
        bias: bool = True,
        dropout: float = 0.0,
        legacy: bool = False,
    ) -> None:
        super().__init__()
        in_dim = 1
        for i, (out_dim, kernel, stride) in enumerate(zip(dims, kernels, strides)):
            conv = nn.Conv1d(in_dim, out_dim, kernel, stride, bias=bias)  # no stride
            if legacy:
                norm = nn.InstanceNorm1d(out_dim, affine=True) if i == 0 else nn.Identity()
            else:
                norm = LayerNorm1d(out_dim)

            self.append(nn.Sequential(conv, nn.Dropout(dropout), norm, nn.GELU()))
            in_dim = out_dim


class Wav2Vec2(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        stem_dims: tuple[int, ...] = (512,) * 7,
        stem_kernels: tuple[int, ...] = (10,) + (3,) * 4 + (2,) * 2,
        stem_strides: tuple[int, ...] = (5,) + (2,) * 6,
        stem_bias: bool = True,
        stem_legacy: bool = False,
        pe_kernel: int = 128,
        pe_groups: int = 16,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.feature_encoder = FeatureEncoder(stem_dims, stem_kernels, stem_strides, stem_bias, dropout, stem_legacy)

        in_dim = stem_dims[-1]
        self.proj = nn.Sequential(nn.LayerNorm(in_dim))
        if in_dim != d_model:
            self.proj.append(nn.Linear(in_dim, d_model))

        self.pe_conv = nn.Sequential(
            nn.ConstantPad1d((pe_kernel // 2, pe_kernel // 2 - 1), 0),  # same padding for even kernel
            nn.Conv1d(d_model, d_model, pe_kernel, groups=pe_groups),
            nn.GELU(),
        )
        self.transformer = Encoder(n_layers, d_model, head_dim, True, mlp_ratio, dropout, pre_norm)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L)
        x = self.feature_encoder(x.unsqueeze(1)).transpose(1, 2)
        x = self.proj(x)

        x = x + self.pe_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.transformer(x)
        return x

    @classmethod
    def from_hf(cls, model_id: str, pretrained: bool = False):
        config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        config = json.loads(requests.get(config_url).content)

        m = cls(
            n_layers=config["num_hidden_layers"],
            d_model=config["hidden_size"],
            stem_bias=config["conv_bias"],
            stem_legacy=config.get("feat_extract_norm", "layer") == "group",
            head_dim=config["hidden_size"] // config["num_attention_heads"],
            pre_norm=config.get("do_stable_layer_norm", False),
        )

        if pretrained:
            ckpt_url = f"https://huggingface.co/{model_id}/resolve/main/pytorch_model.bin"
            state_dict = torch.hub.load_state_dict_from_url(ckpt_url, file_name=model_id.replace("/", "_"))
            state_dict = {k.replace("wav2vec2.", ""): v for k, v in state_dict.items()}
            m.load_hf_state_dict(state_dict)

        return m

    @torch.no_grad()
    def load_hf_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        def copy_w(module: nn.Conv1d | nn.Linear | nn.LayerNorm | nn.InstanceNorm1d, prefix: str):
            module.weight.copy_(state_dict.pop(f"{prefix}.weight"))
            if module.bias is not None:
                module.bias.copy_(state_dict.pop(f"{prefix}.bias"))

        for i, conv in enumerate(self.feature_encoder):
            prefix = f"feature_extractor.conv_layers.{i}"
            copy_w(conv[0], prefix + ".conv")
            if not isinstance(conv[2], nn.Identity):
                copy_w(conv[2], prefix + ".layer_norm")

        copy_w(self.proj[0], "feature_projection.layer_norm")
        if len(self.proj) > 1:
            copy_w(self.proj[1], "feature_projection.projection")

        # reverse torch.nn.utils.weight_norm()
        prefix = "encoder.pos_conv_embed.conv"
        weight_g = state_dict.pop(prefix + ".weight_g")
        weight_v = state_dict.pop(prefix + ".weight_v")
        self.pe_conv[1].weight.copy_(weight_g * F.normalize(weight_v, dim=(0, 1)))
        self.pe_conv[1].bias.copy_(state_dict.pop(prefix + ".bias"))

        copy_w(self.transformer.norm, "encoder.layer_norm")
        for i, block in enumerate(self.transformer.layers):
            prefix = f"encoder.layers.{i}"
            state_dict.pop(f"{prefix}.attention.k_proj.bias")

            copy_w(block.mha.q_proj, prefix + ".attention.q_proj")
            copy_w(block.mha.k_proj, prefix + ".attention.k_proj")
            copy_w(block.mha.v_proj, prefix + ".attention.v_proj")
            copy_w(block.mha.out_proj, prefix + ".attention.out_proj")
            copy_w(block.norm1, prefix + ".layer_norm")

            copy_w(block.mlp.linear1, prefix + ".feed_forward.intermediate_dense")
            copy_w(block.mlp.linear2, prefix + ".feed_forward.output_dense")
            copy_w(block.norm2, prefix + ".final_layer_norm")

        print(state_dict.keys())
