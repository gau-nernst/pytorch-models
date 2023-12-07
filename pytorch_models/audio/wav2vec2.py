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
    STEM_DIMS = (512,) * 7
    STEM_KERNELS = (10,) + (3,) * 4 + (2,) * 2
    STEM_STRIDES = (5,) + (2,) * 6

    PE_KERNEL = 128
    PE_GROUPS = 16

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        stem_bias: bool = True,
        stem_legacy: bool = False,
        dropout: float = 0.0,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.feature_encoder = FeatureEncoder(
            self.STEM_DIMS, self.STEM_KERNELS, self.STEM_STRIDES, stem_bias, dropout, stem_legacy
        )

        in_dim = self.STEM_DIMS[-1]
        self.proj = nn.Sequential(nn.LayerNorm(in_dim))
        if in_dim != d_model:
            self.proj.append(nn.Linear(in_dim, d_model))

        self.pe_conv = nn.Sequential(
            nn.ConstantPad1d((self.PE_KERNEL // 2, self.PE_KERNEL // 2 - 1), 0),  # same padding for even kernel
            nn.Conv1d(d_model, d_model, self.PE_KERNEL, groups=self.PE_GROUPS),
            nn.GELU(),
        )
        self.transformer = Encoder(n_layers, d_model, dropout=dropout, pre_norm=pre_norm)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L)
        x = self.feature_encoder(x.unsqueeze(1)).transpose(1, 2)
        x = self.proj(x)

        x = x + self.pe_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.transformer(x)
        return x

    @classmethod
    def from_hf(cls, model_tag: str, *, pretrained: bool = False, **kwargs):
        config_url = f"https://huggingface.co/{model_tag}/raw/main/config.json"
        config = json.loads(requests.get(config_url).content)

        assert config["hidden_size"] == config["num_attention_heads"] * 64
        _kwargs = dict(
            n_layers=config["num_hidden_layers"],
            d_model=config["hidden_size"],
            stem_bias=config["conv_bias"],
        )
        if "feat_extract_norm" in config:
            _kwargs["stem_legacy"] = config["feat_extract_norm"] == "group"
        if "do_stable_layer_norm" in config:
            _kwargs["pre_norm"] = config["do_stable_layer_norm"]

        m = cls(**_kwargs, **kwargs)

        if pretrained:
            ckpt_url = f"https://huggingface.co/{model_tag}/resolve/main/pytorch_model.bin"
            state_dict = torch.hub.load_state_dict_from_url(ckpt_url, file_name=model_tag.replace("/", "_"))
            state_dict = {k.replace("wav2vec2.", ""): v for k, v in state_dict.items()}
            m.load_hf_state_dict(state_dict)

        return m

    @torch.no_grad()
    def load_hf_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        state_dict = state_dict.copy()  # shallow copy

        def copy_w(module: nn.Conv1d | nn.Linear | nn.LayerNorm | nn.InstanceNorm1d, prefix: str):
            module.weight.copy_(state_dict.pop(f"{prefix}.weight"))
            if module.bias is not None:
                module.bias.copy_(state_dict.pop(f"{prefix}.bias"))

        for i, conv in enumerate(self.feature_encoder):
            prefix = f"feature_extractor.conv_layers.{i}"
            copy_w(conv[0], f"{prefix}.conv")
            if not isinstance(conv[2], nn.Identity):
                copy_w(conv[2], f"{prefix}.layer_norm")

        copy_w(self.proj[0], "feature_projection.layer_norm")
        if len(self.proj) > 1:
            copy_w(self.proj[1], "feature_projection.projection")

        # reverse torch.nn.utils.weight_norm()
        prefix = "encoder.pos_conv_embed.conv"
        weight_g = state_dict.pop(f"{prefix}.weight_g")
        weight_v = state_dict.pop(f"{prefix}.weight_v")
        self.pe_conv[1].weight.copy_(weight_g * F.normalize(weight_v, dim=(0, 1)))
        self.pe_conv[1].bias.copy_(state_dict.pop(f"{prefix}.bias"))

        copy_w(self.transformer.norm, "encoder.layer_norm")
        for i, block in enumerate(self.transformer.layers):
            prefix = f"encoder.layers.{i}"
            copy_w(block.sa.q_proj, f"{prefix}.attention.q_proj")
            copy_w(block.sa.k_proj, f"{prefix}.attention.k_proj")
            copy_w(block.sa.v_proj, f"{prefix}.attention.v_proj")
            copy_w(block.sa.out_proj, f"{prefix}.attention.out_proj")
            copy_w(block.sa_norm, f"{prefix}.layer_norm")

            copy_w(block.mlp.linear1, f"{prefix}.feed_forward.intermediate_dense")
            copy_w(block.mlp.linear2, f"{prefix}.feed_forward.output_dense")
            copy_w(block.mlp_norm, f"{prefix}.final_layer_norm")

        print(state_dict.keys())
