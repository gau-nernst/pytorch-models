# https://arxiv.org/abs/2202.03555
# https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/data2vec_audio.py

import torch
from torch import Tensor, nn

from ..transformer import Encoder
from .wav2vec2 import FeatureEncoder, LayerNorm1d, Wav2Vec2


class Data2VecAudio(Wav2Vec2):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        stem_dims: tuple[int, ...] = (512,) * 7,
        stem_kernels: tuple[int, ...] = (10,) + (3,) * 4 + (2,) * 2,
        stem_strides: tuple[int, ...] = (5,) + (2,) * 6,
        stem_bias: bool = False,
        stem_legacy: bool = False,
        pe_kernel: int = 19,
        pe_groups: int = 16,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pre_norm: bool = False,
    ) -> None:
        assert not stem_legacy
        assert not pre_norm
        nn.Module.__init__(self)
        self.feature_encoder = FeatureEncoder(stem_dims, stem_kernels, stem_strides, stem_bias, dropout, stem_legacy)

        in_dim = stem_dims[-1]
        self.proj = nn.Sequential(nn.LayerNorm(in_dim))
        if in_dim != d_model:
            self.proj.append(nn.Linear(in_dim, d_model))

        self.pe_conv = nn.Sequential()
        for _ in range(5):
            layer = nn.Sequential(
                nn.Conv1d(d_model, d_model, pe_kernel, padding=pe_kernel // 2, groups=pe_groups),
                LayerNorm1d(d_model, elementwise_affine=False),
                nn.GELU(),
            )
            self.pe_conv.append(layer)

        self.transformer = Encoder(n_layers, d_model, head_dim, True, mlp_ratio, dropout, pre_norm)

    @torch.no_grad()
    def load_hf_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        def copy_w(module: nn.Conv1d | nn.Linear | nn.LayerNorm, prefix: str):
            module.weight.copy_(state_dict.pop(prefix + ".weight"))
            if module.bias is not None:
                module.bias.copy_(state_dict.pop(prefix + ".bias"))

        for i, conv in enumerate(self.feature_encoder):
            prefix = f"feature_extractor.conv_layers.{i}"
            copy_w(conv[0], prefix + ".conv")
            if not isinstance(conv[2], nn.Identity):
                copy_w(conv[2], prefix + ".layer_norm")

        copy_w(self.proj[0], "feature_projection.layer_norm")
        if len(self.proj) > 1:
            copy_w(self.proj[1], "feature_projection.projection")

        for i, layer in enumerate(self.pe_conv):
            prefix = f"encoder.pos_conv_embed.layers.{i}"
            copy_w(layer[0], prefix + ".conv")

        copy_w(self.transformer.norm, "encoder.layer_norm")
        for i, block in enumerate(self.transformer.layers):
            prefix = f"encoder.layers.{i}"
            copy_w(block.mha.q_proj, prefix + ".attention.q_proj")
            copy_w(block.mha.k_proj, prefix + ".attention.k_proj")
            copy_w(block.mha.v_proj, prefix + ".attention.v_proj")
            copy_w(block.mha.out_proj, prefix + ".attention.out_proj")
            copy_w(block.norm1, prefix + ".layer_norm")
            copy_w(block.mlp.linear1, prefix + ".feed_forward.intermediate_dense")
            copy_w(block.mlp.linear2, prefix + ".feed_forward.output_dense")
            copy_w(block.norm2, prefix + ".final_layer_norm")

        print(state_dict.keys())
