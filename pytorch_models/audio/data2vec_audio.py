# https://arxiv.org/abs/2202.03555
# https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/data2vec_audio.py

import torch
from torch import Tensor, nn

from ..transformer import Encoder
from .wav2vec2 import FeatureEncoder, LayerNorm1d, Wav2Vec2


class Data2VecAudio(Wav2Vec2):
    PE_KERNEL = 19

    def __init__(self, n_layers: int, d_model: int, stem_bias: bool = False, dropout: float = 0.0) -> None:
        nn.Module.__init__(self)
        self.feature_encoder = FeatureEncoder(self.STEM_DIMS, self.STEM_KERNELS, self.STEM_STRIDES, stem_bias, dropout)

        in_dim = self.STEM_DIMS[-1]
        self.proj = nn.Sequential(nn.LayerNorm(in_dim))
        if in_dim != d_model:
            self.proj.append(nn.Linear(in_dim, d_model))

        self.pe_conv = nn.Sequential()
        for _ in range(5):
            layer = nn.Sequential(
                nn.Conv1d(d_model, d_model, self.PE_KERNEL, padding=self.PE_KERNEL // 2, groups=self.PE_GROUPS),
                LayerNorm1d(d_model, elementwise_affine=False),
                nn.GELU(),
            )
            self.pe_conv.append(layer)

        self.transformer = Encoder(n_layers, d_model, dropout=dropout, pre_norm=False)
        self.norm = nn.LayerNorm(d_model)
        self.pre_norm = False

    @torch.no_grad()
    def load_hf_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        state_dict = state_dict.copy()  # shallow copy

        def copy_w(module: nn.Conv1d | nn.Linear | nn.LayerNorm, prefix: str):
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

        for i, layer in enumerate(self.pe_conv):
            copy_w(layer[0], f"encoder.pos_conv_embed.layers.{i}.conv")

        copy_w(self.norm, "encoder.layer_norm")
        for i, layer in enumerate(self.layers):
            prefix = f"encoder.layers.{i}"
            copy_w(layer.sa.q_proj, f"{prefix}.attention.q_proj")
            copy_w(layer.sa.k_proj, f"{prefix}.attention.k_proj")
            copy_w(layer.sa.v_proj, f"{prefix}.attention.v_proj")
            copy_w(layer.sa.out_proj, f"{prefix}.attention.out_proj")
            copy_w(layer.sa_norm, f"{prefix}.layer_norm")

            copy_w(layer.mlp.linear1, f"{prefix}.feed_forward.intermediate_dense")
            copy_w(layer.mlp.linear2, f"{prefix}.feed_forward.output_dense")
            copy_w(layer.mlp_norm, f"{prefix}.final_layer_norm")

        print(state_dict.keys())
