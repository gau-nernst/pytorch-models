# https://arxiv.org/abs/2109.06870
# https://github.com/asappresearch/sew

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .wav2vec2 import Wav2Vec2


class SEW(Wav2Vec2):
    STEM_DIMS = (64,) + (128,) * 4 + (256,) * 4 + (512,) * 4
    STEM_KERNELS = (10,) + (3, 1) * 4 + (2, 1) * 2
    STEM_STRIDES = (5,) + (2, 1) * 6

    PE_KERNEL = 31

    def __init__(
        self, n_layers: int, d_model: int, stem_bias: bool = True, stem_legacy: bool = True, dropout: float = 0.0
    ) -> None:
        assert stem_legacy
        super().__init__(n_layers, d_model, stem_bias, stem_legacy, dropout, False)
        self.pe_conv[1].stride = 2
        self.upsample = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L)
        x = self.feature_encoder(x.unsqueeze(1)).transpose(1, 2)
        x = self.proj(x)

        T = x.shape[1]
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, 2) + self.pe_conv(x)
        x = self.transformer(x.transpose(1, 2))
        x = self.upsample(x).unflatten(2, (2, -1)).flatten(1, 2)
        if x.shape[1] < T:
            x = F.pad(x, (0, 0, 0, T - x.shape[1]))
        return x

    @torch.no_grad()
    def load_hf_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        state_dict = state_dict.copy()  # shallow copy

        def copy_w(module: nn.Conv1d | nn.Linear | nn.LayerNorm | nn.BatchNorm1d, prefix: str):
            module.weight.copy_(state_dict.pop(f"{prefix}.weight"))
            if module.bias is not None:
                module.bias.copy_(state_dict.pop(f"{prefix}.bias"))

        for i, conv in enumerate(self.feature_encoder):
            prefix = f"feature_extractor.conv_layers.{i}"
            copy_w(conv[0], f"{prefix}.conv")
            if not isinstance(conv[2], nn.Identity):
                copy_w(conv[2], f"{prefix}.layer_norm")

        copy_w(self.proj[0], "layer_norm")
        if len(self.proj) > 1:
            copy_w(self.proj[1], "feature_projection")

        # reverse torch.nn.utils.weight_norm()
        prefix = "encoder.pos_conv_embed.conv"
        weight_g = state_dict.pop(f"{prefix}.weight_g")
        weight_v = state_dict.pop(f"{prefix}.weight_v")
        self.pe_conv[1].weight.copy_(weight_g * F.normalize(weight_v, dim=(0, 1)))
        self.pe_conv[1].bias.copy_(state_dict.pop(f"{prefix}.bias"))

        copy_w(self.transformer.norm, "encoder.layer_norm")
        for i, block in enumerate(self.transformer.layers):
            prefix = f"encoder.layers.{i}"
            state_dict.pop(f"{prefix}.attention.k_proj.bias")

            copy_w(block.sa.q_proj, f"{prefix}.attention.q_proj")
            copy_w(block.sa.k_proj, f"{prefix}.attention.k_proj")
            copy_w(block.sa.v_proj, f"{prefix}.attention.v_proj")
            copy_w(block.sa.out_proj, f"{prefix}.attention.out_proj")
            copy_w(block.sa_norm, f"{prefix}.layer_norm")

            copy_w(block.mlp.linear1, f"{prefix}.feed_forward.intermediate_dense")
            copy_w(block.mlp.linear2, f"{prefix}.feed_forward.output_dense")
            copy_w(block.mlp_norm, f"{prefix}.final_layer_norm")

        copy_w(self.upsample[0], "encoder.upsample.projection")
        print(state_dict.keys())
