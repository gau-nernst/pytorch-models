# https://arxiv.org/pdf/2204.01697.pdf
# https://github.com/google-research/maxvit

import torch
from torch import Tensor, nn

from ..transformer import MHA, MLP
from ..utils import torch_hub_download


def conv_norm_act(in_dim: int, out_dim: int, kernel_size: int, stride: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_dim, eps=1e-3, momentum=0.01),
        nn.GELU(),
    )


class SqueezeExcitation(nn.Sequential):
    def __init__(self, dim: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 16, 1),
            nn.SiLU(),
            nn.Conv2d(dim // 16, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * super().forward(x)


# pre-norm MBConv
# NOTE: we don't include stochastic depth
class MBConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        hidden_dim = out_dim * 4
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            conv_norm_act(in_dim, hidden_dim, 1),
            conv_norm_act(hidden_dim, hidden_dim, 3, stride, hidden_dim),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, out_dim, 1),
        )

        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut.append(nn.AvgPool2d(stride))
        if out_dim != in_dim:
            self.shortcut.append(nn.Conv2d(in_dim, out_dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.residual(x)


def block(x: Tensor, size: int) -> Tensor:
    N, H, W, C = x.shape
    nH = H // size
    nW = W // size
    x = x.view(N, nH, size, nW, size, C).transpose(2, 3).reshape(N, nH * nW, size * size, C)
    return x, nH, nW


def unblock(x: Tensor, nH: int, nW: int, size: int) -> Tensor:
    N, _, _, C = x.shape
    return x.view(N, nH, nW, size, size, C).transpose(2, 3).reshape(N, nH * size, nW * size, C)


# similar to MobileViT's unfold
def grid(x: Tensor, size: int) -> Tensor:
    N, H, W, C = x.shape
    nH = H // size
    nW = W // size
    x = x.view(N, size, nH, size, nW, C).permute(0, 2, 4, 1, 3, 5).reshape(N, nH * nW, size * size, C)
    return x, nH, nW


def ungrid(x: Tensor, nH: int, nW: int, size: int) -> Tensor:
    N, _, _, C = x.shape
    return x.view(N, nH, nW, size, size, C).permute(0, 3, 1, 4, 2, 5).reshape(N, size * nH, size * nW, C)


class RelativeMHA(MHA):
    def __init__(self, input_size: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__(d_model, head_dim=32, dropout=dropout)
        relative_size = 2 * input_size - 1  # [-(input_size - 1), input_size - 1]
        self.attn_bias = nn.Parameter(torch.zeros(self.n_heads, relative_size, relative_size))

        index = torch.empty(input_size, input_size, dtype=torch.long)
        for i in range(input_size):
            for j in range(input_size):
                index[i][j] = j - i + input_size - 1
        self.register_buffer("bias_index", index.view(-1), persistent=False)
        self.bias_index: Tensor

    def forward(self, x: Tensor) -> Tensor:
        bias = self.attn_bias[:, self.bias_index]
        bias = bias[:, :, self.bias_index]
        return super().forward(x, attn_bias=bias)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, window_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sa_norm = nn.LayerNorm(d_model, 1e-5)
        self.sa = RelativeMHA(window_size, d_model, dropout)
        self.mlp_norm = nn.LayerNorm(d_model, 1e-5)
        self.mlp = MLP(d_model, d_model * 4, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.sa(self.sa_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MaxViTBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1, window_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__()
        self.mbconv = MBConv(in_dim, out_dim, stride)
        self.block_layer = EncoderLayer(out_dim, window_size, dropout)
        self.grid_layer = EncoderLayer(out_dim, window_size, dropout)
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.mbconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x, nH, nW = block(x, self.window_size)
        x = self.block_layer(x)
        x = unblock(x, nH, nW, self.window_size)

        x, nH, nW = grid(x, self.window_size)
        x = self.grid_layer(x)
        x = ungrid(x, nH, nW, self.window_size)

        return x


class MaxViT(nn.Module):
    def __init__(self, stem_dim: int, n_blocks: list[int], dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            conv_norm_act(3, stem_dim, 3, 2),
            nn.Conv2d(stem_dim, stem_dim, 3, 1, 1),
        )
        in_dim = stem_dim

        self.stages = nn.Sequential()
        for n_block, dim in zip(n_blocks, dims):
            stage = nn.Sequential()
            for i in range(n_block):
                stage.append(MaxViTBlock(in_dim, dim, stride=2 if i == 0 else 1, dropout=dropout))
                in_dim = dim
            self.stages.append(stage)

        self.norm = nn.LayerNorm(in_dim, 1e-5)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x).permute(0, 2, 3, 1)
        for stage in self.stages:
            x = stage(x)
        return self.norm(x.mean((1, 2)))

    @staticmethod
    def from_google(variant: str, *, pretrained: bool = False, **kwargs) -> "MaxViT":
        # table 1
        stem_dim, n_blocks, dims = dict(
            tiny=(64, [2, 2, 5, 2], [64, 128, 256, 512]),
            small=(64, [2, 2, 5, 2], [96, 192, 384, 768]),
            base=(64, [2, 6, 14, 2], [96, 192, 384, 768]),
            large=(128, [2, 6, 14, 2], [128, 256, 512, 1024]),
            xlarge=(192, [2, 6, 14, 2], [192, 384, 768, 1536]),
        )[variant]

        m = MaxViT(stem_dim, n_blocks, dims, **kwargs)

        if pretrained:
            import tensorflow as tf

            if variant in ("tiny", "small"):
                ds = "i1k"
                step = 92002
            else:
                ds = "i21k_pt"
                step = 279498

            url = f"https://storage.googleapis.com/gresearch/maxvit/ckpts/maxvit{variant}/{ds}/224/model.ckpt-{step}"
            torch_hub_download(f"{url}.data-00000-of-00001", f"maxvit_{variant}")
            ckpt_path = torch_hub_download(f"{url}.index", f"maxvit_{variant}").removesuffix(".index")

            reader = tf.train.load_checkpoint(ckpt_path)
            m.load_google_state_dict(reader)

        return m

    @torch.no_grad()
    def load_google_state_dict(self, reader) -> None:
        keys = set(
            x
            for x in reader.get_variable_to_shape_map().keys()
            if not x.endswith(("ExponentialMovingAverage", "adam_m", "adam_v"))
        )

        def get_param(name: str):
            name = f"maxvit/{name}"
            keys.remove(name)
            return torch.from_numpy(reader.get_tensor(name))

        def load_conv2d(module: nn.Conv2d, prefix: str, depthwise: bool = False):
            if depthwise:
                module.weight.copy_(get_param(f"{prefix}/depthwise_kernel").permute(2, 3, 0, 1))
            else:
                module.weight.copy_(get_param(f"{prefix}/kernel").permute(3, 2, 0, 1))
            if module.bias is not None:
                module.bias.copy_(get_param(f"{prefix}/bias"))

        def load_linear(module: nn.Linear, prefix: str, flatten: int | None = None):
            weight = get_param(f"{prefix}/weight")
            if flatten is not None:
                weight = weight.flatten(flatten, flatten + 1)
            module.weight.copy_(weight.T)
            module.bias.copy_(get_param(f"{prefix}/bias").flatten())

        def load_norm(module: nn.LayerNorm | nn.BatchNorm2d, prefix: str):
            module.weight.copy_(get_param(f"{prefix}/gamma"))
            module.bias.copy_(get_param(f"{prefix}/beta"))

            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.copy_(get_param(f"{prefix}/moving_mean"))
                module.running_var.copy_(get_param(f"{prefix}/moving_variance"))

        load_conv2d(self.stem[0][0], "stem/conv_0")
        load_norm(self.stem[0][1], "stem/norm_0")
        load_conv2d(self.stem[1], "stem/conv_1")

        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block in enumerate(stage):
                prefix = f"block_{stage_idx:02d}_{block_idx:02d}"

                load_norm(block.mbconv.residual[0], f"{prefix}/mbconv/pre_norm")
                load_conv2d(block.mbconv.residual[1][0], f"{prefix}/mbconv/expand_conv")
                load_norm(block.mbconv.residual[1][1], f"{prefix}/mbconv/expand_norm")
                load_conv2d(block.mbconv.residual[2][0], f"{prefix}/mbconv/depthwise_conv", depthwise=True)
                load_norm(block.mbconv.residual[2][1], f"{prefix}/mbconv/depthwise_norm")
                load_conv2d(block.mbconv.residual[3][1], f"{prefix}/mbconv/se/reduce_conv2d")
                load_conv2d(block.mbconv.residual[3][3], f"{prefix}/mbconv/se/expand_conv2d")
                load_conv2d(block.mbconv.residual[4], f"{prefix}/mbconv/shrink_conv")
                if len(block.mbconv.shortcut) == 2:
                    load_conv2d(block.mbconv.shortcut[1], f"{prefix}/mbconv/shortcut_conv")

                for layer, suffix in [(block.block_layer, ""), (block.grid_layer, "_1")]:
                    load_norm(layer.sa_norm, f"{prefix}/attn_layer_norm{suffix}")
                    layer.sa.attn_bias.copy_(get_param(f"{prefix}/attention{suffix}/relative_bias"))
                    load_linear(layer.sa.q_proj, f"{prefix}/attention{suffix}/q", 1)
                    load_linear(layer.sa.k_proj, f"{prefix}/attention{suffix}/k", 1)
                    load_linear(layer.sa.v_proj, f"{prefix}/attention{suffix}/v", 1)
                    load_linear(layer.sa.out_proj, f"{prefix}/attention{suffix}/o", 0)

                    load_norm(layer.mlp_norm, f"{prefix}/ffn_layer_norm{suffix}")
                    load_linear(layer.mlp.linear1, f"{prefix}/ffn{suffix}/expand_dense")
                    load_linear(layer.mlp.linear2, f"{prefix}/ffn{suffix}/shrink_dense")

        load_norm(self.norm, "final_layer_norm")
