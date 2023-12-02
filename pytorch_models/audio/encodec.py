# https://arxiv.org/abs/2210.13438
# https://github.com/facebookresearch/encodec

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Pad1d(nn.Module):
    def __init__(self, kernel_size: int, stride: int, causal: bool) -> None:
        super().__init__()
        padding_total = kernel_size - stride
        self.right = 0 if causal else padding_total // 2
        self.left = padding_total - self.right
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        extra_padding = math.ceil(x.shape[2] / self.stride) * self.stride - x.shape[2]
        return F.pad(x, (self.left, self.right + extra_padding), mode="reflect")


class Unpad1d(nn.Module):
    def __init__(self, kernel_size: int, stride: int, causal: bool = False) -> None:
        super().__init__()
        padding_total = kernel_size - stride
        self.right = padding_total if causal else padding_total // 2
        self.left = padding_total - self.right

    def forward(self, x: Tensor) -> Tensor:
        return x[..., self.left : -self.right]


class Conv1d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        norm_type: str = "weight_norm",
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.pad = Pad1d(kernel_size, stride, causal)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.GroupNorm(1, out_channels) if norm_type == "time_group_norm" else nn.Identity()

        if norm_type == "weight_norm":
            nn.utils.weight_norm(self.conv)


class ConvTranspose1d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        norm_type: str = "weight_norm",
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.GroupNorm(1, out_channels) if norm_type == "time_group_norm" else nn.Identity()
        self.unpad = Unpad1d(kernel_size, stride, causal)

        if norm_type == "weight_norm":
            nn.utils.weight_norm(self.conv)


class LSTM(nn.LSTM):
    def __init__(self, dim: int, n_layers: int) -> None:
        super().__init__(dim, dim, n_layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x.permute(2, 0, 1))[0].permute(1, 2, 0)


class EnCodecBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, norm_type: str, causal: bool) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ELU(),
            Conv1d(dim, dim // 2, kernel_size, 1, norm_type, causal),
            nn.ELU(),
            Conv1d(dim // 2, dim, 1, 1, norm_type, causal),
        )
        self.shortcut = Conv1d(dim, dim, 1, 1, norm_type, causal)

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.layers(x)


class EnCodecEncoder(nn.Sequential):
    def __init__(
        self,
        audio_channels: int,
        base_dim: int = 32,
        dim: int = 128,
        strides: tuple[int, ...] = (2, 4, 5, 8),
        norm_type: str = "weight_norm",
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.append(Conv1d(audio_channels, base_dim, 7, norm_type=norm_type, causal=causal))

        for stride in strides:
            self.append(EnCodecBlock(base_dim, 3, norm_type, causal))
            self.append(nn.ELU())
            self.append(Conv1d(base_dim, base_dim * 2, stride * 2, stride, norm_type, causal))
            base_dim *= 2

        self.append(LSTM(base_dim, 2))
        self.append(nn.ELU())
        self.append(Conv1d(base_dim, dim, 7, 1, norm_type, causal))

    def load_facebook_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.load_state_dict({_rename_key(k): v for k, v in state_dict.items()})


class EnCodecDecoder(nn.Sequential):
    def __init__(
        self,
        audio_channels: int,
        base_dim: int = 32,
        dim: int = 128,
        strides: tuple[int, ...] = (8, 5, 4, 2),
        norm_type: str = "weight_norm",
        causal: bool = False,
    ) -> None:
        super().__init__()
        base_dim *= 2 ** len(strides)
        self.append(Conv1d(dim, base_dim, 7, 1, norm_type, causal))
        self.append(LSTM(base_dim, 2))

        for stride in strides:
            self.append(nn.ELU())
            self.append(ConvTranspose1d(base_dim, base_dim // 2, stride * 2, stride, norm_type, causal))
            self.append(EnCodecBlock(base_dim // 2, 3, norm_type, causal))
            base_dim //= 2

        self.append(nn.ELU())
        self.append(Conv1d(base_dim, audio_channels, 7, 1, norm_type, causal))

    def load_facebook_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.load_state_dict({_rename_key(k): v for k, v in state_dict.items()})


# only support inference
class VQ(nn.Module):
    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.register_buffer("embed", torch.zeros(codebook_size, dim))
        self.embed: Tensor

    def quantize(self, x: Tensor) -> Tensor:
        distances = x.square().sum(-1, keepdim=True) - 2 * x @ self.embed.T + self.embed.square().sum(-1)
        return distances.argmin(-1)

    def dequantize(self, x: Tensor) -> Tensor:
        return F.embedding(x, self.embed)


class RVQ(nn.ModuleList):
    def __init__(self, dim: int, codebook_size: int, n_quantizers: int) -> None:
        super().__init__([VQ(dim, codebook_size) for _ in range(n_quantizers)])

    def quantize(self, x: Tensor, n_quantizers: int | None = None) -> Tensor:
        n_quantizers = n_quantizers or len(self)
        all_indices = []

        for i in range(n_quantizers):
            indices = self[i].quantize(x)
            x = x - self[i].dequantize(indices)
            all_indices.append(indices)

        return torch.stack(all_indices, 0)

    def dequantize(self, x: Tensor) -> Tensor:
        out = self[0].dequantize(x[0])
        for i in range(1, x.shape[0]):
            out = out + self[i].dequantize(x[i])
        return out


class EnCodec(nn.Module):
    def __init__(self, audio_channels: int, norm_type: str, causal: bool, n_quantizers: int, normalize: bool) -> None:
        super().__init__()
        self.encoder = EnCodecEncoder(audio_channels, norm_type=norm_type, causal=causal)
        self.decoder = EnCodecDecoder(audio_channels, norm_type=norm_type, causal=causal)
        self.quantizer = RVQ(128, 1024, n_quantizers)
        self.normalize = normalize

    def encode(self, x: Tensor, n_quantizers: int | None = None) -> tuple[Tensor, Tensor | None]:
        if self.normalize:
            scale = x.mean(1, keepdim=True).square().mean(2, keepdim=True).sqrt() + 1e-8
            x = x / scale
        else:
            scale = None

        x = self.encoder(x)
        x = self.quantizer.quantize(x.transpose(1, 2), n_quantizers).transpose(0, 1)
        return x, scale

    def decode(self, x: Tensor, scale: Tensor | None = None) -> Tensor:
        x = self.quantizer.dequantize(x.transpose(0, 1)).transpose(1, 2)
        x = self.decoder(x)

        if scale is not None:
            x = x * scale
        return x

    @staticmethod
    def from_facebook(variant: str, pretrained: bool = False) -> "EnCodec":
        audio_channels, norm_type, causal, n_quantizers, normalize = {
            "24khz": (1, "weight_norm", True, 32, False),
            "48khz": (2, "time_group_norm", False, 16, True),
        }[variant]
        m = EnCodec(audio_channels, norm_type, causal, n_quantizers, normalize)

        if pretrained:
            ckpt = {
                "24khz": "encodec_24khz-d7cc33bc.th",
                "48khz": "encodec_48khz-7e698e3e.th",
            }[variant]
            base_url = "https://dl.fbaipublicfiles.com/encodec/v0/"
            state_dict = torch.hub.load_state_dict_from_url(base_url + ckpt)
            m.load_facebook_state_dict(state_dict)

        return m

    def load_facebook_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.load_state_dict({_rename_key(k): v for k, v in state_dict.items()}, strict=False)


def _rename_key(key: str) -> str:
    key = key.replace("model.", "")
    key = key.replace("conv.conv.", "conv.")
    key = key.replace("conv.norm.", "norm.")
    key = key.replace("convtr.convtr.", "conv.")
    key = key.replace("convtr.norm.", "norm.")
    key = key.replace("block.", "layers.")
    key = key.replace("lstm.", "")
    key = key.replace("vq.layers.", "")
    key = key.replace("_codebook.", "")
    return key
