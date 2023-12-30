# https://arxiv.org/abs/2005.12872
# https://github.com/facebookresearch/detr

import torch
from torch import Tensor, nn

from ..transformer import MHA, DecoderLayer, EncoderLayer


class Bottleneck(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        bottleneck = out_dim // 4
        self.residual = nn.Sequential(
            nn.Conv2d(in_dim, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, bottleneck, 3, stride, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_dim, out_dim, 1, stride, bias=False), nn.BatchNorm2d(out_dim))
            if stride > 1 or out_dim != in_dim
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.shortcut(x) + self.residual(x))


class ResNetStage(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, stride: int) -> None:
        super().__init__(
            Bottleneck(in_dim, out_dim, stride=stride), *[Bottleneck(out_dim, out_dim) for _ in range(n_layers - 1)]
        )


class ResNet(nn.Module):
    def __init__(self, n_layers: list[int]) -> None:
        super().__init__()
        in_dim = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_dim, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )

        self.stages = nn.Sequential()
        for i, n_layer in enumerate(n_layers):
            self.stages.append(ResNetStage(in_dim, 256 * 2**i, n_layer, 1 if i == 0 else 2))
            in_dim = 256 * 2**i
        self.out_dim = in_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return x


class DETRDecoderLayer(DecoderLayer):
    def __init__(self, d_model: int) -> None:
        super().__init__(d_model, n_heads=8, cross_attn=True, act="relu", mlp_ratio=8, pre_norm=False)

    def forward(self, x: Tensor, memory: Tensor, query_embed: Tensor, pos_embed: Tensor) -> Tensor:
        q = k = x + query_embed
        x = self.sa_norm(x + self.sa(q, k, x))
        x = self.ca_norm(x + self.ca(x + query_embed, memory + pos_embed, memory))
        x = self.mlp_norm(x + self.mlp(x))
        return x


class DETREncoderLayer(EncoderLayer):
    def __init__(self, d_model: int) -> None:
        super().__init__(d_model, n_heads=8, act="relu", mlp_ratio=8, pre_norm=False)

    def forward(self, x: Tensor, pos_embed: Tensor) -> Tensor:
        q = k = x + pos_embed
        x = self.sa_norm(x + self.sa(q, k, x))
        x = self.mlp_norm(x + self.mlp(x))
        return x


class SinusoidalPositionEmbedding2d(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        d_model //= 2  # need to divide by 2, since half is for x, half is for y
        freqs = 10_000 ** (-2 * torch.arange(d_model // 2) / d_model)
        self.register_buffer("freqs", freqs, persistent=False)

    def _make_embed(self, x: int) -> Tensor:
        ts = torch.arange(1, x + 1, device=self.freqs.device, dtype=self.freqs.dtype) / (x + 1e-6) * 2 * torch.pi
        out = ts.view(-1, 1) * self.freqs
        return torch.stack([out.sin(), out.cos()], dim=2).flatten(1)  # interleave pattern

    def forward(self, h: int, w: int) -> Tensor:
        y_emb = self._make_embed(h).view(h, 1, -1).expand(h, w, -1)
        x_emb = self._make_embed(w).view(1, w, -1).expand(h, w, -1)
        return torch.cat([y_emb, x_emb], dim=2)


class DETR(nn.Module):
    def __init__(
        self, backbone_layers: list[int], d_model: int = 256, n_classes: int = 91, n_queries: int = 100
    ) -> None:
        super().__init__()
        self.backbone = ResNet(backbone_layers)

        self.input_proj = nn.Conv2d(self.backbone.out_dim, d_model, 1)
        self.pos_embed = SinusoidalPositionEmbedding2d(d_model)
        self.query_embed = nn.Parameter(torch.zeros(n_queries, d_model))
        self.encoder = nn.ModuleList([DETREncoderLayer(d_model) for _ in range(6)])
        self.decoder = nn.ModuleList([DETRDecoderLayer(d_model) for _ in range(6)])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.input_proj(x)
        pos_embed = self.pos_embed(x.shape[-2], x.shape[-1])

        x = x.flatten(-2).transpose(-1, -2)  # (C, H, W) -> (HW, C)
        pos_embed = pos_embed.flatten(0, 1)
        for layer in self.encoder:
            x = layer(x, pos_embed)

        query = torch.zeros_like(self.query_embed)
        for layer in self.decoder:
            query = layer(query, x, self.query_embed, pos_embed)

        query = self.norm(query)
        logits = self.classifier(query)
        boxes = self.box_head(query)

        return logits, boxes

    @staticmethod
    def from_facebook(model_tag: str, *, pretrained: bool = False) -> "DETR":
        backbone_layers, ckpt = {
            "resnet50": ([3, 4, 6, 3], "detr-r50-e632da11.pth"),
            "resnet50-dc5": ([3, 4, 6, 3], "detr-r50-dc5-f0fb7ef5.pth"),
            "resnet101": ([3, 4, 23, 3], "detr-r101-2c7b67e5.pth"),
            "resnet101-dc5": ([3, 4, 23, 3], "detr-r101-dc5-a2e86def.pth"),
        }[model_tag]

        m = DETR(backbone_layers)

        if pretrained:
            url = f"https://dl.fbaipublicfiles.com/detr/{ckpt}"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")["model"]
            m.load_facebook_state_dict(state_dict)

        return m

    @torch.no_grad()
    def load_facebook_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        state_dict = state_dict.copy()  # shallow copy

        def copy_(m: nn.Conv2d | nn.Linear | nn.BatchNorm2d | nn.LayerNorm, prefix: str):
            m.weight.copy_(state_dict.pop(f"{prefix}.weight"))
            if m.bias is not None:
                m.bias.copy_(state_dict.pop(f"{prefix}.bias"))
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.copy_(state_dict.pop(f"{prefix}.running_mean"))
                m.running_var.copy_(state_dict.pop(f"{prefix}.running_var"))

        def copy_mha_(m: MHA, prefix: str):
            qw, kw, vw = state_dict.pop(f"{prefix}.in_proj_weight").chunk(3, dim=0)
            m.q_proj.weight.copy_(qw)
            m.k_proj.weight.copy_(kw)
            m.v_proj.weight.copy_(vw)

            qb, kb, vb = state_dict.pop(f"{prefix}.in_proj_bias").chunk(3, dim=0)
            m.q_proj.bias.copy_(qb)
            m.k_proj.bias.copy_(kb)
            m.v_proj.bias.copy_(vb)

            copy_(m.out_proj, f"{prefix}.out_proj")

        copy_(self.backbone.stem[0], "backbone.0.body.conv1")
        copy_(self.backbone.stem[1], "backbone.0.body.bn1")

        for stage_idx, stage in enumerate(self.backbone.stages):
            for block_idx, bottleneck in enumerate(stage):
                prefix = f"backbone.0.body.layer{stage_idx + 1}.{block_idx}"

                copy_(bottleneck.residual[0], f"{prefix}.conv1")
                copy_(bottleneck.residual[1], f"{prefix}.bn1")
                copy_(bottleneck.residual[3], f"{prefix}.conv2")
                copy_(bottleneck.residual[4], f"{prefix}.bn2")
                copy_(bottleneck.residual[6], f"{prefix}.conv3")
                copy_(bottleneck.residual[7], f"{prefix}.bn3")

                if block_idx == 0:
                    copy_(bottleneck.shortcut[0], f"{prefix}.downsample.0")
                    copy_(bottleneck.shortcut[1], f"{prefix}.downsample.1")

        copy_(self.input_proj, "input_proj")
        self.query_embed.copy_(state_dict.pop("query_embed.weight"))

        for _t in ["encoder", "decoder"]:
            for layer_idx, layer in enumerate(getattr(self, _t)):
                prefix = f"transformer.{_t}.layers.{layer_idx}"

                copy_mha_(layer.sa, f"{prefix}.self_attn")
                copy_(layer.sa_norm, f"{prefix}.norm1")

                if _t == "decoder":
                    copy_mha_(layer.ca, f"{prefix}.multihead_attn")
                    copy_(layer.ca_norm, f"{prefix}.norm2")

                copy_(layer.mlp.linear1, f"{prefix}.linear1")
                copy_(layer.mlp.linear2, f"{prefix}.linear2")
                copy_(layer.mlp_norm, f"{prefix}.norm2" if _t == "encoder" else f"{prefix}.norm3")

        copy_(self.norm, "transformer.decoder.norm")
        copy_(self.classifier, "class_embed")
        copy_(self.box_head[0], "bbox_embed.layers.0")
        copy_(self.box_head[2], "bbox_embed.layers.1")
        copy_(self.box_head[4], "bbox_embed.layers.2")
