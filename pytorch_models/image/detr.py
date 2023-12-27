# https://arxiv.org/abs/2005.12872
# https://github.com/facebookresearch/detr

import torch
from torch import Tensor, nn

from ..transformer import DecoderLayer, EncoderLayer


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
            nn.Sequential(nn.Conv2d(in_dim, out_dim, bias=False), nn.BatchNorm2d(out_dim))
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
        super().__init__(d_model, cross_attn=True, act="relu", pre_norm=False)

    def forward(self, x: Tensor, memory: Tensor, query_embed: Tensor, pos_embed: Tensor) -> Tensor:
        q = k = x + query_embed
        x = self.sa_norm(x + self.sa(q, k, x))
        x = self.ca_norm(x + self.ca(x + query_embed, memory + pos_embed, memory))
        x = self.mlp_norm(x + self.mlp(x))
        return x


class DETREncoderLayer(EncoderLayer):
    def __init__(self, d_model: int) -> None:
        super().__init__(d_model, act="relu", pre_norm=False)

    def forward(self, x: Tensor, pos_embed: Tensor) -> Tensor:
        q = k = x + pos_embed
        x = self.sa_norm(x + self.sa(q, k, x))
        x = self.mlp_norm(x + self.mlp(x))
        return x


class LearnedPositionEmbedding2d(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.row_embed = nn.Parameter(torch.zeros(50, d_model // 2))
        self.col_embed = nn.Parameter(torch.zeros(50, d_model // 2))

    def forward(self, h: int, w: int) -> Tensor:
        x_emb = self.col_embed[:w].view(1, w, -1).expand(h, w, -1)
        y_emb = self.row_embed[:h].view(h, 1, -1).expand(h, w, -1)
        return torch.cat([x_emb, y_emb], dim=2)  # (H, W, C)


class DETR(nn.Module):
    def __init__(self, backbone_layers: list[int], d_model: int, n_classes: int, n_queries: int) -> None:
        super().__init__()
        self.backbone = ResNet(backbone_layers)

        self.input_proj = nn.Conv2d(self.backbone.out_dim, d_model, 1)
        self.pos_embed = LearnedPositionEmbedding2d(d_model)
        self.query_embed = nn.Parameter(torch.zeros(n_queries, d_model))
        self.encoder = nn.ModuleList([DETREncoderLayer(d_model) for _ in range(6)])
        self.decoder = nn.ModuleList([DETRDecoderLayer(d_model) for _ in range(6)])

        self.classifier = nn.Linear(d_model, n_classes + 1)
        self.bbox_head = nn.Sequential(
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
            query = layer(query, x, pos_embed, self.query_embed)

        logits = self.classifier(query)
        bboxes = self.bbox_head(query)

        return logits, bboxes
