# https://arxiv.org/abs/1706.03762

import torch.nn.functional as F
from torch import Tensor, nn


class MHA(nn.Module):
    def __init__(self, d_model: int, head_dim: int, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias)
        self.k_proj = nn.Linear(d_model, d_model, False)
        self.v_proj = nn.Linear(d_model, d_model, bias)
        self.out_proj = nn.Linear(d_model, d_model, bias)
        self.head_dim = head_dim
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        q = self.q_proj(x).unflatten(-1, (-1, self.head_dim)).transpose(1, 2)  # (B, n_heads, L, head_dim)
        k = self.k_proj(x).unflatten(-1, (-1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).unflatten(-1, (-1, self.head_dim)).transpose(1, 2)

        dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
        return self.out_proj(out.transpose(-2, -3).flatten(-2))


class MLP(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = MHA(d_model, head_dim, bias, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), dropout)
        self.pre_norm = pre_norm

    def forward(self, x: Tensor) -> Tensor:
        if self.pre_norm:
            x = x + self.mha(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            x = self.norm1(x + self.mha(x))
            x = self.norm2(x + self.mlp(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        head_dim: int,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential()
        for _ in range(n_layers):
            self.layers.append(EncoderBlock(d_model, head_dim, bias, mlp_ratio, dropout, pre_norm))
        self.norm = nn.LayerNorm(d_model)
        self.pre_norm = pre_norm

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.layers(x)) if self.pre_norm else self.layers(self.norm(x))
