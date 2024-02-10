# https://arxiv.org/abs/1706.03762

from functools import partial

import torch.nn.functional as F
from torch import Tensor, nn


class MHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int | None = None,
        head_dim: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        # small T5-small use n_heads * head_dim < d_model
        # small ViT use head_dim < 64
        if head_dim is None and n_heads is None:
            head_dim = 64
            n_heads = d_model // head_dim
        elif head_dim is None:
            head_dim = d_model // n_heads
        elif n_heads is None:
            n_heads = d_model // head_dim
        super().__init__()
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        attn_bias: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        k = k if k is not None else q
        v = v if v is not None else k

        q = self.q_proj(q).unflatten(-1, (self.n_heads, self.head_dim)).transpose(-2, -3)  # (*, n_heads, L, head_dim)
        k = self.k_proj(k).unflatten(-1, (self.n_heads, self.head_dim)).transpose(-2, -3)
        v = self.v_proj(v).unflatten(-1, (self.n_heads, self.head_dim)).transpose(-2, -3)

        dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout, is_causal=causal)
        return self.out_proj(out.transpose(-2, -3).flatten(-2))


class MLP(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: float, dropout: float = 0.0, act: str = "gelu") -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.act = dict(
            gelu=nn.GELU,
            approximate_gelu=partial(nn.GELU, approximate="tanh"),
            relu=partial(nn.ReLU, inplace=True),
            silu=nn.SiLU,
        )[act]()
        self.linear2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int | None = None,
        head_dim: int | None = None,
        cross_attn: bool = False,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        act: str = "gelu",
        pre_norm: bool = True,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm

        self.sa_norm = nn.LayerNorm(d_model, norm_eps)
        self.sa = MHA(d_model, n_heads, head_dim, bias, dropout)

        self.ca_norm = nn.LayerNorm(d_model, norm_eps) if cross_attn else None
        self.ca = MHA(d_model, n_heads, head_dim, bias, dropout) if cross_attn else None

        self.mlp_norm = nn.LayerNorm(d_model, norm_eps)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), dropout, act)

    def forward(self, x: Tensor, memory: Tensor | None = None) -> Tensor:
        if self.pre_norm:
            x = x + self.sa(self.sa_norm(x), causal=True)
            x = x + self.ca(self.ca_norm(x), memory) if self.ca is not None else x
            x = x + self.mlp(self.mlp_norm(x))
        else:
            x = self.sa_norm(x + self.sa(x, causal=True))
            x = self.ca_norm(x + self.ca(x, memory)) if self.ca is not None else x
            x = self.mlp_norm(x + self.mlp(x))
        return x


class EncoderLayer(DecoderLayer):
    def __init__(
        self,
        d_model: int,
        n_heads: int | None = None,
        head_dim: int | None = None,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        act: str = "gelu",
        pre_norm: bool = True,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__(d_model, n_heads, head_dim, False, bias, mlp_ratio, dropout, act, pre_norm, norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        if self.pre_norm:
            x = x + self.sa(self.sa_norm(x))
            x = x + self.mlp(self.mlp_norm(x))
        else:
            x = self.sa_norm(x + self.sa(x))
            x = self.mlp_norm(x + self.mlp(x))
        return x


class Encoder(nn.Sequential):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int | None = None,
        head_dim: int | None = None,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        act: str = "gelu",
        pre_norm: bool = True,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        for _ in range(n_layers):
            self.append(EncoderLayer(d_model, n_heads, head_dim, bias, mlp_ratio, dropout, act, pre_norm, norm_eps))


class Decoder(nn.ModuleList):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int | None = None,
        head_dim: int | None = None,
        cross_attn: bool = False,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        act: str = "gelu",
        pre_norm: bool = True,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        for _ in range(n_layers):
            self.append(
                DecoderLayer(d_model, n_heads, head_dim, cross_attn, bias, mlp_ratio, dropout, act, pre_norm, norm_eps)
            )

    def forward(self, x: Tensor, memory: Tensor | None = None) -> Tensor:
        for layer in self:
            x = layer(x, memory)
        return x
