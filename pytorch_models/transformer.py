# https://arxiv.org/abs/1706.03762

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
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout

    def forward(
        self, q: Tensor, k: Tensor | None = None, v: Tensor | None = None, attn_bias: Tensor | None = None
    ) -> Tensor:
        k = q if k is None else k
        v = k if v is None else v

        q = self.q_proj(q).unflatten(-1, (self.n_heads, self.head_dim)).transpose(-2, -3)  # (*, n_heads, L, head_dim)
        k = self.k_proj(k).unflatten(-1, (self.n_heads, self.head_dim)).transpose(-2, -3)
        v = self.v_proj(v).unflatten(-1, (self.n_heads, self.head_dim)).transpose(-2, -3)

        dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout)
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
        n_heads: int | None = None,
        head_dim: int | None = None,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pre_norm: bool = True,
        layernorm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.mha = MHA(d_model, n_heads, head_dim, bias, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=layernorm_eps)
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
        n_heads: int | None = None,
        head_dim: int | None = None,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pre_norm: bool = True,
        layernorm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential()
        for _ in range(n_layers):
            self.layers.append(
                EncoderBlock(d_model, n_heads, head_dim, bias, mlp_ratio, dropout, pre_norm, layernorm_eps)
            )
        self.norm = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.pre_norm = pre_norm

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.layers(x)) if self.pre_norm else self.layers(self.norm(x))
