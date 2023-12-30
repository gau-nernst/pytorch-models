import torch
from torch import Tensor, nn

from ..transformer import MHA
import torch.nn.functional as F


class SamMHA(MHA):
    def __init__(
        self,
        input_size: int | tuple[int, int],  # (height, width)
        d_model: int,
        n_heads: int | None = None,
        head_dim: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(d_model, n_heads, head_dim, bias, dropout)
        if isinstance(input_size, int):
            input_size = [input_size, input_size]

        # shared relative positional embeddings across attention heads
        self.pos_embed_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
        self.pos_embed_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))
        self.register_buffer("pos_embed_h_indices", self.get_relative_positions(input_size[0]))
        self.register_buffer("pos_embed_w_indices", self.get_relative_positions(input_size[1]))
        self.input_size = input_size

    @staticmethod
    def get_relative_positions(length: int) -> Tensor:
        return torch.arange(length).view(-1, 1) - torch.arange(length).view(1, -1) + length - 1

    def get_attn_bias(self, q: Tensor) -> Tensor:
        pos_embed_h = self.pos_embed_h[self.pos_embed_h_indices]  # (H, H, head_dim)
        pos_embed_w = self.pos_embed_w[self.pos_embed_w_indices]  # (W, W, head_dim)

        # attn_bias_h = torch.einsum("*hwc,hkc->*hwk1", _q, pos_embed_h)  # k=h here
        # attn_bias_w = torch.einsum("*hwc,wkc->*hw1k", _q, pos_embed_w)  # k=w here

        H, W = self.input_size
        _q = q.unflatten(-2, (H, W))  # (N, n_heads, H, W, head_dim)
        attn_bias_h = _q @ pos_embed_h.view(H, 1, H, self.head_dim).transpose(-1, -2)  # (N, n_heads, H, W, H)
        attn_bias_w = _q @ pos_embed_w.view(1, W, W, self.head_dim).transpose(-1, -2)  # (N, n_heads, H, W, W)
        return (attn_bias_h.unsqueeze(-1) + attn_bias_w.unsqueeze(-2)).view(q.shape[0], self.n_heads, H * W, H * W)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, HW, C)
        q = self.q_proj(x).unflatten(-1, (self.n_heads, self.head_dim)).transpose(-2, -3)  # (N, n_heads, HW, head_dim)
        k = self.k_proj(x).unflatten(-1, (self.n_heads, self.head_dim)).transpose(-2, -3)
        v = self.v_proj(x).unflatten(-1, (self.n_heads, self.head_dim)).transpose(-2, -3)

        attn_bias = self.get_attn_bias(q)
        dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout)
        return self.out_proj(out.transpose(-2, -3).flatten(-2))
