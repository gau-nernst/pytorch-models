# https://arxiv.org/abs/1810.04805
# https://github.com/google-research/bert

import torch
from torch import Tensor, nn

from ..transformer import Encoder


class BERT(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, n_layers: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_embs = nn.Embedding(vocab_size, d_model)
        self.pos_embs = nn.Parameter(torch.zeros(max_seq_len, d_model))
        self.encoder = Encoder(n_layers, d_model, 64, dropout=dropout, pre_norm=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embs(x)
        x = x + self.pos_embs[: x.shape[-2]]
        x = self.encoder(x)
        return x
