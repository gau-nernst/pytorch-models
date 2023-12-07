# https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
# https://github.com/openai/finetune-transformer-lm

import json
import math

import numpy as np
import requests
import torch
from torch import Tensor, nn

from ..transformer import Decoder
from ..utils import torch_hub_download


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 40478,
        n_layers: int = 12,
        d_model: int = 768,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        vocab_size = math.ceil(vocab_size / 64) * 64  # pad to multiple of 64
        self.token_embs = nn.Embedding(vocab_size, d_model)
        self.pos_embs = nn.Parameter(torch.zeros(max_seq_len, d_model))
        self.layers = Decoder(n_layers, d_model, dropout=dropout, pre_norm=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embs(x)
        x = x + self.pos_embs[: x.shape[-2]]
        x = self.layers(x)
        logits = x @ self.token_embs.weight.T
        return logits

    @staticmethod
    def from_openai(*, pretrained=False, **kwargs) -> "GPT":
        m = GPT(**kwargs)

        if pretrained:
            # https://github.com/openai/finetune-transformer-lm/blob/master/train.py#L404
            BASE_URL = "https://github.com/openai/finetune-transformer-lm/raw/master/model"

            shapes = json.loads(requests.get(f"{BASE_URL}/params_shapes.json").content)
            offsets = np.cumsum([np.prod(shape) for shape in shapes])

            shard_paths = []
            for i in range(10):
                url = f"{BASE_URL}/params_{i}.npy"
                shard_paths.append(torch_hub_download(url, subdir="openai_gpt"))

            shards = [np.load(path) for path in shard_paths]
            params = np.concatenate(shards, axis=0)
            params = np.split(params, offsets)[:-1]
            params = [param.reshape(shape) for param, shape in zip(params, shapes)]

            params = [torch.from_numpy(p) for p in params]

            with torch.no_grad():
                m.pos_embs.copy_(params[0])
                m.token_embs.weight[: params[1].shape[0]] = params[1]

                n = 12
                for i, layer in enumerate(m.layers):
                    w_q, w_k, w_v = params[2 + i * n].squeeze(0).chunk(3, -1)
                    layer.sa.q_proj.weight.copy_(w_q.T)
                    layer.sa.k_proj.weight.copy_(w_k.T)
                    layer.sa.v_proj.weight.copy_(w_v.T)

                    b_q, b_k, b_v = params[3 + i * n].chunk(3, -1)
                    layer.sa.q_proj.bias.copy_(b_q)
                    layer.sa.k_proj.bias.copy_(b_k)
                    layer.sa.v_proj.bias.copy_(b_v)

                    layer.sa.out_proj.weight.copy_(params[4 + i * n].squeeze(0).T)
                    layer.sa.out_proj.bias.copy_(params[5 + i * n])

                    layer.sa_norm.weight.copy_(params[6 + i * n])
                    layer.sa_norm.bias.copy_(params[7 + i * n])

                    layer.mlp.linear1.weight.copy_(params[8 + i * n].squeeze(0).T)
                    layer.mlp.linear1.bias.copy_(params[9 + i * n])

                    layer.mlp.linear2.weight.copy_(params[10 + i * n].squeeze(0).T)
                    layer.mlp.linear2.bias.copy_(params[11 + i * n])

                    layer.mlp_norm.weight.copy_(params[12 + i * n])
                    layer.mlp_norm.bias.copy_(params[13 + i * n])

        return m
