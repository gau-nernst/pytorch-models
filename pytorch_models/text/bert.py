# https://arxiv.org/abs/1810.04805
# https://github.com/google-research/bert

import json
import math

import requests
import torch
from torch import Tensor, nn

from ..transformer import Encoder


class BERT(nn.Module):
    def __init__(
        self, vocab_size: int, n_layers: int, d_model: int, max_seq_len: int = 512, dropout: float = 0.0
    ) -> None:
        super().__init__()
        vocab_size = math.ceil(vocab_size / 64) * 64  # pad to multiple of 64
        self.token_embs = nn.Embedding(vocab_size, d_model)
        self.pos_embs = nn.Parameter(torch.zeros(max_seq_len, d_model))
        self.encoder = Encoder(n_layers, d_model, 64, dropout=dropout, pre_norm=False, layernorm_eps=1e-12)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embs(x)
        x = x + self.pos_embs[: x.shape[-2]]
        x = self.encoder(x)
        return x

    @staticmethod
    def from_hf(model_tag: str, *, pretrained: bool = False, **kwargs) -> "BERT":
        config = None
        for _model_tag in (model_tag, f"gaunernst/{model_tag}"):
            config_url = f"https://huggingface.co/{_model_tag}/raw/main/config.json"
            resp = requests.get(config_url)
            if resp.ok:
                config = json.loads(resp.content)
                break

        if config is None:
            raise ValueError(f"Unsupported model {model_tag}")

        m = BERT(
            vocab_size=config["vocab_size"],
            n_layers=config["num_hidden_layers"],
            d_model=config["hidden_size"],
            max_seq_len=config["max_position_embeddings"],
            **kwargs,
        )

        if pretrained:
            ckpt_url = f"https://huggingface.co/{_model_tag}/resolve/main/pytorch_model.bin"
            state_dict = torch.hub.load_state_dict_from_url(ckpt_url, file_name=_model_tag.replace("/", "_"))
            m.load_hf_state_dict(state_dict)

        return m

    @torch.no_grad()
    def load_hf_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        state_dict = {k.removeprefix("bert."): v for k, v in state_dict.items() if k.startswith("bert.")}

        def copy_(module: nn.Linear | nn.LayerNorm, prefix: str):
            module.weight.copy_(state_dict.pop(f"{prefix}.weight"))
            if module.bias is not None:
                module.bias.copy_(state_dict.pop(f"{prefix}.bias"))

        token_embs = state_dict.pop("embeddings.word_embeddings.weight")
        self.token_embs.weight[: token_embs.shape[0]] = token_embs

        # we merge token_type_embs to pos_embs
        self.pos_embs.copy_(state_dict.pop("embeddings.position_embeddings.weight"))
        self.pos_embs.add_(state_dict.pop("embeddings.token_type_embeddings.weight")[0])
        copy_(self.encoder.norm, "embeddings.LayerNorm")

        for i, layer in enumerate(self.encoder.layers):
            prefix = f"encoder.layer.{i}"
            state_dict.pop(f"{prefix}.attention.self.key.bias")

            copy_(layer.mha.q_proj, f"{prefix}.attention.self.query")
            copy_(layer.mha.k_proj, f"{prefix}.attention.self.key")
            copy_(layer.mha.v_proj, f"{prefix}.attention.self.value")
            copy_(layer.mha.out_proj, f"{prefix}.attention.output.dense")
            copy_(layer.norm1, f"{prefix}.attention.output.LayerNorm")

            copy_(layer.mlp.linear1, f"{prefix}.intermediate.dense")
            copy_(layer.mlp.linear2, f"{prefix}.output.dense")
            copy_(layer.norm2, f"{prefix}.output.LayerNorm")

        print(state_dict.keys())
