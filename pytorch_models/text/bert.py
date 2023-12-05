# https://arxiv.org/abs/1810.04805
# https://github.com/google-research/bert

import json
import math

import requests
import torch
from torch import Tensor, nn

from ..transformer import Encoder


# NOTE: token_type_embeddings, pooler, and classifier not included
# latest google-research/bert use approximate GELU, though it's initial release used exact GELU
# HF uses exact GELU. for simplicity, we use exact GELU
class BERT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        layernorm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        vocab_size = math.ceil(vocab_size / 64) * 64  # pad to multiple of 64
        self.token_embs = nn.Embedding(vocab_size, d_model)
        self.pos_embs = nn.Parameter(torch.zeros(max_seq_len, d_model))
        self.encoder = Encoder(n_layers, d_model, dropout=dropout, pre_norm=False, layernorm_eps=layernorm_eps)

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

        # RoBERTa skips the first 2 position embeddings for no good reasons
        if "roberta" in config["model_type"]:
            config["max_position_embeddings"] -= 2

        m = BERT(
            vocab_size=config["vocab_size"],
            n_layers=config["num_hidden_layers"],
            d_model=config["hidden_size"],
            max_seq_len=config["max_position_embeddings"],
            layernorm_eps=config["layer_norm_eps"],
            **kwargs,
        )

        if pretrained:
            ckpt_url = f"https://huggingface.co/{_model_tag}/resolve/main/pytorch_model.bin"
            state_dict = torch.hub.load_state_dict_from_url(ckpt_url, file_name=_model_tag.replace("/", "_"))
            m.load_hf_state_dict(state_dict)

        return m

    @torch.no_grad()
    def load_hf_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        is_roberta = any(k.startswith("roberta.") for k in state_dict.keys())
        state_dict = {k.removeprefix("bert.").removeprefix("roberta."): v for k, v in state_dict.items()}

        def copy_(module: nn.Linear | nn.LayerNorm, prefix: str):
            module.weight.copy_(state_dict.pop(f"{prefix}.weight"))
            if module.bias is not None:
                module.bias.copy_(state_dict.pop(f"{prefix}.bias"))

        token_embs = state_dict.pop("embeddings.word_embeddings.weight")
        self.token_embs.weight[: token_embs.shape[0]] = token_embs

        # we merge token_type_embs to pos_embs
        pos_embs = state_dict.pop("embeddings.position_embeddings.weight")
        if is_roberta:  # remove the unused first 2 positional embeddings in RoBERTa
            pos_embs = pos_embs[2:]
        token_type_emb = state_dict.pop("embeddings.token_type_embeddings.weight")[0]
        self.pos_embs.copy_(pos_embs + token_type_emb)

        copy_(self.encoder.norm, "embeddings.LayerNorm")
        for i, layer in enumerate(self.encoder.layers):
            prefix = f"encoder.layer.{i}"
            state_dict.pop(f"{prefix}.attention.self.key.bias")

            copy_(layer.sa.q_proj, f"{prefix}.attention.self.query")
            copy_(layer.sa.k_proj, f"{prefix}.attention.self.key")
            copy_(layer.sa.v_proj, f"{prefix}.attention.self.value")
            copy_(layer.sa.out_proj, f"{prefix}.attention.output.dense")
            copy_(layer.sa_norm, f"{prefix}.attention.output.LayerNorm")

            copy_(layer.mlp.linear1, f"{prefix}.intermediate.dense")
            copy_(layer.mlp.linear2, f"{prefix}.output.dense")
            copy_(layer.mlp_norm, f"{prefix}.output.LayerNorm")

        print(state_dict.keys())
