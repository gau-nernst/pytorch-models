# https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
# https://github.com/openai/finetune-transformer-lm

import math

import torch
from torch import Tensor, nn

from ..transformer import Decoder


class GPT2(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        max_seq_len: int = 1024,
        vocab_size: int = 50257,
        dropout: float = 0.0,
    ):
        super().__init__()
        vocab_size = math.ceil(vocab_size / 64) * 64  # pad to multiple of 64
        self.token_embs = nn.Embedding(vocab_size, d_model)
        self.pos_embs = nn.Parameter(torch.zeros(max_seq_len, d_model))
        self.layers = Decoder(n_layers, d_model, dropout=dropout, act="approximate_gelu")
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embs(x)
        x = x + self.pos_embs[: x.shape[-2]]
        x = self.layers(x)
        x = self.norm(x)
        logits = x @ self.token_embs.weight.T
        return logits

    @staticmethod
    def from_hf(model_tag: str, *, pretrained=False, **kwargs) -> "GPT2":
        n_layers, d_model = {
            "gpt2": (12, 768),
            "gpt2-medium": (24, 1024),
            "gpt2-large": (36, 1280),
            "gpt2-xl": (48, 1600),
        }[model_tag]

        m = GPT2(n_layers, d_model, **kwargs)

        if pretrained:
            ckpt_url = f"https://huggingface.co/{model_tag}/resolve/main/pytorch_model.bin"
            state_dict = torch.hub.load_state_dict_from_url(ckpt_url, file_name=model_tag)
            m.load_hf_state_dict(state_dict)

        return m

    @torch.no_grad()
    def load_hf_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        state_dict = {k.removeprefix("transformer."): v for k, v in state_dict.items()}

        def copy_(module: nn.Linear | nn.LayerNorm, prefix: str):
            module.weight.copy_(state_dict.pop(f"{prefix}.weight"))
            if module.bias is not None:
                module.bias.copy_(state_dict.pop(f"{prefix}.bias"))

        token_embs = state_dict.pop("wte.weight")
        self.token_embs.weight[: token_embs.shape[0]] = token_embs
        self.pos_embs.copy_(state_dict.pop("wpe.weight"))

        for i, layer in enumerate(self.layers):
            prefix = f"h.{i}"
            copy_(layer.sa_norm, f"{prefix}.ln_1")
            copy_(layer.sa.out_proj, f"{prefix}.attn.c_proj")

            w_q, w_k, w_v = state_dict.pop(f"{prefix}.attn.c_attn.weight").squeeze(-1).chunk(3, 0)
            layer.sa.q_proj.weight.copy_(w_q)
            layer.sa.k_proj.weight.copy_(w_k)
            layer.sa.v_proj.weight.copy_(w_v)

            b_q, b_k, b_v = state_dict.pop(f"{prefix}.attn.c_attn.bias").chunk(3, 0)
            layer.sa.q_proj.bias.copy_(b_q)
            layer.sa.k_proj.bias.copy_(b_k)
            layer.sa.v_proj.bias.copy_(b_v)

            copy_(layer.mlp_norm, f"{prefix}.ln_2")
            copy_(layer.mlp.linear1, f"{prefix}.mlp.c_fc")
            copy_(layer.mlp.linear2, f"{prefix}.mlp.c_proj")

        copy_(self.norm, f"ln_f")
        print(state_dict.keys())

    @staticmethod
    def get_tokenizer():
        from tiktoken import get_encoding

        return get_encoding("gpt2")
