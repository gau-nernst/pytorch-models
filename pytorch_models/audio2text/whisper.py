# https://arxiv.org/abs/2212.04356
# https://github.com/openai/whisper

import torch
from torch import Tensor, nn

from ..audio.spectrogram import MelSpectrogram
from ..transformer import Decoder, Encoder


class WhisperEncoder(nn.Module):
    max_seq_len = 3000

    def __init__(self, n_layers: int, d_model: int, n_mels: int = 80, dropout: float = 0.0) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, d_model, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, 2, 1),
            nn.GELU(),
        )
        # sinusoids do not match OpenAI weights exactly.
        # initialize pe to zeros and load it from OpenAI weights later.
        self.register_buffer("pos_embs", torch.zeros(self.max_seq_len // 2, d_model))
        self.pos_embs: Tensor
        self.encoder = Encoder(n_layers, d_model, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x).transpose(1, 2)
        x = x + self.pos_embs[: x.shape[1]]
        x = self.encoder(x)
        return x


class WhisperDecoder(nn.Module):
    max_seq_len = 448

    def __init__(self, vocab_size: int, n_layers: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_embs = nn.Embedding(vocab_size, d_model)
        self.pos_embs = nn.Parameter(torch.zeros(self.max_seq_len, d_model))
        self.decoder = Decoder(n_layers, d_model, cross_attn=True, dropout=dropout)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        x = self.token_embs(x)
        x = x + self.pos_embs[: x.shape[1]]
        x = self.decoder(x, memory)
        x = x @ self.token_embs.weight.T  # weight-tying
        return x


class Whisper(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, d_model: int, n_mels: int = 80, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = WhisperEncoder(n_layers, d_model, n_mels, dropout=dropout)
        self.decoder = WhisperDecoder(vocab_size, n_layers, d_model, dropout=dropout)

    def forward(self, x: Tensor, targets: Tensor) -> Tensor:
        return self.decoder(targets, self.encoder(x))

    @staticmethod
    def from_openai(model_tag: str, *, pretrained: bool = False, **kwargs) -> "Whisper":
        n_layers, d_model, ckpt_hash = {
            "tiny": (4, 384, "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9"),
            "tiny.en": (4, 384, "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03"),
            "base": (8, 512, "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e"),
            "base.en": (8, 512, "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead"),
            "small": (12, 768, "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"),
            "small.en": (12, 768, "f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872"),
            "medium": (24, 1024, "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1"),
            "medium.en": (24, 1024, "d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f"),
            "large-v1": (32, 1280, "e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a"),
            "large-v2": (32, 1280, "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524"),
            "large-v3": (32, 1280, "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb"),
        }[model_tag]

        if model_tag == "large-v3":
            n_mels = 128
            vocab_size = 51866
        else:
            n_mels = 80
            vocab_size = 51864 if model_tag.endswith(".en") else 51865

        m = Whisper(vocab_size, n_layers, d_model, n_mels, **kwargs)
        if pretrained:
            url = f"https://openaipublic.azureedge.net/main/whisper/models/{ckpt_hash}/{model_tag}.pt"
            state_dict = torch.hub.load_state_dict_from_url(url, file_name=f"whisper_{model_tag}")["model_state_dict"]
            m.load_openai_state_dict(state_dict)

        return m

    @torch.no_grad()
    def load_openai_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        state_dict = state_dict.copy()

        def copy_w(module: nn.Conv1d | nn.Linear | nn.LayerNorm, prefix: str):
            module.weight.copy_(state_dict.pop(f"{prefix}.weight"))
            if module.bias is not None:
                module.bias.copy_(state_dict.pop(f"{prefix}.bias"))

        copy_w(self.encoder.stem[0], "encoder.conv1")
        copy_w(self.encoder.stem[2], "encoder.conv2")
        self.encoder.pos_embs.copy_(state_dict.pop("encoder.positional_embedding"))

        self.decoder.token_embs.weight.copy_(state_dict.pop("decoder.token_embedding.weight"))
        self.decoder.pos_embs.copy_(state_dict.pop("decoder.positional_embedding"))

        for transformer, _prefix in [(self.encoder.encoder, "encoder"), (self.decoder.decoder, "decoder")]:
            for i, block in enumerate(transformer.layers):
                prefix = f"{_prefix}.blocks.{i}"
                copy_w(block.sa.q_proj, f"{prefix}.attn.query")
                copy_w(block.sa.k_proj, f"{prefix}.attn.key")
                copy_w(block.sa.v_proj, f"{prefix}.attn.value")
                copy_w(block.sa.out_proj, f"{prefix}.attn.out")
                copy_w(block.sa_norm, f"{prefix}.attn_ln")

                if block.ca is not None:
                    copy_w(block.ca.q_proj, f"{prefix}.cross_attn.query")
                    copy_w(block.ca.k_proj, f"{prefix}.cross_attn.key")
                    copy_w(block.ca.v_proj, f"{prefix}.cross_attn.value")
                    copy_w(block.ca.out_proj, f"{prefix}.cross_attn.out")
                    copy_w(block.ca_norm, f"{prefix}.cross_attn_ln")

                copy_w(block.mlp.linear1, f"{prefix}.mlp.0")
                copy_w(block.mlp.linear2, f"{prefix}.mlp.2")
                copy_w(block.mlp_norm, f"{prefix}.mlp_ln")

            copy_w(transformer.norm, "encoder.ln_post" if _prefix == "encoder" else "decoder.ln")

        if len(state_dict) > 0:
            print(state_dict.keys())


class WhisperPreprocessor(MelSpectrogram):
    def __init__(self, variant: str = "tiny") -> None:
        n_mels = 128 if variant == "large-v3" else 80
        super().__init__(400, 160, n_mels, 16_000)

    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)[..., :-1]
        x = x.clamp(0).log10()
        x = x.maximum(x.flatten(-2).max(-1, keepdim=True)[0].unsqueeze(-1) - 8)
        x = (x + 4) / 4
        return x
