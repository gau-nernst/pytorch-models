# T5: https://arxiv.org/pdf/1910.10683
# Flan-T5: https://arxiv.org/abs/2210.11416
# https://github.com/google-research/t5x/blob/main/t5x/examples/t5/network.py

import math
from pathlib import Path

import torch
from torch import Tensor, nn

from ..transformer import MHA


# LayerNorm without mean subtraction and learnable bias
class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        _x = x.float()
        _x = _x * _x.square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        return _x.to(x.dtype) * self.weight.to(x.dtype)


# https://arxiv.org/pdf/2002.05202.pdf
class GEGLU(nn.Module):
    def __init__(self, dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim, mlp_dim, False)
        self.v = nn.Linear(dim, mlp_dim, False)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.w(x)) * self.v(x)


class RelativePositionBias(nn.Module):
    def __init__(self, n_heads: int, n_buckets: int = 32, max_distance: int = 128) -> None:
        super().__init__()
        self.n_buckets = n_buckets
        self.max_distance = max_distance
        self.bias = nn.Parameter(torch.zeros(n_heads, n_buckets))

    def forward(self, length: int, bidirection: bool) -> Tensor:
        indices = torch.arange(length)
        positions = indices.view(1, -1) - indices.view(-1, 1)

        if bidirection:
            n_buckets = self.n_buckets // 2  # half for positives, half for negatives
            offsets = torch.where(positions > 0, n_buckets, 0)  # offset for positives
            positions = positions.abs()
        else:
            n_buckets = self.n_buckets
            offsets = 0
            positions = (-positions).clamp(0)

        # [0, n_buckets // 2] -> linear scale
        # [n_buckets // 2, max_distance] -> log scale
        # [max_distance, infinity] -> clip
        max_exact = n_buckets // 2
        eps = torch.finfo(torch.float32).eps
        scale = (n_buckets - max_exact) / math.log(self.max_distance / max_exact)

        val_if_large = max_exact + (torch.log(positions / max_exact + eps) * scale).long()
        val_if_large = val_if_large.clamp(max=n_buckets - 1)

        indices = torch.where(positions < max_exact, positions, val_if_large) + offsets
        return self.bias[:, indices.to(self.bias.device)]


class T5Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_dim: int, dropout: float = 0.0, cross_attn: bool = False) -> None:
        super().__init__()
        self.sa_norm = LayerNorm(dim)
        self.sa = MHA(dim, n_heads=n_heads, head_dim=64, bias=False, dropout=dropout)

        self.ca_norm = LayerNorm(dim) if cross_attn else None
        self.ca = MHA(dim, n_heads=n_heads, head_dim=64, bias=False, dropout=dropout) if cross_attn else None

        self.mlp_norm = LayerNorm(dim)
        self.mlp = nn.Sequential(
            GEGLU(dim, mlp_dim),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim, False),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, memory: Tensor | None = None, attn_bias: Tensor | None = None) -> Tensor:
        x = x + self.sa(self.sa_norm(x), attn_bias=attn_bias)
        if self.ca is not None:
            x = x + self.ca(self.ca_norm(x), memory)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class T5Encoder(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_layers: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_drop = nn.Dropout(dropout)
        self.attn_bias = RelativePositionBias(n_heads)
        self.layers = nn.Sequential(*[T5Block(dim, n_heads, mlp_dim, dropout, False) for _ in range(n_layers)])
        self.norm = LayerNorm(dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        attn_bias = self.attn_bias(x.shape[-2], bidirection=True)
        x = self.in_drop(x)
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias)
        return self.out_drop(self.norm(x))


class T5Decoder(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_layers: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_drop = nn.Dropout(dropout)
        self.attn_bias = RelativePositionBias(n_heads)
        self.layers = nn.Sequential(*[T5Block(dim, n_heads, mlp_dim, dropout, True) for _ in range(n_layers)])
        self.norm = LayerNorm(dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        causal_mask = x.new_full((x.shape[-2], x.shape[-2]), -1e10).triu(1)
        attn_bias = self.attn_bias(x.shape[-2], bidirection=False) + causal_mask
        x = self.in_drop(x)
        for layer in self.layers:
            x = layer(x, memory, attn_bias=attn_bias)
        return self.out_drop(self.norm(x))


class T5Model(nn.Module):
    def __init__(
        self, vocab_size: int, dim: int, n_heads: int, n_layers: int, mlp_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.token_embs = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(dim, n_heads, n_layers, mlp_dim, dropout)
        self.decoder = T5Decoder(dim, n_heads, n_layers, mlp_dim, dropout)
        self.classifier = nn.Linear(dim, vocab_size, False)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(self.token_embs(x))

    def decode(self, x: Tensor, memory: Tensor) -> Tensor:
        return self.classifier(self.decoder(self.token_embs(x), memory))

    def forward(self, x: Tensor, targets: Tensor) -> Tensor:
        return self.decode(targets, self.encode(x))

    @staticmethod
    def from_t5x(model_tag: str, *, pretrained: bool = False, **kwargs) -> "T5Model":
        variant, size = model_tag.split("-")

        dim, n_heads, n_layers, mlp_dim = dict(
            small=(512, 6, 8, 1024),
            base=(768, 12, 12, 2048),
            large=(1024, 16, 24, 2816),
            xl=(2048, 32, 24, 5120),
            xxl=(4096, 64, 24, 10240),
        )[size]
        vocab_size = 250112 if variant.startswith("mt5") else 32128

        m = T5Model(vocab_size, dim, n_heads, n_layers, mlp_dim, **kwargs)

        if pretrained:
            location = get_checkpoint_location(variant, size)
            ckpt = load_t5x_checkpoint(location)

            state_dict = {}
            for k, v in ckpt.items():
                if k.endswith("kernel"):
                    v = v.T
                if k.endswith(("query.kernel", "key.kernel")):
                    v *= 64**0.25
                state_dict[_rename_key(k)] = v

            m.load_state_dict(state_dict)

        return m

    @staticmethod
    def get_tokenizer(model_tag: str, cache: str = "tokenizers"):
        import requests
        import sentencepiece as spm

        location = "mc4.250000.100extra" if model_tag.startswith("mt5") else "cc_all.32000.100extra"

        cache_path = Path(cache) / location
        if not cache_path.exists():
            BASE_URL = "https://storage.googleapis.com/t5-data/vocabs"
            cache_path.mkdir(parents=True)

            for filename in ("sentencepiece.model", "sentencepiece.vocab"):
                resp = requests.get(f"{BASE_URL}/{location}/{filename}")
                with open(cache_path / filename, "wb") as f:
                    f.write(resp.content)

        return spm.SentencePieceProcessor(str(cache_path / "sentencepiece.model"))


# TODO: refactor into a separate TextGenerator class
class T5Generator(nn.Module):
    def __init__(self, model_tag: str) -> None:
        super().__init__()
        self.model = T5Model.from_t5x(model_tag, pretrained=True).eval()
        self.tokenizer = T5Model.get_tokenizer(model_tag)

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        device = next(self.parameters()).device

        token_ids = self.tokenizer.Encode(prompt, add_eos=True)
        output_ids = [self.tokenizer.pad_id()]
        eos_id = self.tokenizer.eos_id()

        encoded = self.model.encode(torch.tensor(token_ids, device=device))
        while len(output_ids) < max_tokens:
            # TODO: optimize data allocation
            decoded = self.model.decode(torch.tensor(output_ids, device=device), encoded)
            output_ids.append(decoded.argmax(-1)[-1].item())
            if output_ids[-1] == eos_id:
                break

        return self.tokenizer.Decode(output_ids)


def _rename_key(key: str) -> str:
    return (
        key.replace("token_embedder.embedding", "token_embs.weight")
        .replace("decoder.logits_dense.kernel", "classifier.weight")
        .replace(".encoder_norm.scale", ".norm.weight")
        .replace(".decoder_norm.scale", ".norm.weight")
        .replace(".relpos_bias.rel_embedding", ".attn_bias.bias")
        .replace(".layers_", ".layers.")
        .replace(".pre_attention_layer_norm.scale", ".sa_norm.weight")
        .replace(".pre_self_attention_layer_norm.scale", ".sa_norm.weight")
        .replace(".pre_cross_attention_layer_norm.scale", ".ca_norm.weight")
        .replace(".pre_mlp_layer_norm.scale", ".mlp_norm.weight")
        .replace(".attention.", ".sa.")
        .replace(".self_attention.", ".sa.")
        .replace(".encoder_decoder_attention.", ".ca.")
        .replace(".query.kernel", ".q_proj.weight")
        .replace(".key.kernel", ".k_proj.weight")
        .replace(".value.kernel", ".v_proj.weight")
        .replace(".out.kernel", ".out_proj.weight")
        .replace(".wi_0.kernel", ".0.w.weight")
        .replace(".wi_1.kernel", ".0.v.weight")
        .replace(".wo.kernel", ".2.weight")
    )


# remove trailing / before combining
def url_join(*args: str) -> str:
    return "/".join(x.rstrip("/") for x in args)


def load_t5x_checkpoint(location: str, n_threads: int = 16, cache: str = "checkpoints") -> dict[str, torch.Tensor]:
    cache_path = Path(cache) / location
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    import gzip
    import itertools
    from concurrent.futures import Future, ThreadPoolExecutor

    import msgpack
    import requests
    from requests.adapters import HTTPAdapter

    BASE_URL = "https://storage.googleapis.com/t5-data/pretrained_models/t5x/"
    session = requests.Session()
    session.mount(BASE_URL, HTTPAdapter(pool_maxsize=n_threads))
    pool = ThreadPoolExecutor(n_threads)

    state_dict = {}

    def flatten_dict(dct: dict, prefix: str | None = None) -> None:
        if "kvstore" in dct:
            state_dict[prefix] = pool.submit(load_tensorstore, dct)
            return

        for k, v in dct.items():
            new_prefix = k if prefix is None else f"{prefix}.{k}"
            if isinstance(v, msgpack.ExtType):
                shape, dtype, data = msgpack.unpackb(v.data)
                state_dict[new_prefix] = torch.frombuffer(data, dtype=torch.float32).view(shape)

            elif isinstance(v, dict):
                flatten_dict(v, new_prefix)

            else:
                raise ValueError

    def load_tensorstore(dct: dict) -> torch.Tensor:
        path = dct["kvstore"]["path"]  # sometimes this will have trailing /
        shape = dct["metadata"]["shape"]
        chunk_size = dct["metadata"]["chunks"]

        # can be 1D or 2D tensor
        # shape might not be divisible by chunk_size, but each chunk is guaranteed to be chunk_size.
        n_chunks = [math.ceil(s / cs) for s, cs in zip(shape, chunk_size)]
        out = torch.empty([n * cs for n, cs in zip(n_chunks, chunk_size)])

        for indices in itertools.product(*[range(x) for x in n_chunks]):
            filename = ".".join(str(x) for x in indices)
            data = session.get(url_join(BASE_URL, location, path, filename)).content
            data = torch.frombuffer(gzip.decompress(data), dtype=torch.float32).view(chunk_size)

            slices = tuple(slice(idx * cs, (idx + 1) * cs) for idx, cs in zip(indices, chunk_size))
            out[slices] = data

        return out[tuple(slice(0, s) for s in shape)]  # truncate

    # t5x saves some tensors (e.g. LayerNorm weights) in the `checkpoint` file.
    # for other tensors (e.g. Linear weights), t5x saves them as separate files,
    # using tensorstore format (zarr backend).
    ckpt = msgpack.unpackb(session.get(url_join(BASE_URL, location, "checkpoint")).content)
    flatten_dict(ckpt["optimizer"]["target"])

    for k, v in state_dict.items():
        if isinstance(v, Future):
            state_dict[k] = v.result()

    pool.shutdown()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, cache_path)
    return state_dict


def get_checkpoint_location(variant: str, size: str) -> str:
    if variant in ("t5_1_1", "mt5"):
        prefix = f"{variant}_"
        n_steps = 1000000
    elif variant == "t5_1_1_lm_adapted":
        prefix = "t5_1_1_lm100k_"
        n_steps = 1100000
    elif variant == "mt5_lm_adapted":
        prefix = "mt5_lm_adapted/"
        n_steps = 1100000
    elif variant == "flan_t5":
        prefix = "flan_t5_"
        n_steps = dict(small=1198000, base=1184000, large=1164000, xl=1138000, xxl=1114000)[size]
    else:
        raise ValueError(f"Unsupported {variant=}")

    return f"{prefix}{size}/checkpoint_{n_steps}"
