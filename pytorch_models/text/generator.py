from typing import TYPE_CHECKING

import torch
from torch import nn


if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizerBase


class DecoderGenerator:
    def __init__(self, model: nn.Module, tokenizer: "PreTrainedTokenizerBase") -> None:
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, prompt: str, max_tokens: int = 100, topk: int = 1) -> str:
        device = next(self.model.parameters()).device

        tokens = self.tokenizer.encode(prompt)
        n = len(tokens)

        while len(tokens) - n < max_tokens:
            logits = self.model(torch.tensor(tokens, device=device))[-1]

            if topk == 1:  # greedy decoding
                token = logits.argmax(-1).item()

            else:  # top-k sampling
                logits, indices = logits.topk(topk)
                token = indices[torch.multinomial(logits.softmax(-1), 1).item()]

            tokens.append(token)
            if tokens[-1] == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(tokens)
